#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llm_graph_context;

// the experts of a MoE layer, as llm_graph_context::build_moe_ffn passes them to the MoE cache
struct llama_moe_cache_layer {
    int32_t       il;
    ggml_tensor * up_exps;
    ggml_tensor * gate_exps;
    ggml_tensor * down_exps;
    ggml_tensor * gate_up_exps;
    bool          weight_before_ffn; // the router weights scale the expert input
};

// the lookup of the selected experts of a layer in the MoE cache, see llama_moe_cache::build_lookup
struct llama_moe_cache_lookup {
    int32_t       il      = -1;
    ggml_tensor * slots   = nullptr; // [n_expert_used, n_tokens] slots of the selected experts in the MoE cache, nullptr if the layer does not read the cache
    ggml_tensor * cpu_ids = nullptr; // [n_expert_used, n_tokens] misses computed from the host experts, -1 for the other experts, or nullptr
};

// builds the experts selected by mm_ids up to the down projection, from the MoE cache if cached: [n_embd, n_expert_used, n_tokens]
// cur is the expert input and ids are the selected experts, for the biases and scales (see llm_graph_context::build_moe_ffn)
using llama_moe_cache_build_experts_t = ggml_tensor * (*)(ggml_tensor * cur, ggml_tensor * ids, ggml_tensor * mm_ids, bool cached, void * user_data);

struct llama_moe_cache_params {
    size_t size = 0;
    std::vector<size_t> sizes; // bytes on each device of the model, in its device order, instead of the split of size
    llama_moe_cache_mode mode = LLAMA_MOE_CACHE_MODE_LRU;
    std::string device; // hot spare mode: cache device, empty = the device of each layer
    std::string hot;    // hot spare mode: file with the resident experts
    std::string save;   // hot spare mode: file the resident experts are written to at exit
    int64_t refill = -1; // hot spare mode: generated tokens after a prompt for one rebalance from their scores alone, 0 = none, < 0 = 128, or 0 with a predictor
    bool resize = false; // hot spare mode: release the device memory for large batches (see llama_moe_cache::release)
    std::string predictor;        // hot spare mode: predictor file (see llama-moe-predictor.h), empty = none
    std::string predictor_device; // the device it runs on, empty = CPU
    std::string predictor_start;  // hot spare mode with a predictor: start predictor file (role start), empty = none
    int32_t answer_cap = -1;      // without a start predictor: most swaps at an answer's start, 0 = all, < 0 = LLAMA_MOE_PREDICTOR_ANSWER_CAP or 0
    float   answer_gain = -1.0f;  // an answer's start only makes the swaps that bring this many more predicted selections, 0 = all, < 0 = LLAMA_MOE_PREDICTOR_ANSWER_GAIN or 0
    int32_t jit = 0; // hot spare mode: in ubatches above 32 tokens, each layer uploads the experts that at least this many of its tokens select and are not resident, before it runs, 0 = none
    float cpu_share = 0.0f; // lru mode: share of the misses computed on the CPU instead of uploaded, 0 = all uploaded

    // hot spare mode: refresh of the resident experts between ubatches
    float  half_life = 0.0f; // tokens, 0 = the resident experts never change
    float  margin    = 0.5f; // a new expert replaces a resident one if its decayed score is higher by this much
    size_t budget    = 0;    // bytes uploaded per token, unlimited for empty slots and after the prompt
};

// hot spare mode: per layer maps from the selected experts to the ids of the CPU branch and the slots of the cache branch
// -1 skips the expert in that branch (see ggml_mul_mat_id)
struct llama_moe_cache_hot_spare_maps {
    ggml_tensor * miss = nullptr; // I32 [1, n_expert] in host memory: e if expert e is not resident, else -1
    ggml_tensor * slot = nullptr; // I32 [1, n_expert] on the cache device: the slot of e if it is resident, else -1
    ggml_tensor * hit  = nullptr; // I32 [1, n_expert] on the cache device: e if it is resident, else -1
};

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// lru mode: each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
// hot spare mode: one device keeps a set of resident experts, they run there while the CPU runs the other selected experts
class llama_moe_cache {
public:
    // backends are all the backends of the context
    // lru mode: each GPU gets its own cache of the given size for the layers assigned to it
    // hot spare mode: the cache device gets the whole size for all host-resident layers, without a cache device it is like lru mode
    // no_alloc: only the memory use is measured (dummy buffers, see llama_context::memory_breakdown)
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_moe_cache_params & params);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the backend of the device layer il is assigned to
    ggml_backend_t layer_backend(int32_t il) const;

    // hot spare mode: the maps of layer il, if its experts can run on the hot spare for n_tokens tokens, false otherwise
    bool get_hot_spare_maps(int32_t il, int64_t n_tokens, llama_moe_cache_hot_spare_maps & maps) const;

    // hot spare mode: a copy of the selected experts ids of layer il that counts them when it is computed, nullptr if layer il is not cached
    ggml_tensor * record(ggml_context * ctx, int32_t il, ggml_tensor * ids) const;

    // hot spare mode: after a ubatch of n_tokens tokens, update the scores and refresh the resident experts
    // pos0: position of its first token (0 starts a new conversation), -1 = unknown; gen: 1 if it is generation (it
    // outputs all its tokens), 0 if it is a prompt, -1 = unknown (up to 4 tokens is generation)
    void update(int64_t n_tokens, int64_t pos0 = -1, int gen = -1);

    // hot spare mode with a predictor, before a ubatch is computed (pos0 and gen as for update): with a start
    // predictor, a turn's first prompt ubatch uploads the experts it predicts from the turn's text while it computes
    void before_ubatch(const llama_token * tokens, int64_t n_tokens, int64_t pos0 = -1, int gen = -1);

    // hot spare mode with a predictor: the layers whose input it reads at the newest token of each ubatch, empty = none
    const std::vector<int32_t> & predictor_layers() const;

    // hot spare mode with a predictor: the input of layer il at the newest token of the ubatch about to be passed to update
    void set_hidden(int32_t il, const float * data, size_t n);

    // hot spare mode: ubatches of more tokens do not run on the hot spare, unless set_hot_spare_batch raised the limit
    static int64_t max_hot_spare_batch();

    // hot spare mode: ubatches of up to n tokens (at least max_hot_spare_batch()) run on the hot spare; with resize, the
    // batches the cache stays in device memory for, whose compute buffers fit next to it
    void set_hot_spare_batch(int64_t n);

    // the hot spare batch of the last context with resize that set one, 0 = none: contexts without the cache that share
    // its device memory (an MTP draft) keep their small reserve for the same batches
    static int64_t resize_batch();

    // a context with the cache offers n to the contexts without it, before its own small reserve grows to what they pick
    static void set_resize_batch(int64_t n);

    // a context without the cache whose small reserve fits only up to n tokens lowers resize_batch() to n, and the
    // context with the cache follows
    static void limit_resize_batch(int64_t n);

    // hot spare mode with resize (--moe-cache-resize): the device memory of the cache can be released while large batches
    // run, so that their compute buffers can use it, and restored for the small batches of generation
    bool resizes() const;
    void release();
    void restore();

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // lru mode: the graph of a MoE layer, built by llm_graph_context::build_moe_ffn in two steps:

    // 1. the lookup of the selected experts in the cache, built after the expert input and before the expert matmuls
    //    returns no slots if the layer does not read its experts from the cache for this ubatch
    llama_moe_cache_lookup build_lookup(const llm_graph_context & g, const llama_moe_cache_layer & layer, ggml_tensor * selected_experts) const;

    // 2. the experts of a layer with slots, from the cache and from the host experts: [n_embd, n_expert_used, n_tokens]
    // calls build synchronously; does not retain user_data
    ggml_tensor * build_experts(const llm_graph_context & g, const llama_moe_cache_lookup & lookup,
            ggml_tensor * cur, ggml_tensor * selected_experts, llama_moe_cache_build_experts_t build, void * user_data) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
