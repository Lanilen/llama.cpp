#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

struct llama_moe_cache_params {
    size_t size = 0;
    std::vector<size_t> sizes; // bytes on each device of the model, in its device order, instead of the split of size
    llama_moe_cache_mode mode = LLAMA_MOE_CACHE_MODE_LRU;
    std::string device; // hot spare mode: cache device, empty = the device of each layer
    std::string hot;    // hot spare mode: file with the resident experts
    std::string save;   // hot spare mode: file the resident experts are written to at exit
    int64_t refill = 0; // hot spare mode: generated tokens after a prompt for one rebalance from their scores alone, 0 = none
    bool resize = false; // hot spare mode: release the device memory for large batches (see llama_moe_cache::release)
    std::string predictor;        // hot spare mode: predictor file (see llama-moe-predictor.h), empty = none
    std::string predictor_device; // the device it runs on, empty = CPU

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
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_moe_cache_params & params);
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

    // a context without the cache whose small reserve fits only up to n tokens lowers resize_batch() to n, and the
    // context with the cache follows
    static void limit_resize_batch(int64_t n);

    // hot spare mode with resize (--moe-cache-resize): the device memory of the cache can be released while large batches
    // run, so that their compute buffers can use it, and restored for the small batches of generation
    bool resizes() const;
    void release();
    void restore();

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

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
