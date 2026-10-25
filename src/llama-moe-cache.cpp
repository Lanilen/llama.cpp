#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// SLRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_slru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // probation and protected LRU lists; hits promote slots to the protected half
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head[2] = { -1, -1 };
    int32_t tail[2] = { -1, -1 };
    std::vector<bool> is_protected;
    int32_t n_protected = 0;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head[0] = 0;
        tail[0] = n_slots - 1;
        is_protected.assign(n_slots, false);
        seen.assign(n_expert, 0);
    }

    // move slot s to the most recently used end of a segment
    void move(int32_t s, bool protect) {
        const bool from = is_protected[s];
        if (from == protect && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[protect];
        next[s] = -1;
        if (tail[protect] >= 0) {
            next[tail[protect]] = s;
        } else {
            head[protect] = s;
        }
        tail[protect] = s;
        is_protected[s] = protect;
        n_protected += int(protect) - int(from);
    }

    void touch(int32_t s) {
        move(s, true);
        if (n_protected > n_slots/2) {
            move(head[1], false);
        }
    }

    int32_t victim(int32_t il) const {
        for (int segment = 0; segment < 2; ++segment) {
            for (int32_t s = head[segment]; s >= 0; s = next[s]) {
                const int32_t key = key_of[s];
                // keep every expert selected by this request, including earlier fills
                if (key < 0 || key / n_expert != il || seen[key % n_expert] != seen_gen) {
                    return s;
                }
            }
        }
        GGML_ABORT("no SLRU victim");
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, evicting from probation first
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // promote hits before choosing victims for misses
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            move(s, false);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// hot spare mode: copies the selected experts ids [n_expert_used, n_tokens] and counts them, userdata is the count vector of the layer
static void llama_moe_cache_record(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    // every thread of the CPU backend runs the op, one does the work
    if (ith != 0) {
        return;
    }
    auto & count = *(std::vector<uint32_t> *) userdata;
    int32_t * out = (int32_t *) dst->data;
    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
            const int32_t id = *(const int32_t *) ((const char *) a->data + i0*a->nb[0] + i1*a->nb[1]);
            out[i1*a->ne[0] + i0] = id;
            if (id >= 0 && id < (int32_t) count.size()) {
                count[id]++;
            }
        }
    }
}

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // hot spare mode: host memory of the uploads, pinned if possible so that they are asynchronous
        // the shadows of the device maps come first, then the staging memory of the experts
        ggml_backend_buffer_ptr pinned;
        std::vector<uint8_t> unpinned;
        uint8_t * staging      = nullptr;
        size_t    staging_size = 0;
        size_t    staging_used = 0;
        bool      pending      = false; // uploads were queued since the last synchronization
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_slru slru;

        // hot spare mode
        std::vector<int32_t> key_of_slot; // [n_slots] il*n_expert + expert, -1 if empty
        int32_t n_resident = 0;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks

        // hot spare mode, see llama_moe_cache_hot_spare_maps
        ggml_tensor * miss_map = nullptr;
        ggml_tensor * slot_dev = nullptr;
        ggml_tensor * hit_dev  = nullptr;
        std::vector<int32_t> slot_of; // [n_expert] slot of each expert, -1 if not resident
        int32_t * slot_host = nullptr; // shadows of the device maps
        int32_t * hit_host  = nullptr;
        bool dirty = false;           // the shadows changed since they were uploaded

        std::vector<float>    score;  // [n_expert] decayed number of selections
        std::vector<uint32_t> count;  // [n_expert] selections since the last update, see llama_moe_cache_record
    };

    // hot spare mode statistics
    struct spare_stats {
        size_t tokens = 0;
        size_t hits   = 0;
        size_t misses = 0;
        size_t swaps  = 0;
        size_t bytes  = 0;
        int64_t t_us  = 0;
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    llama_moe_cache_mode mode;
    float  half_life;
    float  margin;
    size_t budget;
    spare_stats spare_small; // ubatches that run on the hot spare
    spare_stats spare_large;
    bool rebalance = false;  // large ubatches were processed since the last refresh
    std::vector<ggml_backend_t> layer_backends; // [n_layer] backend of the device of each layer

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_slru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_moe_cache_params & params) :
            n_expert_used(model.hparams.n_expert_used_max()), mode(params.mode),
            half_life(params.half_life), margin(params.margin), budget(params.budget), layers(model.layers.size()) {
        const size_t size = params.size;
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        layer_backends.assign(model.layers.size(), nullptr);
        for (size_t il = 0; il < model.layers.size(); ++il) {
            for (ggml_backend_t b : backends) {
                if (ggml_backend_get_device(b) == model.dev_layer(il)) {
                    layer_backends[il] = b;
                    break;
                }
            }
        }

        // hot spare mode with a cache device caches all host-resident layers on it, otherwise each layer on its own device
        int32_t spare_id = -1;
        if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE && !params.device.empty()) {
            for (size_t id = 0; id < devices.size() && spare_id < 0; ++id) {
                const char * name = ggml_backend_dev_name(ggml_backend_get_device(devices[id].backend));
                if (params.device == name) {
                    spare_id = (int32_t) id;
                }
            }
            if (spare_id < 0) {
                throw std::runtime_error("MoE cache device " + params.device + " is not a GPU of the context, see -dev");
            }
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            int32_t id = spare_id;
            if (id < 0) {
                const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
                if (it_dev == devices.end()) {
                    continue;
                }
                id = (int32_t) (it_dev - devices.begin());
            }
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE) {
                g.key_of_slot.assign(g.n_slots, -1);
                n_tensors[g.id] += 2*g.layers.size();
            } else {
                g.slru.init(model.layers.size(), n_expert, g.n_slots);
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size();
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE) {
                    l.miss_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    l.slot_dev = ggml_new_tensor_2d(ctx,            GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    l.hit_dev  = ggml_new_tensor_2d(ctx,            GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.miss_map, "moe_cache.miss_map-%d", il);
                    ggml_format_name(l.slot_dev, "moe_cache.slot_map-%d", il);
                    ggml_format_name(l.hit_dev,  "moe_cache.hit_map-%d",  il);
                    l.slot_of.assign(g.ref[0]->ne[2], -1);
                    l.score.assign(g.ref[0]->ne[2], 0.0f);
                    l.count.assign(g.ref[0]->ne[2], 0);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.miss_map), alignment_host);
                    devices[g.id].buf_size += 2*GGML_PAD(ggml_nbytes(l.slot_dev), ggml_backend_buft_get_alignment(devices[g.id].buft));
                } else {
                    l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                    layer_of[l.slot_map] = il;
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE) {
                init_hot_spare(params.hot);
            } else {
                for (group & g : groups) {
                    for (int32_t il : g.layers) {
                        if (layers[il].slot_map != nullptr) {
                            g.slru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                        }
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts%s\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0,
                mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE ? ", hot spare mode" : "");
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%), %d resident\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]), g.n_resident);
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ggml_backend_t layer_backend(int32_t il) const {
        return layer_backends[il];
    }

    // hot spare mode: allocate the upload memory of each device, then upload the resident experts listed in the hot file, most used first
    void init_hot_spare(const std::string & path) {
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            if (!d.buf) {
                continue;
            }
            size_t maps_size = 0;
            size_t expert_size_max = 0;
            for (const group & g : groups) {
                if (g.id != (int32_t) id || g.n_slots == 0) {
                    continue;
                }
                maps_size += g.layers.size()*2*GGML_PAD(g.ref[0]->ne[2]*sizeof(int32_t), 64);
                for (const ggml_tensor * t : g.ref) {
                    expert_size_max = std::max(expert_size_max, (size_t) t->nb[2]);
                }
            }
            d.staging_size = std::max({ budget, 4*expert_size_max, (size_t) 64*1024*1024 });

            ggml_backend_buffer_type_t buft_pinned = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
            uint8_t * base = nullptr;
            if (buft_pinned != nullptr) {
                d.pinned.reset(ggml_backend_buft_alloc_buffer(buft_pinned, maps_size + d.staging_size));
            }
            if (d.pinned) {
                base = (uint8_t *) ggml_backend_buffer_get_base(d.pinned.get());
            } else {
                LLAMA_LOG_WARN("%s: no pinned memory for %s, the MoE cache uploads are synchronous\n", __func__, ggml_backend_name(d.backend));
                d.unpinned.resize(maps_size + d.staging_size);
                base = d.unpinned.data();
            }
            for (const group & g : groups) {
                if (g.id != (int32_t) id || g.n_slots == 0) {
                    continue;
                }
                for (int32_t il : g.layers) {
                    layer & l = layers[il];
                    const size_t n = GGML_PAD(l.slot_of.size()*sizeof(int32_t), 64);
                    l.slot_host = (int32_t *) base;
                    l.hit_host  = (int32_t *) (base + n);
                    base += 2*n;
                }
            }
            d.staging = base;
        }

        for (layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (size_t e = 0; e < l.slot_of.size(); ++e) {
                set_resident(l, (int32_t) e, -1);
            }
        }

        if (!path.empty()) {
            std::ifstream f(path);
            if (!f) {
                throw std::runtime_error("failed to open the MoE cache hot file " + path);
            }
            std::string line;
            while (std::getline(f, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::istringstream ss(line);
                int32_t il;
                int32_t e;
                if (!(ss >> il >> e)) {
                    throw std::runtime_error("invalid line in the MoE cache hot file " + path + ": " + line);
                }
                if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
                    continue;
                }
                layer & l = layers[il];
                group & g = groups[l.ig];
                if (e < 0 || e >= (int32_t) l.slot_of.size() || l.slot_of[e] >= 0 || g.n_resident >= g.n_slots) {
                    continue;
                }
                const int32_t s = g.n_resident++;
                g.key_of_slot[s] = il*(int32_t) l.slot_of.size() + e;
                set_resident(l, e, s);
                upload_expert(g, l, e, s);
            }
        }

        flush_maps();
        for (device & d : devices) {
            if (d.pending) {
                ggml_backend_synchronize(d.backend);
                d.pending = false;
            }
            d.staging_used = 0;
        }
    }

    // hot spare mode: make expert e of layer l resident at slot s, or not resident if s < 0
    // the CPU branch sees the change at the next ubatch, the device maps when the shadows are flushed
    void set_resident(layer & l, int32_t e, int32_t s) {
        ((int32_t *) l.miss_map->data)[e] = s < 0 ? e : -1;
        l.slot_host[e] = s;
        l.hit_host[e]  = s < 0 ? -1 : e;
        l.slot_of[e]   = s;
        l.dirty = true;
    }

    // hot spare mode: copy expert e of layer l to the staging memory and queue its upload to slot s
    // the device runs its queue in order, so the next ubatch reads the whole expert
    void upload_expert(const group & g, const layer & l, int32_t e, int32_t s) {
        device & d = devices[g.id];
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const size_t expert_size = l.experts[ip]->nb[2];
            if (d.staging_used + expert_size > d.staging_size) {
                ggml_backend_synchronize(d.backend);
                d.staging_used = 0;
            }
            uint8_t * src = d.staging + d.staging_used;
            memcpy(src, (const uint8_t *) l.experts[ip]->data + e*expert_size, expert_size);
            ggml_backend_tensor_set_async(d.backend, g.banks[ip], src, s*expert_size, expert_size);
            d.staging_used += GGML_PAD(expert_size, 64);
            d.pending = true;
        }
    }

    // hot spare mode: queue the uploads of the changed device maps, after the experts they point to
    void flush_maps() {
        for (layer & l : layers) {
            if (l.ig < 0 || !l.dirty) {
                continue;
            }
            device & d = devices[groups[l.ig].id];
            ggml_backend_tensor_set_async(d.backend, l.slot_dev, l.slot_host, 0, ggml_nbytes(l.slot_dev));
            ggml_backend_tensor_set_async(d.backend, l.hit_dev,  l.hit_host,  0, ggml_nbytes(l.hit_dev));
            l.dirty = false;
            d.pending = true;
        }
    }

    ggml_tensor * record(ggml_context * ctx, int32_t il, ggml_tensor * ids) {
        if (mode != LLAMA_MOE_CACHE_MODE_HOT_SPARE || il < 0 || il >= (int32_t) layers.size() || layers[il].miss_map == nullptr) {
            return nullptr;
        }
        return ggml_map_custom1(ctx, ids, llama_moe_cache_record, 1, &layers[il].count);
    }

    void update(int64_t n_tokens) {
        if (mode != LLAMA_MOE_CACHE_MODE_HOT_SPARE || n_tokens <= 0) {
            return;
        }
        const int64_t t_start_us = ggml_time_us();

        // the uploads of the previous update are done, so the staging memory can be reused
        for (device & d : devices) {
            if (d.pending) {
                ggml_backend_synchronize(d.backend);
                d.pending = false;
            }
            d.staging_used = 0;
        }

        // the selections of this ubatch ran on the hot spare if they were resident, for small ubatches
        const bool small = n_tokens <= max_batch;
        spare_stats & st = small ? spare_small : spare_large;
        st.tokens += n_tokens;

        const float decay = half_life > 0.0f ? powf(0.5f, (float) n_tokens/half_life) : 1.0f;
        for (layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (size_t e = 0; e < l.count.size(); ++e) {
                const uint32_t c = l.count[e];
                if (c > 0) {
                    (l.slot_of[e] >= 0 ? st.hits : st.misses) += c;
                    l.count[e] = 0;
                }
                l.score[e] = l.score[e]*decay + c;
            }
        }

        // large ubatches (the prompt) only update the scores, the first small ubatch after them rebalances the cache without a budget
        if (half_life > 0.0f) {
            if (!small) {
                rebalance = true;
            } else {
                refresh(rebalance ? SIZE_MAX : budget*n_tokens, st);
                rebalance = false;
            }
        }

        st.t_us += ggml_time_us() - t_start_us;
    }

    // hot spare mode: replace the weakest resident experts by stronger ones, empty slots first
    // the budget counts the uploads into occupied slots, the empty ones are filled for free
    void refresh(size_t budget_bytes, spare_stats & st) {
        struct swap {
            float   gain;
            int32_t ig;
            int32_t key; // il*n_expert + expert of the new expert
            int32_t slot;
        };
        std::vector<swap> swaps;
        std::vector<std::pair<float, int32_t>> best;  // score, key of the experts that are not resident
        std::vector<std::pair<float, int32_t>> worst; // score, slot of the resident experts, -1 for the empty slots

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n_expert = g.ref[0]->ne[2];
            size_t expert_size = 0;
            for (const ggml_tensor * t : g.ref) {
                expert_size += t->nb[2];
            }

            best.clear();
            worst.clear();
            int32_t n_empty = 0;
            for (int32_t il : g.layers) {
                const layer & l = layers[il];
                for (int32_t e = 0; e < n_expert; ++e) {
                    if (l.slot_of[e] < 0 && l.score[e] > 0.0f) {
                        best.emplace_back(l.score[e], il*n_expert + e);
                    }
                }
            }
            for (int32_t s = 0; s < g.n_slots; ++s) {
                const int32_t key = g.key_of_slot[s];
                if (key < 0) {
                    n_empty++;
                    worst.emplace_back(-1.0f, s);
                } else {
                    worst.emplace_back(layers[key / n_expert].score[key % n_expert], s);
                }
            }

            const size_t n = std::min({ best.size(), worst.size(), budget_bytes == SIZE_MAX ? worst.size() : (size_t) n_empty + budget_bytes/expert_size });
            if (n == 0) {
                continue;
            }
            std::partial_sort(best.begin(),  best.begin()  + n, best.end(),  [](const auto & a, const auto & b) { return a.first > b.first; });
            std::partial_sort(worst.begin(), worst.begin() + n, worst.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
            for (size_t i = 0; i < n; ++i) {
                const bool empty = worst[i].first < 0.0f;
                if (!empty && best[i].first <= worst[i].first + margin) {
                    break;
                }
                swaps.push_back({ empty ? INFINITY : best[i].first - worst[i].first, (int32_t) ig, best[i].second, worst[i].second });
            }
        }

        std::sort(swaps.begin(), swaps.end(), [](const swap & a, const swap & b) { return a.gain > b.gain; });

        size_t spent = 0;
        for (const swap & sw : swaps) {
            group & g = groups[sw.ig];
            const int32_t n_expert = g.ref[0]->ne[2];
            size_t expert_size = 0;
            for (const ggml_tensor * t : g.ref) {
                expert_size += t->nb[2];
            }
            const int32_t old_key = g.key_of_slot[sw.slot];
            if (old_key >= 0) {
                if (budget_bytes != SIZE_MAX && spent + expert_size > budget_bytes) {
                    continue;
                }
                spent += expert_size;
                set_resident(layers[old_key / n_expert], old_key % n_expert, -1);
            } else {
                g.n_resident++;
            }
            layer & l = layers[sw.key / n_expert];
            g.key_of_slot[sw.slot] = sw.key;
            set_resident(l, sw.key % n_expert, sw.slot);
            upload_expert(g, l, sw.key % n_expert, sw.slot);
            st.swaps++;
            st.bytes += expert_size;
        }

        flush_maps();
    }

    bool get_hot_spare_maps(int32_t il, int64_t n_tokens, llama_moe_cache_hot_spare_maps & maps) const {
        if (mode != LLAMA_MOE_CACHE_MODE_HOT_SPARE || il < 0 || il >= (int32_t) layers.size() || layers[il].miss_map == nullptr) {
            return false;
        }
        // large batches use most experts of a layer, they keep running all host experts with the scheduler's offload
        if (n_tokens == 0 || n_tokens > max_batch) {
            return false;
        }
        const layer & l = layers[il];
        maps = { l.miss_map, l.slot_dev, l.hit_dev };
        return true;
    }

    ~impl() {
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE || il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE || it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.slru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        size_t n_hit = 0;
        if (!g.slru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);

        // large ubatches do not run on the hot spare, only their swaps count
        auto log_spare = [](const char * name, const spare_stats & st, bool hits) {
            if (st.tokens == 0) {
                return;
            }
            const size_t n = st.hits + st.misses;
            LLAMA_LOG_INFO("llama_moe_cache: hot spare, %s: %zu tokens,%s %zu swaps, uploaded = %.2f MiB (%.2f MiB/token), update = %.3f ms/token\n",
                name, st.tokens, hits ? format(" hit rate = %.2f%%,", n > 0 ? 100.0*st.hits/n : 0.0).c_str() : "",
                st.swaps, st.bytes/1024.0/1024.0, st.bytes/1024.0/1024.0/st.tokens, st.t_us/1000.0/st.tokens);
        };
        log_spare("ubatch <= 32", spare_small, true);
        log_spare("ubatch  > 32", spare_large, false);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, const llama_moe_cache_params & params) :
    pimpl(new impl(model, backends, bufts, params)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_backend_t llama_moe_cache::layer_backend(int32_t il) const {
    return pimpl->layer_backend(il);
}

bool llama_moe_cache::get_hot_spare_maps(int32_t il, int64_t n_tokens, llama_moe_cache_hot_spare_maps & maps) const {
    return pimpl->get_hot_spare_maps(il, n_tokens, maps);
}

ggml_tensor * llama_moe_cache::record(ggml_context * ctx, int32_t il, ggml_tensor * ids) const {
    return pimpl->record(ctx, il, ids);
}

void llama_moe_cache::update(int64_t n_tokens) {
    pimpl->update(n_tokens);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
