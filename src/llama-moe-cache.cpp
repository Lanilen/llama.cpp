#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <tuple>
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

// the largest number of experts (dim 2) a bank of each tensor in ref can hold on dev, at most limit: a bank is one tensor,
// and a backend may refuse large ones (Vulkan without 64-bit indexing: maxStorageBufferRange, 4 GiB on RADV)
static int64_t llama_moe_cache_max_bank_experts(ggml_backend_dev_t dev, const std::vector<ggml_tensor *> & ref, int64_t limit) {
    ggml_init_params params = {
        /*.mem_size   =*/ ref.size()*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    std::vector<ggml_tensor *> probes;
    for (const ggml_tensor * t : ref) {
        probes.push_back(ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], 1));
    }
    auto fits = [&](int64_t n) {
        for (ggml_tensor * x : probes) {
            x->ne[2] = n;
            if (!ggml_backend_dev_supports_op(dev, x)) {
                return false;
            }
        }
        return true;
    };
    if (fits(limit)) {
        return limit;
    }
    if (!fits(1)) {
        return 0;
    }
    int64_t lo = 1;
    int64_t hi = limit;
    while (hi - lo > 1) {
        const int64_t mid = lo + (hi - lo)/2;
        (fits(mid) ? lo : hi) = mid;
    }
    return lo;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // hot spare mode: copies the experts to the staging memory on a thread of its own,
    // their background uploads are submitted once the copies are done
    struct copy_worker {
        struct job {
            uint8_t       * dst;
            const uint8_t * src;
            size_t          n;
        };

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<job> queue;
        size_t head = 0; // the next job of the queue to take
        int    busy = 0; // jobs being copied
        bool   stop = false;
        std::atomic<int64_t>  t_us  { 0 }; // copying, summed over the threads
        std::atomic<uint64_t> bytes { 0 };
        std::vector<std::thread> threads;

        // the threads start in the body, after the members they use are built
        explicit copy_worker(int n_threads) {
            for (int i = 0; i < std::max(n_threads, 1); ++i) {
                threads.emplace_back([this] { run(); });
            }
        }

        ~copy_worker() {
            {
                std::lock_guard<std::mutex> lock(mutex);
                stop = true;
            }
            cv.notify_all();
            for (std::thread & t : threads) {
                t.join();
            }
        }

        // each thread takes one job at a time, so the copies of a batch are spread over the threads
        void run() {
            std::unique_lock<std::mutex> lock(mutex);
            for (;;) {
                cv.wait(lock, [this] { return stop || head < queue.size(); });
                if (head >= queue.size()) {
                    return;
                }
                const job j = queue[head++];
                if (head == queue.size()) {
                    queue.clear();
                    head = 0;
                }
                busy++;
                lock.unlock();
                const int64_t t_start_us = ggml_time_us();
                memcpy(j.dst, j.src, j.n);
                t_us  += ggml_time_us() - t_start_us;
                bytes += j.n;
                lock.lock();
                if (--busy == 0 && head >= queue.size()) {
                    cv.notify_all();
                }
            }
        }

        void post(std::vector<job> & jobs) {
            if (jobs.empty()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                queue.insert(queue.end(), jobs.begin(), jobs.end());
            }
            jobs.clear();
            cv.notify_all();
        }

        void wait() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this] { return head >= queue.size() && busy == 0; });
        }
    };

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

        // background uploads, if the backend has them (see ggml_backend_upload_bg_t)
        // the staging memory is used in two halves, a half is reused once its uploads are done
        ggml_backend_upload_bg_t        upload_bg        = nullptr;
        ggml_backend_upload_bg_submit_t upload_bg_submit = nullptr;
        ggml_backend_upload_bg_done_t   upload_bg_done   = nullptr;
        int      half        = 0;
        uint64_t half_ticket[2] = { 0, 0 };
        bool     bg_recorded = false;    // background uploads were recorded since the last submission
        std::vector<copy_worker::job> jobs; // the copies of the recorded uploads, for the copy worker
        bool     submit_waiting = false; // the recorded uploads are submitted at the next update, after their copies
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        int32_t max_slots = INT32_MAX;    // the largest bank the device takes, less the extra slot
        moe_cache_slru slru;

        // hot spare mode
        std::vector<int32_t> key_of_slot; // [n_slots] il*n_expert + expert, -1 if empty
        std::vector<uint8_t> slot_pending; // [n_slots] the expert of the slot is being uploaded
        int32_t n_resident = 0;
    };

    // hot spare mode: an expert being uploaded in the background, it becomes resident when its ticket is done
    struct pending_upload {
        int32_t  ig;
        int32_t  key;
        int32_t  slot;
        uint64_t ticket; // 0 until its batch is submitted
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
        std::vector<float>    score_gen; // [n_expert] the same for the tokens generated since the last prompt, until the refill
        std::vector<uint32_t> count;  // [n_expert] selections since the last update, see llama_moe_cache_record
        std::vector<uint8_t>  uploading; // [n_expert] the expert is being uploaded in the background
    };

    // hot spare mode statistics
    struct spare_stats {
        size_t tokens = 0;
        size_t hits   = 0;
        size_t misses = 0;
        size_t swaps  = 0;
        size_t bytes  = 0;
        int64_t t_us  = 0;
        int64_t t_copy_us = 0; // of t_us, copying the experts to the staging memory
        int64_t t_sync_us = 0; // of t_us, waiting for the devices to finish the ubatch
        int64_t t_wait_us = 0; // of t_us, waiting for the copy worker
        int64_t t_wait_half_us = 0; // of t_wait_us, in a refresh whose uploads filled a half of the staging memory
        int64_t t_wait_next_us = 0; // of t_wait_us, at the next update, for the copies of the previous one
        int64_t t_dma_us  = 0; // of t_us, waiting for the uploads of a half before reusing it
        int64_t t_full_us = 0; // of t_us, full rebalances (after a prompt, the refill)
        size_t  n_full    = 0;
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
    int64_t hot_batch = max_batch; // ubatches of up to this many tokens run on the hot spare, more with resize (set_hot_spare_batch)

    int32_t n_expert_used;
    llama_moe_cache_mode mode;
    float  half_life;
    float  margin;
    size_t budget;
    spare_stats spare_small; // ubatches that run on the hot spare
    spare_stats spare_large;
    spare_stats * spare_cur = nullptr; // the stats of the running update
    spare_stats spare_restore; // the restores of the cache after large batches (t_us: their time)
    std::unique_ptr<copy_worker> worker; // LLAMA_MOE_CACHE_ASYNC=0 copies on the generation thread
    bool rebalance = false;  // large ubatches were processed since the last refresh
    bool warm      = false;  // the resident experts were loaded from a file
    int64_t refill_tokens = 0; // generated tokens after which the cache is rebalanced from their scores, 0 = never
    int64_t refill_left   = 0; // generated tokens until that rebalance, 0 = none pending
    std::string save_path;   // the resident experts are written there at exit

    // hot spare mode with resize: the device memory of the cache is released while large batches run
    bool resize   = false;
    bool released = false;
    std::string hot_pending; // the hot file, loaded at the first restore
    size_t n_restores    = 0;
    size_t restore_bytes = 0;
    std::vector<pending_upload> pending_uploads;
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
        if (mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE && !model.hparams.no_alloc) {
            save_path = params.save;
            resize    = params.resize;
        } else if (params.resize && mode != LLAMA_MOE_CACHE_MODE_HOT_SPARE) {
            LLAMA_LOG_WARN("%s: --moe-cache-resize needs --moe-cache-mode hot-spare, ignored\n", __func__);
        }
        refill_tokens = std::max<int64_t>(0, params.refill);
        refill_left   = refill_tokens; // also for a first prompt too short to count as one
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

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory,
        // or each device has the size given for it
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (!params.sizes.empty()) {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                const size_t i = it - model.devices.begin();
                d.split = i < params.sizes.size() ? (double) params.sizes[i] : 0.0;
            } else if (split_by_free) {
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
        if (!params.sizes.empty()) {
            if (params.sizes.size() > model.devices.size()) {
                LLAMA_LOG_WARN("%s: %zu MoE cache sizes for %zu devices, the extra sizes are ignored\n", __func__, params.sizes.size(), model.devices.size());
            }
            split_sum = (double) size; // the sum of the sizes, so each device gets its own
        } else if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // a group whose budget needs banks larger than its device takes is split by layers, each part keeping the share of
        // the budget of its experts; the experts of a layer can then only use the slots of its part
        {
            std::vector<group> fitted;
            for (group & g : groups) {
                const device & d = devices[g.id];
                const int32_t n_expert = g.ref[0]->ne[2];
                const size_t  budget   = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
                size_t slot_bytes = 0;
                for (const ggml_tensor * t : g.ref) {
                    slot_bytes += t->nb[2];
                }
                const int64_t wanted = std::min<int64_t>(budget/std::max<size_t>(slot_bytes, 1), (int64_t) g.layers.size()*n_expert);
                const int64_t max_slots = llama_moe_cache_max_bank_experts(ggml_backend_get_device(d.backend), g.ref, wanted + 1) - 1;
                if (max_slots >= wanted || max_slots < 1) {
                    g.max_slots = (int32_t) std::max<int64_t>(max_slots, 0);
                    fitted.push_back(std::move(g));
                    continue;
                }
                const size_t n_parts = std::min(g.layers.size(), (size_t) ((wanted + max_slots - 1)/max_slots));
                LLAMA_LOG_INFO("%s: %s: banks of %lld experts are larger than the device takes (%lld), the %zu layers are split into %zu groups\n",
                    __func__, ggml_backend_dev_name(ggml_backend_get_device(d.backend)), (long long) wanted, (long long) max_slots, g.layers.size(), n_parts);
                for (size_t ip = 0; ip < n_parts; ++ip) {
                    group part;
                    part.id        = g.id;
                    part.max_slots = (int32_t) max_slots;
                    for (size_t i = ip*g.layers.size()/n_parts; i < (ip + 1)*g.layers.size()/n_parts; ++i) {
                        const int32_t il = g.layers[i];
                        auto experts = llama_moe_cache_layer_experts(model.layers[il]);
                        if (part.layers.empty()) {
                            part.ref = experts;
                        }
                        part.layers.push_back(il);
                        for (const ggml_tensor * t : experts) {
                            part.host_bytes += ggml_nbytes(t);
                        }
                    }
                    fitted.push_back(std::move(part));
                }
            }
            groups = std::move(fitted);
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = std::min<int32_t>(g.layers.size()*n_expert, g.max_slots);
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
                g.slot_pending.assign(g.n_slots, 0);
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
                    l.score_gen.assign(g.ref[0]->ne[2], 0.0f);
                    l.count.assign(g.ref[0]->ne[2], 0);
                    l.uploading.assign(g.ref[0]->ne[2], 0);
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
            // a half holds about 8 tokens of the budget, so the refresh of a decode step (the budget times its tokens)
            // rarely fills one and waits for its copies; LLAMA_MOE_CACHE_STAGING_MIB sets the whole size
            d.staging_size = 2*std::max({ 8*budget, 4*expert_size_max, (size_t) 64*1024*1024 });
            if (const char * env = getenv("LLAMA_MOE_CACHE_STAGING_MIB")) {
                d.staging_size = std::max((size_t) 2*4*expert_size_max, (size_t) (atof(env)*1024*1024));
            }
            LLAMA_LOG_INFO("%s: %s: %.0f MiB of staging memory for the expert uploads\n", __func__, ggml_backend_name(d.backend),
                d.staging_size/1024.0/1024.0);

            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            auto dev_pinned_buffer_type = (ggml_backend_dev_pinned_buffer_type_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_pinned_buffer_type");
            d.upload_bg        = (ggml_backend_upload_bg_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_bg");
            d.upload_bg_submit = (ggml_backend_upload_bg_submit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_bg_submit");
            d.upload_bg_done   = (ggml_backend_upload_bg_done_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_bg_done");
            // LLAMA_MOE_CACHE_BG=0 keeps the uploads on the compute queue, to compare
            const char * env_bg = getenv("LLAMA_MOE_CACHE_BG");
            if (!d.upload_bg || !d.upload_bg_submit || !d.upload_bg_done || (env_bg != nullptr && atoi(env_bg) == 0)) {
                d.upload_bg = nullptr;
            }

            ggml_backend_buffer_type_t buft_pinned = dev_pinned_buffer_type ? dev_pinned_buffer_type(dev) : ggml_backend_dev_host_buffer_type(dev);
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

            LLAMA_LOG_INFO("%s: %10s hot spare uploads: %s, %s\n", __func__, ggml_backend_name(d.backend),
                d.pinned ? ggml_backend_buft_name(ggml_backend_buffer_get_type(d.pinned.get())) : "unpinned memory",
                d.upload_bg ? "in the background" : "on the compute queue");
        }

        for (layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (size_t e = 0; e < l.slot_of.size(); ++e) {
                set_resident(l, (int32_t) e, -1);
            }
        }

        // with resize, the device memory is released before the first batch, the hot file is loaded when it is restored
        if (resize) {
            hot_pending = path;
        } else if (!path.empty()) {
            load_hot(path);
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

    // hot spare mode: upload the resident experts listed in the hot file, most used first, into the free slots
    void load_hot(const std::string & path) {
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
            upload_expert(g, l, e, s, false);
            warm = true;
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
    // in the background if the device has background uploads and bg is set: then it returns true and the expert becomes resident when its batch is done
    // otherwise on the compute queue: the device runs it in order, so the next ubatch reads the whole expert
    bool upload_expert(const group & g, const layer & l, int32_t e, int32_t s, bool bg) {
        device & d = devices[g.id];
        bg = bg && d.upload_bg != nullptr;
        const size_t half_size = d.staging_size/2;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const size_t expert_size = l.experts[ip]->nb[2];
            if (bg) {
                // a full half is submitted, the other half is reused once its uploads are done
                if (d.staging_used + expert_size > half_size) {
                    const int64_t t0_us = ggml_time_us();
                    submit_bg(d);
                    const int64_t t1_us = ggml_time_us();
                    d.half ^= 1;
                    d.upload_bg_done(d.backend, d.half_ticket[d.half], true);
                    d.staging_used = 0;
                    if (spare_cur != nullptr) {
                        spare_cur->t_wait_half_us += t1_us - t0_us;
                        spare_cur->t_dma_us       += ggml_time_us() - t1_us;
                    }
                }
                uint8_t * src = d.staging + d.half*half_size + d.staging_used;
                const uint8_t * expert = (const uint8_t *) l.experts[ip]->data + e*expert_size;
                // the upload is recorded now and reads the staging memory when it runs, after its submission
                if (d.upload_bg(d.backend, g.banks[ip], src, s*expert_size, expert_size)) {
                    if (get_worker() != nullptr) {
                        d.jobs.push_back({ src, expert, expert_size });
                    } else {
                        copy_to_staging(src, expert, expert_size);
                    }
                    d.staging_used += GGML_PAD(expert_size, 64);
                    d.bg_recorded = true;
                    continue;
                }
                // the backend cannot upload this one in the background, so none of this device
                d.upload_bg = nullptr;
                bg = false;
                GGML_ASSERT(ip == 0 && "background upload failed after the first tensor of an expert");
                if (worker) {
                    worker->post(d.jobs);
                    worker->wait();
                }
                ggml_backend_synchronize(d.backend);
                d.staging_used = 0;
            }
            if (d.staging_used + expert_size > d.staging_size) {
                ggml_backend_synchronize(d.backend);
                d.staging_used = 0;
            }
            uint8_t * src = d.staging + d.staging_used;
            copy_to_staging(src, (const uint8_t *) l.experts[ip]->data + e*expert_size, expert_size);
            ggml_backend_tensor_set_async(d.backend, g.banks[ip], src, s*expert_size, expert_size);
            d.staging_used += GGML_PAD(expert_size, 64);
            d.pending = true;
        }
        return bg;
    }

    void copy_to_staging(uint8_t * dst, const uint8_t * src, size_t size) {
        const int64_t t_start_us = ggml_time_us();
        memcpy(dst, src, size);
        if (spare_cur != nullptr) {
            spare_cur->t_copy_us += ggml_time_us() - t_start_us;
        }
    }

    copy_worker * get_worker() {
        static const bool async = [] {
            const char * env = getenv("LLAMA_MOE_CACHE_ASYNC");
            return env == nullptr || atoi(env) != 0;
        }();
        // LLAMA_MOE_CACHE_COPY_THREADS=N spreads the copies to the staging memory over N threads
        static const int n_threads = [] {
            const char * env = getenv("LLAMA_MOE_CACHE_COPY_THREADS");
            return env != nullptr ? std::max(1, atoi(env)) : 1;
        }();
        if (async && !worker) {
            worker = std::make_unique<copy_worker>(n_threads);
        }
        return worker.get();
    }

    // hot spare mode: submit the recorded background uploads of device d and give their ticket to the uploads that wait for one
    // with the copy worker and defer, the copies run in the background and the uploads are submitted by finish_submits
    void submit_bg(device & d, bool defer = false) {
        if (!d.bg_recorded) {
            return;
        }
        if (worker) {
            worker->post(d.jobs);
            if (defer) {
                d.submit_waiting = true;
                return;
            }
            const int64_t t_start_us = ggml_time_us();
            worker->wait();
            if (spare_cur != nullptr) {
                spare_cur->t_wait_us += ggml_time_us() - t_start_us;
            }
        }
        d.submit_waiting = false;
        const uint64_t ticket = d.upload_bg_submit(d.backend);
        d.half_ticket[d.half] = ticket;
        d.bg_recorded = false;
        for (pending_upload & pu : pending_uploads) {
            if (pu.ticket == 0 && &devices[groups[pu.ig].id] == &d) {
                pu.ticket = ticket;
            }
        }
    }

    // hot spare mode: submit the uploads whose copies were left to the copy worker
    void finish_submits() {
        const int64_t t_start_us = ggml_time_us();
        for (device & d : devices) {
            if (d.submit_waiting) {
                submit_bg(d);
            }
        }
        if (spare_cur != nullptr) {
            spare_cur->t_wait_next_us += ggml_time_us() - t_start_us;
        }
    }

    // hot spare mode: the experts whose background uploads are done become resident
    void activate_uploads() {
        size_t n_keep = 0;
        for (size_t i = 0; i < pending_uploads.size(); ++i) {
            const pending_upload pu = pending_uploads[i];
            group & g = groups[pu.ig];
            device & d = devices[g.id];
            if (pu.ticket == 0 || !d.upload_bg_done(d.backend, pu.ticket, false)) {
                pending_uploads[n_keep++] = pu;
                continue;
            }
            const int32_t n_expert = g.ref[0]->ne[2];
            layer & l = layers[pu.key / n_expert];
            l.uploading[pu.key % n_expert] = 0;
            g.slot_pending[pu.slot] = 0;
            set_resident(l, pu.key % n_expert, pu.slot);
        }
        pending_uploads.resize(n_keep);
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

        // the compute of this ubatch is done, so the slots of the victims can be overwritten
        // the uploads on the compute queue are done too, so their staging memory can be reused
        for (device & d : devices) {
            if (!d.buf) {
                continue;
            }
            ggml_backend_synchronize(d.backend);
            if (d.pending) {
                d.pending = false;
                if (d.upload_bg == nullptr) {
                    d.staging_used = 0;
                }
            }
        }
        const int64_t t_sync_us = ggml_time_us() - t_start_us;

        // the selections of this ubatch ran on the hot spare if they were resident, for small ubatches
        const bool small = n_tokens <= max_batch;
        spare_stats & st = small ? spare_small : spare_large;
        st.tokens += n_tokens;
        st.t_sync_us += t_sync_us;
        spare_cur = &st;

        // the copies of the uploads of the previous update are done in the background by now
        finish_submits();

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
                if (small && refill_left > 0) {
                    l.score_gen[e] = l.score_gen[e]*decay + c;
                }
            }
        }

        if (released) {
            // no device memory: the scores are kept, the cache is refreshed when it is restored
            if (half_life > 0.0f && !small) {
                rebalance = true;
                start_refill();
            }
            st.t_us += ggml_time_us() - t_start_us;
            spare_cur = nullptr;
            return;
        }

        activate_uploads();

        if (half_life > 0.0f) {
            if (!small) {
                // large ubatches (the prompt) only update the scores, the first small ubatch after them rebalances the cache without a budget
                // then the generated tokens are counted apart, for one more rebalance from them alone
                rebalance = true;
                start_refill();
            } else if (refill_left > 0 && (refill_left -= std::min<int64_t>(refill_left, n_tokens)) == 0) {
                // the first generated tokens predict the rest of the generation better than the prompt
                for (layer & l : layers) {
                    if (l.ig >= 0) {
                        l.score = l.score_gen;
                    }
                }
                refresh(SIZE_MAX, st);
                rebalance = false;
            } else {
                // a resident set from --moe-cache-hot predicts the generation better than the prompt, so it only changes within the budget
                refresh(rebalance && !warm ? SIZE_MAX : budget*n_tokens, st);
                rebalance = false;
            }
        }

        flush_maps();

        st.t_us += ggml_time_us() - t_start_us;
        spare_cur = nullptr;
    }

    // hot spare mode: replace the weakest resident experts by stronger ones, empty slots first
    // the budget counts the uploads into occupied slots, the empty ones are filled for free
    void refresh(size_t budget_bytes, spare_stats & st) {
        const int64_t t_start_us = ggml_time_us();
        refresh_impl(budget_bytes, st);
        if (budget_bytes == SIZE_MAX) {
            st.t_full_us += ggml_time_us() - t_start_us;
            st.n_full++;
        }
    }

    void refresh_impl(size_t budget_bytes, spare_stats & st) {
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
                    if (l.slot_of[e] < 0 && !l.uploading[e] && l.score[e] > 0.0f) {
                        best.emplace_back(l.score[e], il*n_expert + e);
                    }
                }
            }
            for (int32_t s = 0; s < g.n_slots; ++s) {
                if (g.slot_pending[s]) {
                    continue;
                }
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
            const int32_t e = sw.key % n_expert;
            g.key_of_slot[sw.slot] = sw.key;
            if (upload_expert(g, l, e, sw.slot, true)) {
                l.uploading[e] = 1;
                g.slot_pending[sw.slot] = 1;
                pending_uploads.push_back({ (int32_t) sw.ig, sw.key, sw.slot, 0 });
            } else {
                set_resident(l, e, sw.slot);
            }
            st.swaps++;
            st.bytes += expert_size;
        }

        for (device & d : devices) {
            if (d.upload_bg != nullptr) {
                submit_bg(d, true);
            }
        }
    }

    // the driver may make freed device memory available some time after the buffer is released (amdgpu releases
    // buffers that the GPU used in a deferred way): a buffer allocated before that may be placed in host memory
    static void wait_for_free_memory(ggml_backend_t backend, size_t target, const char * what) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        const int64_t t_start_us = ggml_time_us();
        for (;;) {
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            if (free >= target || total == 0) {
                return;
            }
            if (ggml_time_us() - t_start_us > 2000000) {
                LLAMA_LOG_WARN("%s: %s: %.2f MiB free on %s after 2 s, waited for %.2f MiB\n", __func__, what,
                    free/1024.0/1024.0, ggml_backend_name(backend), target/1024.0/1024.0);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

    // hot spare mode with resize: free the device memory of the cache, the scores and the resident set are kept
    void release() {
        if (!resize || released) {
            return;
        }
        // the uploads in flight write to the banks
        finish_submits();
        for (device & d : devices) {
            if (!d.buf) {
                continue;
            }
            if (d.upload_bg != nullptr) {
                d.upload_bg_done(d.backend, std::max(d.half_ticket[0], d.half_ticket[1]), true);
            }
            ggml_backend_synchronize(d.backend);
        }
        // an upload that did not finish keeps its expert in the resident set (key_of_slot), it is uploaded again
        for (const pending_upload & pu : pending_uploads) {
            group & g = groups[pu.ig];
            const int32_t n_expert = g.ref[0]->ne[2];
            layers[pu.key / n_expert].uploading[pu.key % n_expert] = 0;
            g.slot_pending[pu.slot] = 0;
        }
        pending_uploads.clear();
        for (layer & l : layers) {
            if (l.ig < 0) {
                continue;
            }
            for (size_t e = 0; e < l.slot_of.size(); ++e) {
                if (l.slot_of[e] >= 0) {
                    set_resident(l, (int32_t) e, -1);
                }
            }
        }
        for (device & d : devices) {
            if (!d.buf) {
                continue;
            }
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(ggml_backend_get_device(d.backend), &free, &total);
            d.buf.reset();
            for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                t->buffer = nullptr;
                t->data   = nullptr;
            }
            d.staging_used = 0;
            d.half         = 0;
            d.pending      = false;
            wait_for_free_memory(d.backend, free + d.buf_size/10*9, "releasing the MoE cache");
        }
        released = true;
    }

    // hot spare mode with resize: allocate the device memory again and upload the resident experts:
    // the hot file the first time, the resident set if it came from a hot file, otherwise the next refresh fills
    // the empty slots from the scores, like the rebalance after a prompt
    void restore() {
        if (!resize || !released) {
            return;
        }
        const int64_t t_start_us = ggml_time_us();
        spare_cur = &spare_restore;
        restore_impl();
        spare_restore.t_us += ggml_time_us() - t_start_us;
        spare_cur = nullptr;
    }

    void restore_impl() {
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            wait_for_free_memory(d.backend, d.buf_size + 64*1024*1024, "restoring the MoE cache");
            d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
            if (!d.buf) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(d.buf.get(), 0);
            ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        released = false;
        n_restores++;
        for (layer & l : layers) {
            if (l.ig >= 0) {
                l.dirty = true; // the device maps are new
            }
        }

        if (!hot_pending.empty()) {
            const std::string path = hot_pending;
            hot_pending.clear();
            load_hot(path);
        } else if (warm) {
            for (size_t ig = 0; ig < groups.size(); ++ig) {
                group & g = groups[ig];
                const int32_t n_expert = g.n_slots > 0 ? (int32_t) g.ref[0]->ne[2] : 0;
                for (int32_t s = 0; s < g.n_slots; ++s) {
                    const int32_t key = g.key_of_slot[s];
                    if (key < 0) {
                        continue;
                    }
                    layer & l = layers[key / n_expert];
                    const int32_t e = key % n_expert;
                    if (upload_expert(g, l, e, s, true)) {
                        l.uploading[e] = 1;
                        g.slot_pending[s] = 1;
                        pending_uploads.push_back({ (int32_t) ig, key, s, 0 });
                    } else {
                        set_resident(l, e, s);
                    }
                    for (const ggml_tensor * t : g.ref) {
                        restore_bytes += t->nb[2];
                    }
                }
            }
            for (device & d : devices) {
                if (d.upload_bg != nullptr) {
                    submit_bg(d, true);
                }
            }
        } else {
            for (group & g : groups) {
                std::fill(g.key_of_slot.begin(), g.key_of_slot.end(), -1);
                g.n_resident = 0;
            }
        }

        flush_maps();
        for (device & d : devices) {
            if (d.pending) {
                ggml_backend_synchronize(d.backend);
                d.pending = false;
                if (d.upload_bg == nullptr) {
                    d.staging_used = 0;
                }
            }
        }
    }

    // hot spare mode: count the next generated tokens apart, for one rebalance from their scores
    void start_refill() {
        refill_left = refill_tokens;
        for (layer & l : layers) {
            if (l.ig >= 0) {
                std::fill(l.score_gen.begin(), l.score_gen.end(), 0.0f);
            }
        }
    }

    bool get_hot_spare_maps(int32_t il, int64_t n_tokens, llama_moe_cache_hot_spare_maps & maps) const {
        if (mode != LLAMA_MOE_CACHE_MODE_HOT_SPARE || il < 0 || il >= (int32_t) layers.size() || layers[il].miss_map == nullptr) {
            return false;
        }
        // large batches use most experts of a layer, they keep running all host experts with the scheduler's offload
        // with resize, the cache has no device memory while large batches run
        if (n_tokens == 0 || n_tokens > hot_batch || released) {
            return false;
        }
        const layer & l = layers[il];
        maps = { l.miss_map, l.slot_dev, l.hit_dev };
        return true;
    }

    ~impl() {
        // the background uploads read the staging memory until they are done
        finish_submits();
        for (device & d : devices) {
            if (d.upload_bg != nullptr) {
                d.upload_bg_done(d.backend, std::max(d.half_ticket[0], d.half_ticket[1]), true);
            }
        }
        log_stats();
        if (!save_path.empty()) {
            save_hot_spare(save_path);
        }
    }

    // hot spare mode: write the resident experts, then the other experts with a score, best first, in the format of --moe-cache-hot
    void save_hot_spare(const std::string & path) const {
        std::vector<std::tuple<bool, float, int32_t, int32_t>> experts; // resident, score, il, expert
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            for (size_t e = 0; e < l.slot_of.size(); ++e) {
                if (l.slot_of[e] >= 0 || l.score[e] > 0.0f) {
                    experts.emplace_back(l.slot_of[e] >= 0, l.score[e], (int32_t) il, (int32_t) e);
                }
            }
        }
        std::sort(experts.begin(), experts.end(), [](const auto & a, const auto & b) {
            return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) : std::get<1>(a) > std::get<1>(b);
        });
        std::ofstream f(path);
        if (!f) {
            LLAMA_LOG_WARN("%s: failed to write the MoE cache hot file %s\n", __func__, path.c_str());
            return;
        }
        f << "# MoE cache hot spare: resident experts, then the other experts with a score, best first, \"layer expert\" per line\n";
        for (const auto & [resident, score, il, e] : experts) {
            f << il << " " << e << "\n";
        }
        LLAMA_LOG_INFO("%s: wrote %zu experts to %s\n", __func__, experts.size(), path.c_str());
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
            LLAMA_LOG_INFO("llama_moe_cache: hot spare, %s: %zu tokens,%s %zu swaps, uploaded = %.2f MiB (%.2f MiB/token), update = %.3f ms/token (sync %.3f, copy %.3f, waiting for the copy worker %.3f)\n",
                name, st.tokens, hits ? format(" hit rate = %.2f%%,", n > 0 ? 100.0*st.hits/n : 0.0).c_str() : "",
                st.swaps, st.bytes/1024.0/1024.0, st.bytes/1024.0/1024.0/st.tokens, st.t_us/1000.0/st.tokens,
                st.t_sync_us/1000.0/st.tokens, st.t_copy_us/1000.0/st.tokens, st.t_wait_us/1000.0/st.tokens);
            LLAMA_LOG_INFO("llama_moe_cache: hot spare, %s: per token, waiting for the copies %.3f ms in a refresh that filled a staging half "
                "and %.3f at the next update, for a half's uploads %.3f; %zu full rebalances, %.1f ms each\n",
                name, st.t_wait_half_us/1000.0/st.tokens, st.t_wait_next_us/1000.0/st.tokens, st.t_dma_us/1000.0/st.tokens,
                st.n_full, st.n_full > 0 ? st.t_full_us/1000.0/st.n_full : 0.0);
        };
        log_spare("ubatch <= 32", spare_small, true);
        log_spare("ubatch  > 32", spare_large, false);
        if (resize) {
            LLAMA_LOG_INFO("llama_moe_cache: hot spare, resize: released for large batches %zu times, %.2f MiB uploaded again, "
                "%.1f ms per restore (waiting for the copies %.1f, for a half's uploads %.1f)\n",
                n_restores, restore_bytes/1024.0/1024.0, n_restores > 0 ? spare_restore.t_us/1000.0/n_restores : 0.0,
                n_restores > 0 ? spare_restore.t_wait_us/1000.0/n_restores : 0.0, n_restores > 0 ? spare_restore.t_dma_us/1000.0/n_restores : 0.0);
        }
        if (worker) {
            const uint64_t bytes = worker->bytes;
            const int64_t  t_us  = worker->t_us;
            LLAMA_LOG_INFO("llama_moe_cache: hot spare, copy worker: %zu threads, %.2f GiB copied in %.1f s of thread time, %.2f GB/s per thread\n",
                worker->threads.size(), bytes/1024.0/1024.0/1024.0, t_us/1e6, t_us > 0 ? bytes/1e3/t_us : 0.0);
        }
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

int64_t llama_moe_cache::max_hot_spare_batch() {
    return impl::max_batch;
}

// the largest hot spare ubatch of a context with resize, for the contexts without the cache that share its device memory
static std::atomic<int64_t> g_resize_batch{0};

void llama_moe_cache::set_hot_spare_batch(int64_t n) {
    pimpl->hot_batch = std::max(n, impl::max_batch);
    if (pimpl->resize) {
        g_resize_batch = pimpl->hot_batch;
    }
}

int64_t llama_moe_cache::resize_batch() {
    return g_resize_batch;
}

void llama_moe_cache::limit_resize_batch(int64_t n) {
    int64_t cur = g_resize_batch;
    while (n < cur && !g_resize_batch.compare_exchange_weak(cur, n)) {
    }
}

bool llama_moe_cache::resizes() const {
    return pimpl->resize;
}

void llama_moe_cache::release() {
    pimpl->release();
}

void llama_moe_cache::restore() {
    pimpl->restore();
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
