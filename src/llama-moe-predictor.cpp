#include "llama-moe-predictor.h"

#include "llama-impl.h"

#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// the predictor file, version 1 (see the exporter, moe-pred/export_predictor.py):
//   moe_predictor.{version, n_layer, n_expert, n_expert_used, n_embd, every}: u32; .norm_eps, .map_eps: f32
//   moe_predictor.{hidden_layers, buckets, horizons}: i32 arrays; .hidden_slots, .maps: string arrays
//   tensors (f32, ggml order): prior [n_expert, n_layer]; norm.weight, norm.bias [D]; mlp.0.weight [D + n_buckets, F],
//   mlp.0.bias [F]; mlp.2.weight [F, R], mlp.2.bias [R]; per horizon H: head.H.w [n_buckets, n_maps, n_layer],
//   head.H.bias [n_expert, n_layer], head.H.v [R, n_expert*n_layer]; D = hidden layers * n_embd
// per head: logits[l, e] = bias[l, e] + sum_f w[b, f, l] * logp_f[l, e] + v[l*n_expert + e] . z with
// z = mlp.2(gelu_erf(mlp.0([layernorm(hidden), onehot(b)]))), b the bucket nearest to t, logp_f = log(p_f + map_eps)
// where p_f is map f made a distribution per layer (uniform if the layer has no counts); prediction = softmax over e

static const char * const MAP_NAMES[llama_moe_predictor::MAP_COUNT] = {
    "live64", "ctx512", "context", "turn", "prev_answer", "answer", "answer32", "answer8", "prior",
};

const char * llama_moe_predictor::map_name(map_kind k) {
    return k >= 0 && k < MAP_COUNT ? MAP_NAMES[k] : "?";
}

struct llama_moe_predictor::impl {
    int32_t n_layer  = 0;
    int32_t n_expert = 0;
    int32_t n_embd   = 0;
    int32_t every    = 16;
    float   norm_eps = 1e-5f;
    float   map_eps  = 1e-4f;
    std::vector<int32_t>  hidden_layers;
    std::vector<map_kind> maps;
    std::vector<int32_t>  buckets;
    std::vector<int32_t>  horizons;
    std::vector<float>    prior; // [n_layer*n_expert]

    ggml_backend_ptr        backend;
    ggml_context_ptr        ctx_w;   // weights
    ggml_backend_buffer_ptr buf_w;
    ggml_context_ptr        ctx_g;   // graph
    ggml_gallocr_t          galloc = nullptr;
    ggml_cgraph *           gf     = nullptr;
    ggml_tensor *           in_hid  = nullptr; // [D]
    ggml_tensor *           in_b    = nullptr; // [n_buckets] one-hot
    ggml_tensor *           in_logp = nullptr; // [n_maps, n_expert, n_layer]
    std::vector<ggml_tensor *> out;           // per horizon [n_expert, n_layer]
    std::vector<float>      logp;              // host copy of in_logp

    ~impl() {
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
    }
};

llama_moe_predictor::llama_moe_predictor() : pimpl(new impl()) {}
llama_moe_predictor::~llama_moe_predictor() = default;

const std::vector<int32_t> & llama_moe_predictor::hidden_layers() const { return pimpl->hidden_layers; }
const std::vector<llama_moe_predictor::map_kind> & llama_moe_predictor::maps() const { return pimpl->maps; }
const std::vector<int32_t> & llama_moe_predictor::horizons() const { return pimpl->horizons; }
int32_t llama_moe_predictor::every() const { return pimpl->every; }

std::unique_ptr<llama_moe_predictor> llama_moe_predictor::load(const std::string & path, const std::string & device,
                                                               int32_t n_layer, int32_t n_expert, int32_t n_embd) {
    const char * fn = path.c_str();
    ggml_context * ctx_meta = nullptr;
    gguf_context_ptr gctx(gguf_init_from_file(fn, { /*no_alloc =*/ true, /*ctx =*/ &ctx_meta }));
    if (!gctx) {
        LLAMA_LOG_ERROR("%s: cannot read the predictor file %s\n", __func__, fn);
        return nullptr;
    }
    ggml_context_ptr ctx_meta_ptr(ctx_meta);
    const gguf_context * g = gctx.get();

    auto fail = [&](const std::string & why) -> std::unique_ptr<llama_moe_predictor> {
        LLAMA_LOG_ERROR("%s: %s: %s\n", __func__, fn, why.c_str());
        return nullptr;
    };
    auto key = [&](const char * name) { return gguf_find_key(g, (std::string("moe_predictor.") + name).c_str()); };
    auto u32 = [&](const char * name, uint32_t & v) {
        const int64_t k = key(name);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_UINT32) {
            return false;
        }
        v = gguf_get_val_u32(g, k);
        return true;
    };
    auto i32s = [&](const char * name, std::vector<int32_t> & v) {
        const int64_t k = key(name);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, k) != GGUF_TYPE_INT32) {
            return false;
        }
        const int32_t * d = (const int32_t *) gguf_get_arr_data(g, k);
        v.assign(d, d + gguf_get_arr_n(g, k));
        return true;
    };
    auto strs = [&](const char * name, std::vector<std::string> & v) {
        const int64_t k = key(name);
        if (k < 0 || gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, k) != GGUF_TYPE_STRING) {
            return false;
        }
        v.clear();
        for (size_t i = 0; i < gguf_get_arr_n(g, k); ++i) {
            v.emplace_back(gguf_get_arr_str(g, k, i));
        }
        return true;
    };

    const int64_t k_arch = gguf_find_key(g, "general.architecture");
    if (k_arch < 0 || std::string(gguf_get_val_str(g, k_arch)) != "moe-predictor") {
        return fail("not a predictor file (general.architecture is not moe-predictor)");
    }
    uint32_t version = 0, f_layer = 0, f_expert = 0, f_embd = 0, every = 16;
    if (!u32("version", version) || version != 1) {
        return fail("unsupported predictor file version " + std::to_string(version) + ", this build reads version 1");
    }
    if (!u32("n_layer", f_layer) || !u32("n_expert", f_expert) || !u32("n_embd", f_embd)) {
        return fail("missing n_layer, n_expert or n_embd");
    }
    if ((int32_t) f_layer != n_layer || (int32_t) f_expert != n_expert || (int32_t) f_embd != n_embd) {
        return fail("made for " + std::to_string(f_layer) + " layers of " + std::to_string(f_expert) + " experts and " +
                    std::to_string(f_embd) + " embeddings, the model has " + std::to_string(n_layer) + ", " +
                    std::to_string(n_expert) + " and " + std::to_string(n_embd));
    }
    u32("every", every);

    std::unique_ptr<llama_moe_predictor> p(new llama_moe_predictor());
    impl & m = *p->pimpl;
    m.n_layer = n_layer; m.n_expert = n_expert; m.n_embd = n_embd; m.every = std::max<int32_t>(1, (int32_t) every);
    if (const int64_t k = key("norm_eps"); k >= 0) { m.norm_eps = gguf_get_val_f32(g, k); }
    if (const int64_t k = key("map_eps");  k >= 0) { m.map_eps  = gguf_get_val_f32(g, k); }

    std::vector<std::string> slots, maps;
    if (!i32s("hidden_layers", m.hidden_layers) || !i32s("buckets", m.buckets) || !i32s("horizons", m.horizons) ||
            !strs("hidden_slots", slots) || !strs("maps", maps)) {
        return fail("missing hidden_layers, buckets, horizons, hidden_slots or maps");
    }
    if (slots != std::vector<std::string>{ "last" }) {
        return fail("reads hidden slots this build doesn't provide (version 1 provides \"last\")");
    }
    for (int32_t il : m.hidden_layers) {
        if (il < 0 || il >= n_layer) {
            return fail("hidden layer " + std::to_string(il) + " out of range");
        }
    }
    for (const std::string & name : maps) {
        const char * const * it = std::find_if(std::begin(MAP_NAMES), std::end(MAP_NAMES), [&](const char * x) { return name == x; });
        if (it == std::end(MAP_NAMES)) {
            return fail("unknown count map " + name);
        }
        m.maps.push_back((map_kind) (it - std::begin(MAP_NAMES)));
    }
    if (m.buckets.empty() || m.horizons.empty() || !std::is_sorted(m.buckets.begin(), m.buckets.end())) {
        return fail("no buckets or horizons, or unsorted buckets");
    }

    // the tensors, with the shapes the version 1 contract gives them
    const int64_t D  = (int64_t) m.hidden_layers.size()*n_embd;
    const int64_t NB = m.buckets.size();
    const int64_t NF = m.maps.size();
    const int64_t NE = n_expert, NL = n_layer;
    ggml_tensor * t_mlp0 = ggml_get_tensor(ctx_meta, "mlp.0.weight");
    ggml_tensor * t_mlp2 = ggml_get_tensor(ctx_meta, "mlp.2.weight");
    if (!t_mlp0 || !t_mlp2) {
        return fail("missing mlp.0.weight or mlp.2.weight");
    }
    const int64_t F = t_mlp0->ne[1], R = t_mlp2->ne[1];
    struct want { std::string name; std::vector<int64_t> ne; };
    std::vector<want> wants = {
        { "prior", { NE, NL } }, { "norm.weight", { D } }, { "norm.bias", { D } },
        { "mlp.0.weight", { D + NB, F } }, { "mlp.0.bias", { F } }, { "mlp.2.weight", { F, R } }, { "mlp.2.bias", { R } },
    };
    for (int32_t H : m.horizons) {
        const std::string h = "head." + std::to_string(H) + ".";
        wants.push_back({ h + "w", { NB, NF, NL } });
        wants.push_back({ h + "bias", { NE, NL } });
        wants.push_back({ h + "v", { R, NE*NL } });
    }
    for (const want & w : wants) {
        const ggml_tensor * t = ggml_get_tensor(ctx_meta, w.name.c_str());
        if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) > (int) w.ne.size()) {
            return fail("missing or not f32: " + w.name);
        }
        for (size_t i = 0; i < w.ne.size(); ++i) {
            if (t->ne[i] != w.ne[i]) {
                return fail("tensor " + w.name + " has the wrong shape");
            }
        }
    }

    // the backend: the named device, the CPU by default
    ggml_backend_dev_t dev = device.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) : ggml_backend_dev_by_name(device.c_str());
    if (!dev) {
        return fail("no device " + (device.empty() ? std::string("CPU") : device));
    }
    m.backend.reset(ggml_backend_dev_init(dev, nullptr));
    if (!m.backend) {
        return fail("cannot start a backend on " + std::string(ggml_backend_dev_name(dev)));
    }
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        // the generation thread waits for it, and the model's threads are idle then
        const char * env = getenv("LLAMA_MOE_PREDICTOR_THREADS");
        const int n_threads = env ? std::max(1, atoi(env)) : 4;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")) {
            fn(m.backend.get(), n_threads);
        }
    }

    // the weights, read into a buffer of the backend
    {
        ggml_init_params ip = { ggml_tensor_overhead()*(wants.size() + 1), nullptr, true };
        m.ctx_w.reset(ggml_init(ip));
        for (const want & w : wants) {
            ggml_tensor * src = ggml_get_tensor(ctx_meta, w.name.c_str());
            ggml_tensor * dst = ggml_dup_tensor(m.ctx_w.get(), src);
            ggml_set_name(dst, w.name.c_str());
        }
        m.buf_w.reset(ggml_backend_alloc_ctx_tensors(m.ctx_w.get(), m.backend.get()));
        if (!m.buf_w) {
            return fail("cannot allocate the weights");
        }
        FILE * f = fopen(fn, "rb");
        if (!f) {
            return fail("cannot open");
        }
        std::vector<uint8_t> data;
        bool ok = true;
        for (const want & w : wants) {
            ggml_tensor * t = ggml_get_tensor(m.ctx_w.get(), w.name.c_str());
            const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, gguf_find_tensor(g, w.name.c_str()));
            data.resize(ggml_nbytes(t));
            ok = ok && fseek(f, (long) off, SEEK_SET) == 0 && fread(data.data(), 1, data.size(), f) == data.size();
            if (ok) {
                ggml_backend_tensor_set(t, data.data(), 0, data.size());
            }
            if (ok && w.name == "prior") {
                m.prior.assign((const float *) data.data(), (const float *) data.data() + NL*NE);
            }
        }
        fclose(f);
        if (!ok) {
            return fail("cannot read the tensor data");
        }
    }

    // the graph, built once: the bucket is an input (one-hot), so it doesn't change
    {
        const size_t n_nodes = 64 + 16*m.horizons.size();
        ggml_init_params ip = { ggml_tensor_overhead()*n_nodes + ggml_graph_overhead_custom(n_nodes, false), nullptr, true };
        m.ctx_g.reset(ggml_init(ip));
        ggml_context * ctx = m.ctx_g.get();
        auto W = [&](const std::string & name) { return ggml_get_tensor(m.ctx_w.get(), name.c_str()); };

        m.in_hid  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D);
        m.in_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NB);
        m.in_logp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, NF, NE, NL);
        for (ggml_tensor * t : { m.in_hid, m.in_b, m.in_logp }) {
            ggml_set_input(t);
        }

        ggml_tensor * h = ggml_norm(ctx, m.in_hid, m.norm_eps);
        h = ggml_add(ctx, ggml_mul(ctx, h, W("norm.weight")), W("norm.bias"));
        ggml_tensor * x = ggml_concat(ctx, h, m.in_b, 0);
        ggml_tensor * y = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, W("mlp.0.weight"), x), W("mlp.0.bias")));
        ggml_tensor * z = ggml_add(ctx, ggml_mul_mat(ctx, W("mlp.2.weight"), y), W("mlp.2.bias"));

        m.gf = ggml_new_graph_custom(ctx, n_nodes, false);
        for (int32_t H : m.horizons) {
            const std::string hn = "head." + std::to_string(H) + ".";
            // the weights of the bucket: [n_buckets, n_maps*n_layer] times the one-hot
            ggml_tensor * wb = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, W(hn + "w"), NB, NF*NL), m.in_b);
            wb = ggml_reshape_3d(ctx, wb, NF, 1, NL);
            ggml_tensor * counts = ggml_sum_rows(ctx, ggml_mul(ctx, m.in_logp, wb));          // [1, n_expert, n_layer]
            counts = ggml_reshape_2d(ctx, counts, NE, NL);
            ggml_tensor * hv = ggml_reshape_2d(ctx, ggml_mul_mat(ctx, W(hn + "v"), z), NE, NL);
            ggml_tensor * logits = ggml_add(ctx, ggml_add(ctx, counts, hv), W(hn + "bias"));
            ggml_tensor * prob = ggml_soft_max(ctx, logits);
            ggml_set_output(prob);
            ggml_build_forward_expand(m.gf, prob);
            m.out.push_back(prob);
        }
        m.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend.get()));
        if (!ggml_gallocr_alloc_graph(m.galloc, m.gf)) {
            return fail("cannot allocate the graph");
        }
        m.logp.resize(NF*NE*NL);
    }

    std::string names;
    for (map_kind k : m.maps) {
        names += std::string(names.empty() ? "" : " ") + map_name(k);
    }
    LLAMA_LOG_INFO("%s: %s: predicts %zu horizons every %d tokens from the inputs of %zu layers and %zu count maps (%s), on %s\n",
        __func__, fn, m.horizons.size(), m.every, m.hidden_layers.size(), m.maps.size(), names.c_str(), ggml_backend_dev_name(dev));
    return p;
}

bool llama_moe_predictor::predict(const std::vector<const float *> & maps, const std::vector<const float *> & hidden, int64_t t,
                                  std::vector<std::vector<float>> & out) {
    impl & m = *pimpl;
    const int64_t NF = m.maps.size(), NE = m.n_expert, NL = m.n_layer;
    GGML_ASSERT(maps.size() == m.maps.size() && hidden.size() == m.hidden_layers.size());

    // each map as a distribution per layer, its log in [n_maps, n_expert, n_layer] order
    for (int64_t f = 0; f < NF; ++f) {
        const float * src = m.maps[f] == MAP_PRIOR ? m.prior.data() : maps[f];
        for (int64_t l = 0; l < NL; ++l) {
            const float * row = src + l*NE;
            double sum = 0.0;
            for (int64_t e = 0; e < NE; ++e) {
                sum += row[e];
            }
            for (int64_t e = 0; e < NE; ++e) {
                const float p = sum > 0.0 ? (float) (row[e]/std::max(sum, 1e-9)) : 1.0f/NE;
                m.logp[(l*NE + e)*NF + f] = logf(p + m.map_eps);
            }
        }
    }
    ggml_backend_tensor_set(m.in_logp, m.logp.data(), 0, m.logp.size()*sizeof(float));
    for (size_t i = 0; i < hidden.size(); ++i) {
        ggml_backend_tensor_set(m.in_hid, hidden[i], i*m.n_embd*sizeof(float), m.n_embd*sizeof(float));
    }
    // the nearest bucket, the lower one on a tie
    size_t b = 0;
    for (size_t i = 1; i < m.buckets.size(); ++i) {
        if (std::llabs(m.buckets[i] - t) < std::llabs(m.buckets[b] - t)) {
            b = i;
        }
    }
    std::vector<float> onehot(m.buckets.size(), 0.0f);
    onehot[b] = 1.0f;
    ggml_backend_tensor_set(m.in_b, onehot.data(), 0, onehot.size()*sizeof(float));

    if (ggml_backend_graph_compute(m.backend.get(), m.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    out.resize(m.out.size());
    for (size_t i = 0; i < m.out.size(); ++i) {
        out[i].resize(NE*NL);
        ggml_backend_tensor_get(m.out[i], out[i].data(), 0, out[i].size()*sizeof(float));
    }
    return true;
}
