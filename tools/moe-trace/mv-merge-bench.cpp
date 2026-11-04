// measures what merging mat-vecs that read the same input would save on a GPU: for each group of weight shapes (the
// groups Flash-Next's verify pass has), a graph of L layers with distinct weights (so they come cold from VRAM, as in
// the model) computes each layer's mat-vecs as separate MUL_MATs, or as one MUL_MAT of the weights stacked by rows
// usage: llama-mv-merge-bench [layers (16)] [repetitions (20)]

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct group {
    const char * name;
    ggml_type type;
    int64_t k;
    std::vector<int64_t> rows;
    int per_pass; // layers of the model with this group
};

// median microseconds of one graph compute
static double time_graph(ggml_backend_t backend, ggml_cgraph * gf, int reps) {
    ggml_backend_graph_compute(backend, gf); // warm up
    std::vector<int64_t> t;
    for (int i = 0; i < reps; ++i) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        t.push_back(ggml_time_us() - t0);
    }
    std::sort(t.begin(), t.end());
    return (double) t[t.size()/2];
}

// us per layer of a group, its weights separate or stacked
static double run(ggml_backend_t backend, const group & g, int64_t n, bool merged, int L, int reps) {
    const size_t n_w = merged ? 1 : g.rows.size();
    ggml_init_params params = { ggml_tensor_overhead()*(L*(n_w + 1) + 16) + ggml_graph_overhead_custom(4096, false), nullptr, true };
    ggml_context * ctx_w = ggml_init(params);
    ggml_context * ctx   = ggml_init(params);

    const int64_t m_all = std::accumulate(g.rows.begin(), g.rows.end(), (int64_t) 0);
    std::vector<std::vector<ggml_tensor *>> w(L);
    std::vector<ggml_tensor *> x(L);
    for (int l = 0; l < L; ++l) {
        if (merged) {
            w[l].push_back(ggml_new_tensor_2d(ctx_w, g.type, g.k, m_all));
        } else {
            for (int64_t m : g.rows) {
                w[l].push_back(ggml_new_tensor_2d(ctx_w, g.type, g.k, m));
            }
        }
        x[l] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, g.k, n);
    }
    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
    ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // the same random rows in every weight (distinct memory is what matters), quantized once
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> f(g.k*m_all);
    for (float & v : f) { v = dist(rng); }
    std::vector<uint8_t> q(ggml_row_size(g.type, g.k)*m_all);
    if (g.type == GGML_TYPE_F32) {
        memcpy(q.data(), f.data(), q.size());
    } else {
        ggml_quantize_chunk(g.type, f.data(), q.data(), 0, m_all, g.k, nullptr);
    }
    std::vector<float> xs(g.k*n);
    for (float & v : xs) { v = dist(rng); }
    for (int l = 0; l < L; ++l) {
        for (ggml_tensor * t : w[l]) {
            ggml_backend_tensor_set(t, q.data(), 0, ggml_nbytes(t));
        }
        ggml_backend_tensor_set(x[l], xs.data(), 0, ggml_nbytes(x[l]));
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4096, false);
    for (int l = 0; l < L; ++l) {
        for (ggml_tensor * t : w[l]) {
            ggml_build_forward_expand(gf, ggml_mul_mat(ctx, t, x[l]));
        }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    const double us = time_graph(backend, gf, reps)/L;

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);
    return us;
}

int main(int argc, char ** argv) {
    const int L    = argc > 1 ? atoi(argv[1]) : 16;
    const int reps = argc > 2 ? atoi(argv[2]) : 20;

    // Flash-Next's mat-vecs that read the same input (fn-ops-night): per layer, the rows of each weight
    const std::vector<group> groups = {
        { "linear-attn q8_0 (qkv, gate, shexp gate, up)", GGML_TYPE_Q8_0,  2560, { 10240, 6144, 640, 640 },      36 },
        { "  of which qkv and gate",                      GGML_TYPE_Q8_0,  2560, { 10240, 6144 },                36 },
        { "  of which shexp gate and up",                 GGML_TYPE_Q8_0,  2560, { 640, 640 },                   36 },
        { "linear-attn f32 (router, shexp gate, a, b)",   GGML_TYPE_F32,   2560, { 512, 1, 48, 48 },             36 },
        { "hc q8_0 (attn down, ffn down)",                GGML_TYPE_Q8_0, 10240, { 320, 320 },                   48 },
        { "hc f32 (attn inject, ffn inject)",             GGML_TYPE_F32,  10240, { 4, 4 },                       48 },
        { "full-attn q8_0 (q, k, v, shexp gate, up)",     GGML_TYPE_Q8_0,  2560, { 12288, 512, 512, 640, 640 }, 12 },
        { "full-attn bf16 (indexer k, q)",                GGML_TYPE_BF16,  2560, { 128, 512 },                   12 },
    };

    ggml_time_init();
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s (%s), %d layers of distinct weights, median of %d runs; us per layer, and per pass over the model's layers\n",
               ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), L, reps);
        for (int64_t n : { 1, 3 }) {
            printf("%-46s %3s %9s %9s %9s %9s\n", "group", "n", "separate", "merged", "saved", "ms/pass");
            double total = 0.0;
            for (const group & g : groups) {
                const double sep = run(backend, g, n, false, L, reps);
                const double mer = run(backend, g, n, true,  L, reps);
                const bool part = g.name[0] == ' ';
                if (!part) {
                    total += (sep - mer)*g.per_pass/1000.0;
                }
                printf("%-46s %3d %9.1f %9.1f %9.1f %9.2f\n", g.name, (int) n, sep, mer, sep - mer, (sep - mer)*g.per_pass/1000.0);
            }
            printf("%-46s %3d %39.2f\n", "saved per pass (the groups, not their parts)", (int) n, total);
        }
        ggml_backend_free(backend);
    }
    return 0;
}
