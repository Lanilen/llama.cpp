// checks the CPU's fused MUL_MAT_ID(gate) + MUL_MAT_ID(up) + SWIGLU (GGML_CPU_MMID_GLU=1, with GGML_CPU_MMID_FLAT)
// against the same ops unfused: the reference graph marks gate as an output, which keeps the CPU backend from fusing.
// Flash-Next's shapes (640 x 2560, 512 experts here 64), ids of 10 slots a token with some -1, 1/3/8 tokens.
// usage: GGML_CPU_MMID_GLU=1 GGML_CPU_MMID_FLAT=128 llama-mmid-glu-check [--threads N]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    int n_threads = 12;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        }
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, n_threads);
    const int64_t n_embd = 2560, n_ff = 640, n_expert = 64, n_slots = 10;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    bool all_ok = true;
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0 }) {
        std::vector<uint8_t> wq[2];
        for (auto & q : wq) {
            std::vector<float> wf(n_embd*n_ff*n_expert);
            for (float & v : wf) {
                v = uni(rng)*0.05f;
            }
            q.resize(ggml_row_size(type, n_embd)*n_ff*n_expert);
            ggml_quantize_chunk(type, wf.data(), q.data(), 0, n_ff*n_expert, n_embd, nullptr);
        }
        for (int64_t n_tokens : { 1, 3, 8 }) {
            std::vector<float> x(n_embd*n_tokens);
            for (float & v : x) {
                v = uni(rng);
            }
            // each token: a few distinct experts, the other slots -1 (the hits of the CPU branch)
            std::vector<int32_t> ids(n_slots*n_tokens, -1);
            for (int64_t t = 0; t < n_tokens; ++t) {
                const int n = 1 + (int) (t % 4);
                for (int j = 0; j < n; ++j) {
                    ids[t*n_slots + j*2] = (int32_t) ((t*7 + j*13 + 1) % n_expert);
                }
            }
            auto run = [&](bool reference, std::vector<float> & out) {
                ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * wg = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, n_expert);
                ggml_tensor * wu = ggml_new_tensor_3d(ctx, type, n_embd, n_ff, n_expert);
                ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
                ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_slots, n_tokens);
                ggml_tensor * g  = ggml_mul_mat_id(ctx, wg, b, id);
                ggml_tensor * u  = ggml_mul_mat_id(ctx, wu, b, id);
                ggml_tensor * y  = ggml_swiglu_split(ctx, g, u);
                if (reference) {
                    ggml_set_output(g);
                }
                ggml_set_output(y);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, y);
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
                ggml_backend_tensor_set(wg, wq[0].data(), 0, wq[0].size());
                ggml_backend_tensor_set(wu, wq[1].data(), 0, wq[1].size());
                ggml_backend_tensor_set(b, x.data(), 0, x.size()*sizeof(float));
                ggml_backend_tensor_set(id, ids.data(), 0, ids.size()*sizeof(int32_t));
                ggml_backend_graph_compute(cpu, gf);
                out.resize(ggml_nelements(y));
                ggml_backend_tensor_get(y, out.data(), 0, out.size()*sizeof(float));
                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            };
            std::vector<float> ref, got;
            run(true, ref);
            run(false, got);
            double err = 0, mag = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                err = std::max(err, (double) std::fabs(ref[i] - got[i]));
                mag = std::max(mag, (double) std::fabs(ref[i]));
            }
            const bool ok = err <= 1e-6*std::max(mag, 1.0);
            all_ok = all_ok && ok;
            printf("%-5s %lld tokens: max abs diff %.3e (max %.3e)%s\n", ggml_type_name(type), (long long) n_tokens, err, mag, ok ? "" : " FAIL");
        }
    }
    printf(all_ok ? "fused = unfused\n" : "MISMATCH\n");
    ggml_backend_free(cpu);
    return all_ok ? 0 : 1;
}
