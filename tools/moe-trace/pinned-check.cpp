// checks that a GPU computes MUL_MAT_ID with its weights in its own pinned host memory, read in place over the bus
// (the MoE cache's pinned tier): the same op on the CPU is the reference; prints the error and the GPU's rate.
// Flash-Next's miss shapes: 128 experts, 2 used a token, 640 x 2560 (gate/up) and 2560 x 640 (down), 1/3/8 tokens.
// usage: llama-pinned-check [--dev NAME] [--reps N]

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
#include <string>
#include <vector>

typedef ggml_backend_buffer_type_t (*pinned_buft_fn)(ggml_backend_dev_t dev);

int main(int argc, char ** argv) {
    std::string dev_name = "Vulkan0";
    int reps = 20;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dev") && i + 1 < argc) {
            dev_name = argv[++i];
        } else if (!strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = atoi(argv[++i]);
        }
    }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(dev_name.c_str());
    if (!dev) {
        fprintf(stderr, "no device %s\n", dev_name.c_str());
        return 1;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto pinned_fn = (pinned_buft_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_pinned_buffer_type");
    if (!pinned_fn) {
        fprintf(stderr, "%s has no pinned buffer type\n", dev_name.c_str());
        return 1;
    }
    ggml_backend_buffer_type_t buft_pinned = pinned_fn(dev);
    printf("%s, weights in %s: the device %s reading it in place\n", dev_name.c_str(), ggml_backend_buft_name(buft_pinned),
           ggml_backend_dev_supports_buft(dev, buft_pinned) ? "reports" : "does NOT report");
    ggml_backend_t gpu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    struct shape { ggml_type type; int64_t m, k; };
    const shape shapes[] = { { GGML_TYPE_Q4_K, 640, 2560 }, { GGML_TYPE_Q6_K, 640, 2560 }, { GGML_TYPE_Q5_1, 2560, 640 }, { GGML_TYPE_Q8_0, 2560, 640 } };
    const int64_t n_expert = 128, n_used = 2;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    bool all_ok = true;
    printf("%-5s %11s %3s | %12s | %9s %7s\n", "type", "m x k", "tok", "max rel err", "us a call", "GB/s");
    for (const shape & sh : shapes) {
        for (int64_t n : { 1, 3, 8 }) {
            // the weights: f32 quantized into the type, in pinned memory
            std::vector<float> wf(sh.m * sh.k * n_expert);
            for (float & v : wf) {
                v = uni(rng) * 0.1f;
            }
            const size_t row_size = ggml_row_size(sh.type, sh.k);
            std::vector<uint8_t> wq(row_size * sh.m * n_expert);
            ggml_quantize_chunk(sh.type, wf.data(), wq.data(), 0, sh.m * n_expert, sh.k, nullptr);
            std::vector<float> x(sh.k * n_used * n);
            for (float & v : x) {
                v = uni(rng);
            }
            std::vector<int32_t> ids(n_used * n);
            for (int64_t t = 0; t < n; ++t) {
                for (int64_t u = 0; u < n_used; ++u) {
                    ids[t * n_used + u] = (int32_t) ((t * 37 + u * 11 + 5) % n_expert);
                }
            }
            auto run = [&](ggml_backend_t backend, bool pinned, std::vector<float> & out, double * us) {
                ggml_init_params ip = { 12 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * as = ggml_new_tensor_3d(ctx, sh.type, sh.k, sh.m, n_expert);
                ggml_tensor * b  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, sh.k, n_used, n);
                ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n);
                // two reads of the weights in one graph, so the device's hazard checks see them beside other nodes
                ggml_tensor * y  = ggml_mul_mat_id(ctx, as, b, id);
                y = ggml_add(ctx, y, ggml_mul_mat_id(ctx, as, b, id));
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, y);
                ggml_backend_buffer_t wbuf = nullptr;
                if (pinned) {
                    wbuf = ggml_backend_buft_alloc_buffer(buft_pinned, ggml_nbytes(as) + 64);
                    ggml_tallocr ta = ggml_tallocr_new(wbuf);
                    ggml_tallocr_alloc(&ta, as);
                }
                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
                ggml_backend_tensor_set(as, wq.data(), 0, wq.size());
                ggml_backend_tensor_set(b, x.data(), 0, x.size() * sizeof(float));
                ggml_backend_tensor_set(id, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_graph_compute(backend, gf);
                if (us) {
                    const int64_t t0 = ggml_time_us();
                    for (int r = 0; r < reps; ++r) {
                        ggml_backend_graph_compute(backend, gf);
                    }
                    ggml_backend_synchronize(backend);
                    *us = (ggml_time_us() - t0) / (double) reps;
                }
                out.resize(ggml_nelements(y));
                ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
                ggml_backend_buffer_free(buf);
                if (wbuf) {
                    ggml_backend_buffer_free(wbuf);
                }
                ggml_free(ctx);
            };
            std::vector<float> ref, got;
            run(cpu, false, ref, nullptr);
            double us = 0;
            run(gpu, true, got, &us);
            double err = 0, mag = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                err = std::max(err, (double) std::fabs(ref[i] - got[i]));
                mag = std::max(mag, (double) std::fabs(ref[i]));
            }
            const double rel = err / std::max(mag, 1e-9);
            const bool ok = rel < 2e-2;
            all_ok = all_ok && ok;
            int64_t distinct = 0;
            { std::vector<int32_t> u = ids; std::sort(u.begin(), u.end()); distinct = std::unique(u.begin(), u.end()) - u.begin(); }
            const double gbs = 2 * distinct * sh.m * row_size / 1e3 / us;
            printf("%-5s %5lldx%-5lld %3lld | %12.2e%s | %9.1f %7.1f\n", ggml_type_name(sh.type), (long long) sh.m, (long long) sh.k,
                   (long long) n, rel, ok ? "" : " FAIL", us, gbs);
        }
    }
    printf(all_ok ? "all match the CPU\n" : "MISMATCH\n");
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return all_ok ? 0 : 1;
}
