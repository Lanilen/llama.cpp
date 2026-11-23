// The fork's one-row RMS_NORM on Vulkan (rms_norm_ordered.comp) sums in the order of upstream's ADD + RMS_NORM
// partials, so a norm gives the same bits whether or not an ADD right before it hands it partial sums. This checks it,
// for each norm the partials path has (plain, * weight, * weight + residual, then * a scale), at several row lengths:
// the norm right after an ADD in one graph (the partials path) against the same norm of the ADD's result computed in a
// graph of its own (the norm alone), byte for byte. If upstream changes its partials, this fails; then
// GGML_VK_DISABLE_ORDERED_RMS_NORM=1 returns to upstream's plain norm (and this test skips; with TEST_NORM_ORDER_FORCE=1
// it runs and shows the cases that upstream's norm sums in another order).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int EXIT_SKIP = 77;

enum norm_kind { NORM_PLAIN, NORM_MUL, NORM_MUL_ADD, NORM_MUL_ADD_MUL };
static const char * kind_name[] = { "plain", "* weight", "* weight + residual", "* weight + residual, * scale" };

struct inputs { std::vector<float> a, b, w, r, e; };

// the norm chain of `kind` on x; w, r, e are inputs of ctx
static ggml_tensor * build_norm(ggml_context * ctx, ggml_tensor * x, norm_kind kind, ggml_tensor * w, ggml_tensor * r, ggml_tensor * e) {
    ggml_tensor * out = ggml_rms_norm(ctx, x, 1e-6f);
    if (kind >= NORM_MUL)         { out = ggml_mul(ctx, out, w); }
    if (kind >= NORM_MUL_ADD)     { out = ggml_add(ctx, out, r); }
    if (kind >= NORM_MUL_ADD_MUL) { out = ggml_mul(ctx, out, e); }
    return out;
}

static void set(ggml_tensor * t, const std::vector<float> & v) { ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(float)); }

// run_graph: a fresh context with the tensors of `build`, its graph computed twice (the partials buffer is sized after
// the first run), the output read back
template <typename F>
static std::vector<float> run_graph(ggml_backend_t be, F build) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    std::vector<std::pair<ggml_tensor *, const std::vector<float> *>> sets;
    ggml_tensor * out = build(ctx, sets);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (auto & s : sets) { set(s.first, *s.second); }
    ggml_backend_graph_compute(be, gf);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size()*sizeof(float));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main() {
    // TEST_NORM_ORDER_FORCE=1 runs the comparison with upstream's plain norm too: it shows that the check bites (there
    // the cases differ)
    if (getenv("GGML_VK_DISABLE_ORDERED_RMS_NORM") && !getenv("TEST_NORM_ORDER_FORCE")) {
        printf("skipped: GGML_VK_DISABLE_ORDERED_RMS_NORM is set (upstream's plain norm)\n");
        return EXIT_SKIP;
    }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("Vulkan0");
    if (dev == nullptr) {
        printf("skipped: no Vulkan device\n");
        return EXIT_SKIP;
    }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    printf("on %s\n", ggml_backend_dev_description(dev));

    std::mt19937 rng(20261123);
    int n_fail = 0, n_cases = 0;
    for (int64_t n : { 2304, 4096, 7168, 1000, 512, 300, 100, 17 }) {
        inputs in;
        std::normal_distribution<float> d(0.0f, 1.0f);
        for (auto * v : { &in.a, &in.b, &in.w, &in.r }) {
            v->resize(n);
            for (float & x : *v) { x = d(rng); }
        }
        // a few large activations, as residual streams have
        for (int i = 0; i < 4; ++i) { in.a[rng() % n] *= 300.0f; }
        in.e = { 0.73f };
        for (norm_kind kind : { NORM_PLAIN, NORM_MUL, NORM_MUL_ADD, NORM_MUL_ADD_MUL }) {
            auto inputs_of = [&](ggml_context * ctx, auto & sets, ggml_tensor *& w, ggml_tensor *& r, ggml_tensor *& e) {
                w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ w, &in.w });
                r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ r, &in.r });
                e = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1); sets.push_back({ e, &in.e });
            };
            // the ADD and the norm in one graph: the norm after the ADD's partial sums
            const std::vector<float> fused = run_graph(be, [&](ggml_context * ctx, auto & sets) {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ a, &in.a });
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ b, &in.b });
                ggml_tensor * w, * r, * e; inputs_of(ctx, sets, w, r, e);
                return build_norm(ctx, ggml_add(ctx, a, b), kind, w, r, e);
            });
            // the ADD alone, then the norm of its result in a graph of its own
            std::vector<float> sum;
            sum = run_graph(be, [&](ggml_context * ctx, auto & sets) {
                ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ a, &in.a });
                ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ b, &in.b });
                return ggml_add(ctx, a, b);
            });
            const std::vector<float> alone = run_graph(be, [&](ggml_context * ctx, auto & sets) {
                ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n); sets.push_back({ x, &sum });
                ggml_tensor * w, * r, * e; inputs_of(ctx, sets, w, r, e);
                return build_norm(ctx, x, kind, w, r, e);
            });
            int64_t n_diff = 0;
            for (int64_t i = 0; i < n; ++i) {
                n_diff += memcmp(&fused[i], &alone[i], sizeof(float)) != 0;
            }
            ++n_cases;
            if (n_diff) {
                ++n_fail;
            }
            printf("  %-4s n %5lld, %-28s: %s", n_diff ? "FAIL" : "ok", (long long) n, kind_name[kind],
                   n_diff ? "" : "the same to the bit after an ADD and alone\n");
            if (n_diff) {
                printf("%lld of %lld values differ after an ADD and alone\n", (long long) n_diff, (long long) n);
            }
        }
    }
    ggml_backend_free(be);
    printf("%s: %d of %d cases differ\n", n_fail ? "FAILED" : "passed", n_fail, n_cases);
    return n_fail ? 1 : 0;
}
