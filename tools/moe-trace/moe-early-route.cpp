// how well a layer's experts can be predicted before its attention runs: Qwen3.8-Flash-Next's own router applied to the
// residual one block early (before the layer's attention, "pre") and two blocks early (before the previous layer's FFN,
// "pre2"), against the real selection. With LLAMA_MOE_EARLY_ROUTE=1 the graph computes those probabilities (qwen4exp
// build_early_route); this tool reads them, the real selections (ffn_moe_topk) and, with the hot spare, the real misses
// (ffn_moe_misses) for every generated token, and reports per predictor and N (its top N experts):
//   selected  the share of the real top-k in the top N
//   misses    the share of the real misses (the CPU's experts) in the top N: what a prefetch of the top N would catch
//   extra     experts in the top N that weren't selected, a token and layer (what a prefetch would touch for nothing)
// and the previous token's selection as the baseline predictor. Greedy generation; the prompt's tokens don't count.
// usage: LLAMA_MOE_EARLY_ROUTE=1 llama-moe-early-route -m model.gguf -f prompt.txt -n 512 [the server's placement flags]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

struct route_layer {
    int64_t n_expert = 0;
    std::vector<int32_t> topk;    // [tokens, k]
    std::vector<int32_t> misses;  // [tokens, k], -1 where the expert was resident (empty without the hot spare)
    std::vector<float>   probs;   // [tokens, n_expert], the real router's
    std::vector<float>   pre;     // [tokens, n_expert]
    std::vector<float>   pre2;    // [tokens, n_expert]
};

struct route_data {
    bool recording = false;   // the generated tokens only
    int64_t k = 0;
    std::map<int, route_layer> layers;
};

static bool layer_of(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') {
        return false;
    }
    char * end = nullptr;
    il = (int) strtol(name + n + 1, &end, 10);
    return end != name + n + 1 && *end == '\0';
}

static bool route_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (route_data *) user_data;
    int il = -1;
    const bool is_topk = layer_of(t->name, "ffn_moe_topk", il);
    const bool is_miss = !is_topk && layer_of(t->name, "ffn_moe_misses", il);
    const bool is_prob = !is_topk && !is_miss && layer_of(t->name, "ffn_moe_probs", il);
    const bool is_pre  = !is_topk && !is_miss && !is_prob && layer_of(t->name, "ffn_moe_probs_pre", il);
    const bool is_pre2 = !is_topk && !is_miss && !is_prob && !is_pre && layer_of(t->name, "ffn_moe_probs_pre2", il);
    const bool wanted  = is_topk || is_miss || is_prob || is_pre || is_pre2;
    if (ask) {
        return wanted && data->recording;
    }
    if (!wanted || !data->recording) {
        return true;
    }
    route_layer & l = data->layers[il];
    if (is_topk) {
        GGML_ASSERT(t->type == GGML_TYPE_I32 && t->ne[1] == 1);
        data->k = data->k ? data->k : t->ne[0];
        GGML_ASSERT(data->k == t->ne[0]);
        const size_t start = l.topk.size();
        l.topk.resize(start + t->ne[0]);
        ggml_backend_tensor_get(t, l.topk.data() + start, 0, t->ne[0]*sizeof(int32_t));
    } else if (is_miss) {
        GGML_ASSERT(t->type == GGML_TYPE_I32 && ggml_is_contiguous(t));
        const size_t start = l.misses.size();
        l.misses.resize(start + ggml_nelements(t));
        ggml_backend_tensor_get(t, l.misses.data() + start, 0, ggml_nbytes(t));
    } else {
        GGML_ASSERT(t->type == GGML_TYPE_F32 && t->ne[1] == 1 && ggml_is_contiguous(t));
        l.n_expert = t->ne[0];
        std::vector<float> & v = is_prob ? l.probs : is_pre ? l.pre : l.pre2;
        const size_t start = v.size();
        v.resize(start + t->ne[0]);
        ggml_backend_tensor_get(t, v.data() + start, 0, ggml_nbytes(t));
    }
    return true;
}

// the indices of the N largest of v, largest first
static std::vector<int32_t> top_n(const float * v, int64_t n, int N) {
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    N = (int) std::min<int64_t>(N, n);
    std::partial_sort(idx.begin(), idx.begin() + N, idx.end(), [&](int32_t a, int32_t b) { return v[a] > v[b]; });
    idx.resize(N);
    return idx;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION)) {
        return 1;
    }
    if (getenv("LLAMA_MOE_EARLY_ROUTE") == nullptr) {
        LOG_ERR("%s: set LLAMA_MOE_EARLY_ROUTE=1, so that the graph computes the early routing\n", __func__);
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    route_data data;
    params.cb_eval = route_cb_eval;
    params.cb_eval_user_data = &data;
    params.warmup = false;
    params.sampling.temp = 0.0f;   // greedy

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt, use -p or -f\n", __func__);
        return 1;
    }
    const int n_prompt = (int) tokens.size();
    if (n_prompt + std::max(params.n_predict, 0) > (int) llama_n_ctx(ctx)) {
        LOG_ERR("%s: %d prompt tokens + %d to predict do not fit in the context of %d\n", __func__, n_prompt, params.n_predict, llama_n_ctx(ctx));
        return 1;
    }
    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    for (llama_token id : tokens) {
        common_sampler_accept(smpl.get(), id, false);
    }
    for (int i = 0; i < n_prompt; i += params.n_batch) {
        const int n = std::min(params.n_batch, n_prompt - i);
        common_batch batch = common_batch_get_one(ctx, tokens.data() + i, n);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: failed to process the prompt\n", __func__);
            return 1;
        }
    }
    LOG_INF("%s: %d prompt tokens, generating %d\n", __func__, n_prompt, params.n_predict);

    data.recording = true;
    int n_gen = 0;
    std::string text;
    for (int i = 0; i < params.n_predict; ++i) {
        const llama_token id = common_sampler_sample(smpl.get(), ctx, -1);
        common_sampler_accept(smpl.get(), id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        text += common_token_to_piece(ctx, id);
        common_batch batch = common_batch_get_one(ctx, &id, 1);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: failed to decode\n", __func__);
            return 1;
        }
        n_gen++;
    }
    LOG_INF("%s: %d tokens generated: %.60s...\n", __func__, n_gen, text.c_str());

    // the tokens with a complete record (the last decode's selections aren't followed by a sampled token, but count)
    const int64_t k = data.k;
    const std::vector<int> Ns = { (int) k, 12, 16, 20, 32 };
    struct acc { double sel = 0, miss = 0, extra = 0; };   // sums over tokens and layers
    const char * names[] = { "real top-k (check)", "pre: one block early", "pre2: two blocks early", "previous token" };
    std::map<std::pair<int, int>, acc> sums;   // (predictor, N) -> sums
    std::map<int, std::map<std::pair<int, int>, acc>> by_layer;
    double n_sel = 0, n_miss = 0, n_pairs = 0;
    int n_layers_used = 0;
    for (auto & [il, l] : data.layers) {
        const int64_t T = (int64_t) l.topk.size()/k;
        if (T == 0 || l.pre.empty() || l.n_expert == 0) {
            continue;
        }
        n_layers_used++;
        const int64_t E = l.n_expert;
        const bool have_miss = l.misses.size() == l.topk.size();
        for (int64_t t = 0; t < T; ++t) {
            const int32_t * sel = l.topk.data() + t*k;
            std::vector<char> is_sel(E, 0), is_miss(E, 0);
            int64_t n_m = 0;
            for (int64_t j = 0; j < k; ++j) {
                is_sel[sel[j]] = 1;
                if (have_miss && l.misses[t*k + j] >= 0) {
                    is_miss[l.misses[t*k + j]] = 1;
                    n_m++;
                }
            }
            n_sel += k;
            n_miss += n_m;
            n_pairs += 1;
            for (int p = 0; p < 4; ++p) {
                const float * v = p == 0 ? l.probs.data() + t*E : p == 1 ? l.pre.data() + t*E : p == 2 ? l.pre2.data() + t*E : nullptr;
                if (p == 0 && l.probs.size() < (size_t) (t + 1)*E) {
                    continue;
                }
                if (p == 2 && l.pre2.size() < (size_t) (t + 1)*E) {
                    continue;
                }
                for (int N : Ns) {
                    std::vector<int32_t> pred;
                    if (p == 3) {
                        if (N != k || t == 0) {
                            continue;
                        }
                        pred.assign(l.topk.data() + (t - 1)*k, l.topk.data() + t*k);
                    } else {
                        pred = top_n(v, E, N);
                    }
                    acc & a = sums[{ p, N }];
                    acc & b = by_layer[il][{ p, N }];
                    for (int32_t e : pred) {
                        const double s = is_sel[e], m = is_miss[e];
                        a.sel += s; a.miss += m; a.extra += 1 - s;
                        b.sel += s; b.miss += m; b.extra += 1 - s;
                    }
                }
            }
        }
    }
    if (n_pairs == 0) {
        LOG_ERR("%s: nothing recorded: is the model Flash-Next (qwen4exp), and LLAMA_MOE_EARLY_ROUTE set?\n", __func__);
        return 1;
    }
    const bool have_miss = n_miss > 0;
    printf("\n%d generated tokens, %d MoE layers, %lld experts a token; %.2f misses a token and layer%s\n", n_gen, n_layers_used,
           (long long) k, n_miss/n_pairs, have_miss ? "" : " (no hot spare: no misses recorded)");
    printf("%-24s %4s | %9s | %9s | %7s\n", "predictor", "N", "selected", "misses", "extra");
    for (int p = 0; p < 4; ++p) {
        for (int N : Ns) {
            auto it = sums.find({ p, N });
            if (it == sums.end()) {
                continue;
            }
            const acc & a = it->second;
            printf("%-24s %4d | %8.1f%% | %8s | %7.2f\n", names[p], N, 100.0*a.sel/n_sel,
                   have_miss ? (std::to_string((int) (1000.0*a.miss/n_miss)/10.0).substr(0, 4) + "%").c_str() : "-", a.extra/n_pairs);
        }
    }
    // per layer: the one-block-early predictor at N = k and N = 20, the misses it catches
    printf("\nper layer, pre (one block early): selected at N=%lld, misses caught at N=%lld / 20, misses a token\n", (long long) k, (long long) k);
    for (auto & [il, l] : data.layers) {
        const int64_t T = (int64_t) l.topk.size()/k;
        if (T == 0 || l.pre.empty()) {
            continue;
        }
        const acc & a = by_layer[il][{ 1, (int) k }];
        const acc & b = by_layer[il][{ 1, 20 }];
        double lm = 0;
        for (int32_t e : l.misses) {
            lm += e >= 0;
        }
        printf("  layer %2d: %5.1f%%  %5s%%  %5s%%  %5.2f\n", il, 100.0*a.sel/(T*k),
               lm > 0 ? std::to_string((int) (1000.0*a.miss/lm)/10.0).substr(0, 4).c_str() : "-",
               lm > 0 ? std::to_string((int) (1000.0*b.miss/lm)/10.0).substr(0, 4).c_str() : "-", lm/T);
    }
    return 0;
}
