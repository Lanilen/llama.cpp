// how far the logits of the same positions move with the length of the batch that computes them, without a draft model:
// a greedy answer to -p or -f in batches of one token sets the positions and is the base, then every round reads those
// positions again from a cleared memory and compares them with it. A flip is a position whose top token changed.
// the times are medians, so a graph's first run does not count
// LLAMA_DRIFT_ROUNDS  rounds, comma separated (default 1,2,3,4,8,9,1): N is batches of N tokens, in order, as a verify
//                     batch whose drafts all stand; rN is the same with a rollback, each batch keeping 1 to N tokens by a
//                     fixed sequence with the rest wrong tokens that are removed before the next batch
// LLAMA_DRIFT_RS      recurrent-state snapshots to allocate, what --spec-draft-n-max gives a server with a draft-mtp head
// LLAMA_DRIFT_SAVE    write the base to this file
// LLAMA_DRIFT_BASE    read the base from this file instead of generating it: compares another placement or cache state
// LLAMA_DRIFT_PROMPT  S: only time a prompt's batches: up to 1024 tokens in one batch, then the rest of -f in batches of S tokens with only the last one output, at most
//                     LLAMA_DRIFT_PROMPT_BATCHES (default 64) of them

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct drift_base {
    int32_t n_vocab = 0;
    std::vector<llama_token> tokens;   // tokens[i] is the top token of position i
    std::vector<float> logits;         // [n_positions, n_vocab]
    const float * at(size_t i) const { return logits.data() + i * n_vocab; }
};

static bool base_save(const drift_base & b, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    const int32_t n = (int32_t) b.tokens.size();
    bool ok = fwrite(&b.n_vocab, sizeof(int32_t), 1, f) == 1 && fwrite(&n, sizeof(int32_t), 1, f) == 1;
    ok = ok && fwrite(b.tokens.data(), sizeof(llama_token), n, f) == (size_t) n;
    ok = ok && fwrite(b.logits.data(), sizeof(float), b.logits.size(), f) == b.logits.size();
    return fclose(f) == 0 && ok;
}

static bool base_load(drift_base & b, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    int32_t n = 0;
    bool ok = fread(&b.n_vocab, sizeof(int32_t), 1, f) == 1 && fread(&n, sizeof(int32_t), 1, f) == 1;
    if (ok) {
        b.tokens.resize(n);
        b.logits.resize((size_t) n * b.n_vocab);
        ok = fread(b.tokens.data(), sizeof(llama_token), n, f) == (size_t) n;
        ok = ok && fread(b.logits.data(), sizeof(float), b.logits.size(), f) == b.logits.size();
    }
    fclose(f);
    return ok;
}

static void top2(const float * l, int n, int & a, int & b) {
    a = 0;
    b = -1;
    for (int i = 1; i < n; ++i) {
        if (l[i] > l[a]) {
            b = a;
            a = i;
        } else if (b < 0 || l[i] > l[b]) {
            b = i;
        }
    }
}

// KL(p || q) of the two softmaxes
static double kl(const float * p, const float * q, int n) {
    double mp = -INFINITY, mq = -INFINITY;
    for (int i = 0; i < n; ++i) {
        mp = std::max(mp, (double) p[i]);
        mq = std::max(mq, (double) q[i]);
    }
    double sp = 0, sq = 0;
    for (int i = 0; i < n; ++i) {
        sp += std::exp(p[i] - mp);
        sq += std::exp(q[i] - mq);
    }
    double out = 0;
    for (int i = 0; i < n; ++i) {
        const double lp = p[i] - mp - std::log(sp);
        const double lq = q[i] - mq - std::log(sq);
        const double pp = std::exp(lp);
        if (pp > 1e-12) {
            out += pp * (lp - lq);
        }
    }
    return out;
}

static double pct(std::vector<double> v, double q) {
    if (v.empty()) {
        return 0;
    }
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t) (q * v.size()))];
}

struct drift_run {
    llama_context * ctx;
    common_batch & batch;
    const std::vector<llama_token> & prompt;
    int n_vocab;

    // the prompt from a cleared memory: its last token gives the logits of position 0
    bool start() {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int n_batch = (int) llama_n_batch(ctx);
        for (int i = 0; i < (int) prompt.size(); i += n_batch) {
            batch.clear();
            for (int k = i; k < std::min(i + n_batch, (int) prompt.size()); ++k) {
                batch.add(prompt[k], k, 0, k + 1 == (int) prompt.size());
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                return false;
            }
        }
        return true;
    }
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MOE_TRACE)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    if (const char * env = getenv("LLAMA_DRIFT_RS")) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = atoi(env);
    }
    params.warmup = false;

    auto llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const std::vector<llama_token> prompt = common_tokenize(ctx, params.prompt, true, true);
    if (prompt.empty()) {
        LOG_ERR("%s: empty prompt, use -p or -f\n", __func__);
        return 1;
    }
    const int n_max_batch = (int) llama_n_batch(ctx);
    common_batch batch(ctx);
    drift_run run{ctx, batch, prompt, n_vocab};

    if (const char * env = getenv("LLAMA_DRIFT_PROMPT")) {
        const int S = atoi(env);
        const char * env_n = getenv("LLAMA_DRIFT_PROMPT_BATCHES");
        const int n_batches = env_n ? atoi(env_n) : 64;
        const int n_warm = std::min(1024, (int) prompt.size()/2);
        auto decode = [&](int i0, int i1) {
            batch.clear();
            for (int k = i0; k < i1; ++k) {
                batch.add(prompt[k], k, 0, k + 1 == i1);
            }
            return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0 && llama_get_logits_ith(ctx, -1) != nullptr;
        };
        if (S < 1 || S > n_max_batch || !decode(0, n_warm)) {
            LOG_ERR("%s: can't decode the prompt\n", __func__);
            return 1;
        }
        std::vector<double> t_batches;
        for (int i = n_warm; i + S <= (int) prompt.size() && (int) t_batches.size() < n_batches; i += S) {
            const int64_t t0 = ggml_time_us();
            if (!decode(i, i + S)) {
                LOG_ERR("%s: decode failed\n", __func__);
                return 1;
            }
            t_batches.push_back((ggml_time_us() - t0) / 1e3);
        }
        printf("prompt batches of %d tokens: %d batches, median %.1f ms a batch, %.2f ms a token\n", S, (int) t_batches.size(), pct(t_batches, .5), pct(t_batches, .5)/S);
        fflush(stdout);
        llama_init.reset();
        llama_backend_free();
        return 0;
    }

    drift_base base;
    const char * base_path = getenv("LLAMA_DRIFT_BASE");
    if (base_path) {
        if (!base_load(base, base_path) || base.n_vocab != n_vocab) {
            LOG_ERR("%s: can't read a base for this model from %s\n", __func__, base_path);
            return 1;
        }
    } else {
        base.n_vocab = n_vocab;
        if (!run.start()) {
            LOG_ERR("%s: decode failed\n", __func__);
            return 1;
        }
        for (int i = 0; i < params.n_predict; ++i) {
            const float * l = llama_get_logits_ith(ctx, -1);
            int a, b;
            top2(l, n_vocab, a, b);
            base.tokens.push_back(a);
            base.logits.insert(base.logits.end(), l, l + n_vocab);
            if (llama_vocab_is_eog(vocab, a)) {
                break;
            }
            batch.clear();
            batch.add(a, (llama_pos) prompt.size() + i, 0, true);
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: decode failed\n", __func__);
                return 1;
            }
        }
        if (const char * path = getenv("LLAMA_DRIFT_SAVE")) {
            if (!base_save(base, path)) {
                LOG_ERR("%s: can't write %s\n", __func__, path);
                return 1;
            }
        }
    }
    const int n = (int) base.tokens.size();

    std::vector<double> margin;
    for (int i = 0; i < n; ++i) {
        int a, b;
        top2(base.at(i), n_vocab, a, b);
        margin.push_back(base.at(i)[a] - base.at(i)[b]);
    }
    printf("base: %d prompt tokens, %d positions, %s; margin of the top token, median %.3f, 10%% under %.3f, 1%% under %.3f\n",
            (int) prompt.size(), n, base_path ? base_path : "generated in batches of 1", pct(margin, .5), pct(margin, .1), pct(margin, .01));
    printf("round  |  read | flips | agree   | mean KL   | |change of margin|: median     p90     max | ms/batch\n");

    std::vector<std::string> rounds;
    {
        const char * env = getenv("LLAMA_DRIFT_ROUNDS");
        std::string s = env ? env : "1,2,3,4,8,9,1";
        size_t p = 0;
        while (p <= s.size()) {
            const size_t e = std::min(s.find(',', p), s.size());
            if (e > p) {
                rounds.push_back(s.substr(p, e - p));
            }
            p = e + 1;
        }
    }

    for (const auto & r : rounds) {
        const bool reject = r[0] == 'r';
        const int L = atoi(r.c_str() + (reject ? 1 : 0));
        if (L < 1 || L > n_max_batch) {
            LOG_ERR("%s: bad round %s\n", __func__, r.c_str());
            return 1;
        }
        if (!run.start()) {
            LOG_ERR("%s: decode failed\n", __func__);
            return 1;
        }

        std::vector<double> dm;
        int flips = 0, read = 0;
        double sum_kl = 0;
        auto compare = [&](int pos, const float * l) {
            int a, b, c, d;
            top2(base.at(pos), n_vocab, a, b);
            top2(l, n_vocab, c, d);
            flips += c != a;
            dm.push_back(std::fabs((l[a] - l[b]) - (base.at(pos)[a] - base.at(pos)[b])));
            sum_kl += kl(base.at(pos), l, n_vocab);
            read++;
        };
        compare(0, llama_get_logits_ith(ctx, -1));

        uint32_t seq = 12345;
        std::vector<double> t_batches;
        // position i + 1 is the logits after tokens[i]
        for (int i = 0; i + 1 < n;) {
            const int len = std::min(L, n - 1 - i);
            seq = seq * 1664525u + 1013904223u;
            const int keep = reject ? 1 + (seq >> 16) % len : len;
            batch.clear();
            for (int k = 0; k < len; ++k) {
                const llama_token t = k < keep ? base.tokens[i + k] : (base.tokens[i + k] + 1000) % n_vocab;
                batch.add(t, (llama_pos) prompt.size() + i + k, 0, true);
            }
            const int64_t t0 = ggml_time_us();
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                LOG_ERR("%s: decode failed\n", __func__);
                return 1;
            }
            const float * first = llama_get_logits_ith(ctx, 0);   // waits for the graph
            t_batches.push_back((ggml_time_us() - t0) / 1e3);
            compare(i + 1, first);
            for (int k = 1; k < keep; ++k) {
                compare(i + k + 1, llama_get_logits_ith(ctx, k));
            }
            if (keep < len && !llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) prompt.size() + i + keep, -1)) {
                LOG_ERR("%s: can't roll back %d tokens\n", __func__, len - keep);
                return 1;
            }
            i += keep;
        }
        printf("%-6s | %5d | %5d | %6.2f%% | %9.6f | %27.4f %7.4f %7.4f | %7.1f\n", r.c_str(), read, flips, 100.0 * (read - flips) / read,
                sum_kl / read, pct(dm, .5), pct(dm, .9), pct(dm, 1.0), pct(t_batches, .5));
        fflush(stdout);
    }

    llama_init.reset(); // the backends print their timing as they go
    llama_backend_free();
    return 0;
}
