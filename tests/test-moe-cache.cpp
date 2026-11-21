// the MoE caches on tiny random models, one scenario a process:
//   test-moe-cache --arch qwen4exp --scenario hot-spare [--ref-dir DIR] [-v]
//   test-moe-cache --list
// The model is test-llama-archs's fixture of the architecture with more layers and experts, the experts quantized
// with a type per layer so that the cache has more than one group. Every scenario decodes the same tokens in the
// same batches as a run without a cache (the reference, from a process with a clean environment), compares the
// logits, and checks the cache's counters. Exit code 77: skipped (no GPU, no second device).
//   test-moe-cache --arch qwen4exp --make-ref --ref-dir DIR
// writes the references of this build once for all scenarios (ctest does); without --ref-dir each scenario makes its own.

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
int test_llama_archs_main(int argc, char ** argv);
#define main test_llama_archs_main
#include "test-llama-archs.cpp"
#undef main
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include "../src/llama-moe-cache.h"
#include "sampling.h"
#include "speculative.h"

#include <cstdarg>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <thread>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

static const int EXIT_SKIP = 77;

// the fixture: trunk layers (the first one dense where the architecture has leading dense layers), experts, experts used
static const uint32_t FIX_N_LAYER  = 6;
static const uint32_t FIX_N_EXPERT = 32;
static const uint32_t FIX_N_USED   = 4;
static const uint32_t FIX_N_FF_EXP = 512; // a multiple of 256, for the k-quants' and i-quants' rows
static const size_t   FIX_SEED     = 20261117;
static const float    FIX_STDEV    = 0.1f;

// the largest logit difference to the reference that passes, as a share of the standard deviation of the reference's
// logits, and the margin of the reference's top token, in the same unit, above which another top token fails.
// The reference is the same to the bit from run to run; what a cache changes is which kernel computes an expert
// (the device's for a resident one, the CPU's for the others). Measured on a 14600KF + RTX 4070 Ti and on lavapipe:
// up to 0.0007 with k-quant and i-quant experts (qwen4exp), up to 0.024 where the fixture's rows only take the legacy
// quants (glm5-next: n_embd 128), from the device's fp16 matmul.
// What this bound catches is a computation that went wrong (a graph of another batch, a branch left out). It does
// not catch a wrong expert in a slot: with a quarter of the uploaded experts replaced by their neighbours the logits
// of these random models move by 0.003 to 0.005 (qwen4exp) and 0.04 to 0.09 (glm5-next). The read-back checks do
// (LLAMA_MOE_CACHE_VERIFY, LLAMA_MOE_CACHE_JIT_CHECK, LLAMA_OFFLOAD_FAST_CHECK: byte for byte against the host's).
static float tol_logit(const llm_arch arch) {
    return arch == LLM_ARCH_GLM5_NEXT ? 0.1f : 0.002f;
}
static float tol_margin(const llm_arch arch) {
    return 4.0f*tol_logit(arch);
}

static bool g_verbose = false;
static std::string g_log; // the library's warnings and errors, printed when a scenario fails

static void log_callback(enum ggml_log_level level, const char * text, void *) {
    if (g_verbose) {
        fputs(text, stderr);
    } else if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        g_log += text;
    }
}

static long self_pid() {
#if defined(__linux__)
    return (long) getpid();
#else
    return 0;
#endif
}

struct check_list {
    int n_failed = 0;
    void operator()(bool ok, const char * fmt, ...) {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        printf("  %s  %s\n", ok ? "ok  " : "FAIL", buf);
        n_failed += ok ? 0 : 1;
    }
};

//
// fixture
//

// the type of an expert tensor of the i-th MoE layer of n: the first half q4_K, then one layer with a larger down
// projection (a group of its own, as GLM-5.3-Flash's layers 11 and 12), the rest iq4_xs; rows that aren't a
// multiple of 256 take the legacy type of the same class
static ggml_type fixture_expert_type(uint32_t i, uint32_t n, bool down, int64_t n_per_row) {
    const uint32_t mid = n/2;
    ggml_type type = i < mid ? GGML_TYPE_Q4_K : i == mid ? (down ? GGML_TYPE_Q6_K : GGML_TYPE_Q4_K) : GGML_TYPE_IQ4_XS;
    if (n_per_row % 256 != 0) {
        type = type == GGML_TYPE_Q4_K ? GGML_TYPE_Q4_0 : type == GGML_TYPE_Q6_K ? GGML_TYPE_Q8_0 : GGML_TYPE_Q5_0;
    }
    return type;
}

// in a batch that outputs only some of its tokens, the last layer computes its FFN for those alone
static bool fixture_last_layer_outputs_only(const llm_arch arch) {
    return arch == LLM_ARCH_QWEN4EXP;
}

// inkling2_loop: the core [FIX_LOOP_START, FIX_LOOP_END) runs FIX_LOOP_PASSES times, the later passes in cache slots after
// the physical layers that reuse the core's weights: the cache keeps one resident set for the passes of a layer
static const uint32_t FIX_LOOP_START  = 2;
static const uint32_t FIX_LOOP_END    = 4;
static const uint32_t FIX_LOOP_PASSES = 3;

// the times a token runs physical layer il
static uint32_t fixture_passes(const llm_arch arch, const int32_t il) {
    return arch == LLM_ARCH_INKLING2_LOOP && il >= (int32_t) FIX_LOOP_START && il < (int32_t) FIX_LOOP_END ? FIX_LOOP_PASSES : 1;
}

static uint32_t fixture_n_dense(const llm_arch arch) {
    return arch == LLM_ARCH_GLM5_NEXT ? 1 : 0; // get_gguf_ctx's leading_dense_block_count, which only glm5-next reads
}

static gguf_context_ptr fixture_gguf(const llm_arch arch, const bool mtp) {
    gguf_context_ptr ret = get_gguf_ctx(arch, true, FIX_N_LAYER + (mtp ? 1 : 0));
    kv_writer ms(arch, ret.get());
    ms.add_kv(LLM_KV_EXPERT_COUNT,               FIX_N_EXPERT);
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          FIX_N_USED);
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, FIX_N_FF_EXP);
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }
    if (arch == LLM_ARCH_INKLING2_LOOP) {
        ms.add_kv(LLM_KV_LOOP_START, FIX_LOOP_START);
        ms.add_kv(LLM_KV_LOOP_END,   FIX_LOOP_END);
        ms.add_kv(LLM_KV_LOOP_COUNT, FIX_LOOP_PASSES);
    }

    const int64_t kid = gguf_find_key(ret.get(), llama_arch_kv_name(arch, LLM_KV_EMBEDDING_LENGTH).c_str());
    GGML_ASSERT(kid >= 0);
    const int64_t n_embd = gguf_get_val_u32(ret.get(), kid);

    // the loader takes the type of a tensor the metadata lists, see llama_model_init_from_user
    const uint32_t n_dense = fixture_n_dense(arch);
    for (uint32_t il = n_dense; il < FIX_N_LAYER; il++) {
        for (const char * name : { "ffn_gate_exps", "ffn_up_exps", "ffn_gate_up_exps", "ffn_down_exps" }) {
            const bool down = strcmp(name, "ffn_down_exps") == 0;
            ggml_tensor t;
            memset(&t, 0, sizeof(ggml_tensor));
            t.type = fixture_expert_type(il - n_dense, FIX_N_LAYER - n_dense, down, down ? FIX_N_FF_EXP : n_embd);
            ggml_format_name(&t, "blk.%" PRIu32 ".%s.weight", il, name);
            gguf_add_tensor(ret.get(), &t);
        }
    }
    return ret;
}

static void fixture_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;
    // test-llama-archs's router bias is a ramp that makes every token select the same experts: no bias here
    if (strstr(tensor->name, "exp_probs_b") != nullptr && tensor->type == GGML_TYPE_F32) {
        std::vector<float> zero(ggml_nelements(tensor), 0.0f);
        ggml_backend_tensor_set(tensor, zero.data(), 0, ggml_nbytes(tensor));
        return;
    }
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16) {
        set_tensor_data(tensor, userdata);
        return;
    }
    std::vector<float> tmp(ggml_nelements(tensor));
    fill_normal(tmp, tensor->name, params);
    // rows in parallel: the i-quants take seconds a layer on one thread
    std::vector<uint8_t> q(ggml_nbytes(tensor));
    const int64_t n_per_row = tensor->ne[0];
    const int64_t n_rows    = ggml_nelements(tensor)/n_per_row;
    const int64_t n_threads = std::max<int64_t>(1, std::min<int64_t>(std::thread::hardware_concurrency(), 16));
    ggml_quantize_init(tensor->type);
    std::vector<std::thread> workers;
    for (int64_t it = 0; it < n_threads; it++) {
        const int64_t r0 = n_rows*it/n_threads;
        const int64_t r1 = n_rows*(it + 1)/n_threads;
        workers.emplace_back([&, r0, r1]() {
            ggml_quantize_chunk(tensor->type, tmp.data(), q.data(), r0*n_per_row, r1 - r0, n_per_row, nullptr); // writes at its rows
        });
    }
    for (std::thread & w : workers) {
        w.join();
    }
    ggml_backend_tensor_set(tensor, q.data(), 0, q.size());
}

struct fixture {
    llm_arch arch;
    std::vector<ggml_backend_dev_t> devs;
    std::string host_regex;
    llama_model_tensor_buft_override overrides[2];
    std::vector<float> split;
    llama_model_ptr model;

    // the experts of the trunk's layers stay in host memory (as -ncmoe), the layers are split evenly between the devices
    fixture(const llm_arch arch, const std::vector<ggml_backend_dev_t> & devs, const bool mtp) : arch(arch), devs(devs) {
        gguf_context_ptr gguf = fixture_gguf(arch, mtp);

        host_regex = "blk\\.(";
        for (uint32_t il = 0; il < FIX_N_LAYER; il++) {
            host_regex += (il > 0 ? "|" : "") + std::to_string(il);
        }
        host_regex += std::string(")") + LLM_FFN_EXPS_REGEX;
        overrides[0] = { host_regex.c_str(), ggml_backend_cpu_buffer_type() };
        overrides[1] = { nullptr, nullptr };

        llama_model_params mparams = llama_model_default_params();
        mparams.progress_callback = silent_model_load_progress;
        std::vector<ggml_backend_dev_t> devs_copy = devs;
        devs_copy.push_back(nullptr);
        mparams.devices               = devs_copy.data();
        mparams.split_mode            = LLAMA_SPLIT_MODE_LAYER;
        mparams.tensor_buft_overrides = overrides;
        mparams.load_mtp              = mtp;
        split.assign(llama_max_devices(), 0.0f);
        for (size_t i = 0; i < devs.size(); i++) {
            split[i] = 1.0f;
        }
        mparams.tensor_split = split.data();

        tensor_data_params tparams = { FIX_SEED, FIX_STDEV };
        model.reset(llama_model_init_from_user(gguf.get(), fixture_tensor_data, &tparams, mparams));
        if (!model) {
            throw std::runtime_error("failed to create the model");
        }
    }

    uint32_t n_moe_layers() const {
        return FIX_N_LAYER - fixture_n_dense(arch);
    }

    llama_context_params ctx_params() const {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = 1024;
        cparams.n_batch         = 256;
        cparams.n_ubatch        = 256;
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 4;
        return cparams;
    }
};

//
// the batches every run decodes, and their logits
//

struct step {
    int32_t n_batches;
    int32_t n_tokens;  // per batch
    bool    all;       // every token is output (generation, or the verification of drafts), else the last one (a prompt)
};

// a prompt, an answer, verify batches of 3 and 8, a prompt that fits a small reserve and one that doesn't, each
// followed by an answer: 432 tokens
static const std::vector<step> STEPS = {
    {  1,  24, false },
    { 48,   1, true  },
    {  8,   3, true  },
    {  4,   8, true  },
    {  1,  64, false },
    { 16,   1, true  },
    {  1, 200, false },
    { 24,   1, true  },
};

struct run_result {
    uint32_t n_vocab = 0;
    std::vector<float>   logits;    // [n_rows, n_vocab]
    std::vector<int32_t> row_batch; // tokens in the batch of each row
    size_t n_tokens = 0;
    size_t n_rows() const { return row_batch.size(); }
    // tokens and outputs of the ubatches of up to 32 tokens, and of the larger ones (llama_moe_cache_counters)
    size_t n_tokens_small = 0, n_tokens_large = 0;
    size_t n_out_small    = 0, n_out_large    = 0;
};

// n_steps: only the first steps, their rows are the reference's first rows
static run_result run_steps(llama_model * model, llama_context * lctx, const size_t n_steps = STEPS.size()) {
    const std::vector<step> steps(STEPS.begin(), STEPS.begin() + n_steps);
    run_result res;
    res.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    size_t n_total = 0;
    for (const step & s : steps) {
        n_total += (size_t) s.n_batches*s.n_tokens;
    }
    GGML_ASSERT(n_total <= llama_n_ctx(lctx));
    const std::vector<llama_token> tokens = get_tokens(n_total, res.n_vocab, FIX_SEED);

    common_batch batch(lctx);
    llama_pos pos = 0;
    for (const step & s : steps) {
        for (int32_t ib = 0; ib < s.n_batches; ib++) {
            batch.clear();
            for (int32_t i = 0; i < s.n_tokens; i++) {
                batch.add(tokens[pos + i], pos + i, 0, s.all || i == s.n_tokens - 1);
            }
            if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                throw std::runtime_error("failed to decode a batch of " + std::to_string(s.n_tokens) + " tokens at position " + std::to_string(pos));
            }
            for (int32_t i = s.all ? 0 : s.n_tokens - 1; i < s.n_tokens; i++) {
                const float * l = llama_get_logits_ith(lctx, i);
                if (l == nullptr) {
                    throw std::runtime_error("no logits");
                }
                res.logits.insert(res.logits.end(), l, l + res.n_vocab);
                res.row_batch.push_back(s.n_tokens);
            }
            pos += s.n_tokens;
            (s.n_tokens <= 32 ? res.n_tokens_small : res.n_tokens_large) += s.n_tokens;
            (s.n_tokens <= 32 ? res.n_out_small    : res.n_out_large)    += s.all ? s.n_tokens : 1;
        }
    }
    res.n_tokens = n_total;
    return res;
}

static bool ref_save(const run_result & r, const std::string & path) {
    const std::string tmp = path + ".tmp" + std::to_string(self_pid());
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        return false;
    }
    const uint32_t hdr[2] = { r.n_vocab, (uint32_t) r.n_rows() };
    bool ok = fwrite(hdr, sizeof(hdr), 1, f) == 1;
    ok = ok && fwrite(r.row_batch.data(), sizeof(int32_t), r.n_rows(), f) == r.n_rows();
    ok = ok && fwrite(r.logits.data(), sizeof(float), r.logits.size(), f) == r.logits.size();
    ok = fclose(f) == 0 && ok;
    return ok && std::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool ref_load(run_result & r, const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t hdr[2];
    bool ok = fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] > 0 && hdr[0] < (1u << 20) && hdr[1] < (1u << 20);
    if (ok) {
        r.n_vocab = hdr[0];
        r.row_batch.resize(hdr[1]);
        r.logits.resize((size_t) hdr[0]*hdr[1]);
        ok = fread(r.row_batch.data(), sizeof(int32_t), r.row_batch.size(), f) == r.row_batch.size();
        ok = ok && fread(r.logits.data(), sizeof(float), r.logits.size(), f) == r.logits.size();
    }
    fclose(f);
    return ok;
}

// rows read one token at a time and rows of larger batches, each against the same row of the reference
static void check_logits(check_list & check, const llm_arch arch, const run_result & ref, const run_result & got) {
    if (ref.n_vocab != got.n_vocab || got.n_rows() > ref.n_rows() || !std::equal(got.row_batch.begin(), got.row_batch.end(), ref.row_batch.begin())) {
        check(false, "the run read the reference's rows (%zu of %zu)", got.n_rows(), ref.n_rows());
        return;
    }
    double sum = 0.0, sum2 = 0.0;
    for (const float v : ref.logits) {
        sum  += v;
        sum2 += (double) v*v;
    }
    const double stdev = sqrt(std::max(1e-12, sum2/ref.logits.size() - (sum/ref.logits.size())*(sum/ref.logits.size())));
    const double tol   = tol_logit(arch)*stdev;
    const double tol_m = tol_margin(arch)*stdev;
    struct acc_t { double max_diff = 0.0; size_t n = 0, flips = 0, flips_bad = 0; bool finite = true; } one, more;
    for (size_t r = 0; r < got.n_rows(); r++) {
        const float * a = ref.logits.data() + r*ref.n_vocab;
        const float * b = got.logits.data() + r*ref.n_vocab;
        acc_t & acc = ref.row_batch[r] == 1 ? one : more;
        uint32_t top_a = 0, top_b = 0;
        float second_a = -INFINITY;
        for (uint32_t i = 0; i < ref.n_vocab; i++) {
            if (!std::isfinite(b[i])) {
                acc.finite = false;
            }
            acc.max_diff = std::max(acc.max_diff, (double) fabsf(a[i] - b[i]));
            if (a[i] > a[top_a]) {
                top_a = i;
            }
            if (b[i] > b[top_b]) {
                top_b = i;
            }
        }
        for (uint32_t i = 0; i < ref.n_vocab; i++) {
            if (i != top_a) {
                second_a = std::max(second_a, a[i]);
            }
        }
        acc.n++;
        if (top_a != top_b) {
            acc.flips++;
            acc.flips_bad += a[top_a] - second_a > tol_m ? 1 : 0;
        }
    }
    for (const auto & it : { std::make_pair("one token at a time", &one), std::make_pair("larger batches     ", &more) }) {
        const acc_t & acc = *it.second;
        if (acc.n == 0) {
            continue;
        }
        check(acc.finite && acc.max_diff <= tol && acc.flips_bad == 0,
            "logits, %s: %3zu rows, largest difference %.5f of their deviation (bound %.3f), %zu other top tokens, %zu of them above a margin of %.3f",
            it.first, acc.n, acc.max_diff/stdev, (double) tol_logit(arch), acc.flips, acc.flips_bad, (double) tol_margin(arch));
    }
}

//
// devices and child processes
//

static std::vector<ggml_backend_dev_t> gpu_devices() {
    std::vector<ggml_backend_dev_t> ret;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ret.push_back(dev);
        }
    }
    return ret;
}

// runs this program again and waits for it; -1 if it can't
static int run_self(const std::vector<std::string> & args, const std::vector<std::pair<std::string, std::string>> & env) {
#if defined(__linux__)
    char self[4096];
    const ssize_t n_self = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n_self <= 0) {
        return -1;
    }
    self[n_self] = '\0';
    fflush(stdout);
    fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        for (const auto & kv : env) {
            setenv(kv.first.c_str(), kv.second.c_str(), 1);
        }
        std::vector<char *> argv;
        argv.push_back(self);
        for (const std::string & a : args) {
            argv.push_back(const_cast<char *>(a.c_str()));
        }
        argv.push_back(nullptr);
        execv(self, argv.data());
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
#else
    (void) args;
    (void) env;
    return -1;
#endif
}

// drops the switches of the fork, of the backends and of OpenMP, but not which devices there are
static void clean_env() {
#if defined(__linux__)
    std::vector<std::string> names;
    for (char ** e = environ; *e != nullptr; e++) {
        const std::string kv = *e;
        const std::string name = kv.substr(0, kv.find('='));
        if ((name.rfind("LLAMA_", 0) == 0 || name.rfind("GGML_", 0) == 0 || name.rfind("GOMP_", 0) == 0) && name != "GGML_VK_VISIBLE_DEVICES") {
            names.push_back(name);
        }
    }
    for (const std::string & name : names) {
        unsetenv(name.c_str());
    }
#endif
}

//
// scenarios
//

struct scenario_ctx {
    const fixture & fx;
    const run_result & ref;
    check_list & check;
    std::string tmp_dir;
};

struct scenario {
    const char * name;
    const char * what;
    int n_devices;
    bool mtp;
    std::vector<std::pair<const char *, const char *>> env;
    std::function<void(scenario_ctx &)> run;
};

static size_t mib(double v) {
    return (size_t) (v*1024.0*1024.0);
}

static llama_context_ptr make_ctx(const fixture & fx, const llama_context_params & cparams) {
    llama_context_ptr lctx(llama_init_from_model(fx.model.get(), cparams));
    if (!lctx) {
        throw std::runtime_error("failed to create the context");
    }
    return lctx;
}

static llama_moe_cache_counters get_counters(llama_context * lctx) {
    llama_moe_cache_counters c;
    if (!llama_moe_cache_get_counters(lctx, c)) {
        throw std::runtime_error("the context has no MoE cache");
    }
    return c;
}

// the experts in the cache's slots; serving: without the ones still being uploaded
static std::set<int32_t> residents(llama_context * lctx, const bool serving = false) {
    std::set<int32_t> r;
    for (const auto & g : get_counters(lctx).groups) {
        r.insert(g.residents.begin(), g.residents.end());
        if (serving) {
            for (const int32_t key : g.uploading) {
                r.erase(key);
            }
        }
    }
    return r;
}

static int32_t cached_layers(const llama_moe_cache_counters & c) {
    int32_t n = 0;
    for (const auto & g : c.groups) {
        n += g.n_slots > 0 ? (int32_t) g.layers.size() : 0;
    }
    return n;
}

// the selections the cached layers make over n_tokens tokens of which n_out are output
static size_t expected_selections(const fixture & fx, const llama_moe_cache_counters & c, const size_t n_tokens, const size_t n_out) {
    size_t n = 0;
    for (const auto & g : c.groups) {
        for (const int32_t il : g.layers) {
            if (g.n_slots > 0) {
                n += FIX_N_USED*fixture_passes(fx.arch, il)*(il == (int32_t) FIX_N_LAYER - 1 && fixture_last_layer_outputs_only(fx.arch) ? n_out : n_tokens);
            }
        }
    }
    return n;
}

static std::string groups_str(const llama_moe_cache_counters & c) {
    std::string s;
    for (const auto & g : c.groups) {
        s += string_format("%s[dev %d, %d layers %s, %d slots, %d resident]", s.empty() ? "" : " ", g.device, (int) g.layers.size(), ggml_type_name(g.type), g.n_slots, g.n_resident);
    }
    return s;
}

// what holds for the hot spare after any run
static void check_hot_spare_counters(check_list & check, const fixture & fx, const llama_moe_cache_counters & c, const run_result & run, const bool every_layer = true) {
    int32_t n_layers = 0;
    bool slots_ok = true;
    for (const auto & g : c.groups) {
        n_layers += (int32_t) g.layers.size();
        std::set<int32_t> keys(g.residents.begin(), g.residents.end());
        slots_ok = slots_ok && g.n_resident <= g.n_slots && (int32_t) g.residents.size() == g.n_resident && keys.size() == g.residents.size() &&
            (keys.empty() || (*keys.begin() >= 0 && *keys.rbegin() < (int32_t) (FIX_N_LAYER*FIX_N_EXPERT)));
    }
    check(c.mode == LLAMA_MOE_CACHE_MODE_HOT_SPARE && c.n_expert_used == (int32_t) FIX_N_USED, "hot spare mode, %d experts a token", c.n_expert_used);
    check(n_layers == (int32_t) fx.n_moe_layers() && c.groups.size() >= 3, "groups: %s", groups_str(c).c_str());
    if (every_layer) {
        check(cached_layers(c) == n_layers, "every MoE layer is cached (%d of %d)", cached_layers(c), n_layers);
    }
    check(slots_ok, "each group: residents <= slots, every resident expert once");
    check(c.spare_small.tokens == run.n_tokens_small && c.spare_large.tokens == run.n_tokens_large, "every token is counted: %zu in small ubatches, %zu in large ones",
        c.spare_small.tokens, c.spare_large.tokens);
    const size_t n_small = expected_selections(fx, c, run.n_tokens_small, run.n_out_small);
    const size_t n_large = expected_selections(fx, c, run.n_tokens_large, run.n_out_large);
    check(c.spare_small.hits + c.spare_small.misses == n_small, "small ubatches: hits %zu + misses %zu = the selections of the cached layers (%zu)",
        c.spare_small.hits, c.spare_small.misses, n_small);
    check(c.spare_large.hits + c.spare_large.misses == n_large, "large ubatches: hits %zu + misses %zu = the selections of the cached layers (%zu)",
        c.spare_large.hits, c.spare_large.misses, n_large);
}

static llama_context_params hot_spare_params(const fixture & fx, const double cache_mib) {
    llama_context_params cparams = fx.ctx_params();
    cparams.moe_cache_mode       = LLAMA_MOE_CACHE_MODE_HOT_SPARE;
    cparams.moe_cache_size       = mib(cache_mib);
    cparams.moe_cache_half_life  = 64.0f;
    cparams.moe_cache_budget_mib = 1.0f;
    return cparams;
}

static void scn_lru(scenario_ctx & s, const float cpu_share) {
    llama_context_params cparams = s.fx.ctx_params();
    cparams.moe_cache_mode      = LLAMA_MOE_CACHE_MODE_LRU;
    cparams.moe_cache_size      = mib(12);
    cparams.moe_cache_cpu_share = cpu_share;
    llama_context_ptr lctx = make_ctx(s.fx, cparams);
    const run_result got = run_steps(s.fx.model.get(), lctx.get());
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    s.check(c.mode == LLAMA_MOE_CACHE_MODE_LRU && cached_layers(c) > 0, "lru mode, groups: %s", groups_str(c).c_str());
    s.check(c.lru_small.hits > 0 && c.lru_small.misses > 0, "ubatches of up to 8 tokens read the cache: %zu hits, %zu misses", c.lru_small.hits, c.lru_small.misses);
    if (cpu_share > 0.0f) {
        s.check(c.lru_small.cpu > 0 && c.lru_small.cpu < c.lru_small.misses, "some of the misses on the CPU: %zu of %zu", c.lru_small.cpu, c.lru_small.misses);
    } else {
        s.check(c.lru_small.cpu == 0, "no miss on the CPU");
    }
}

static void scn_hot_spare(scenario_ctx & s) {
    llama_context_ptr lctx = make_ctx(s.fx, hot_spare_params(s.fx, 12));
    const run_result got = run_steps(s.fx.model.get(), lctx.get());
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    check_hot_spare_counters(s.check, s.fx, c, got);
    s.check(c.spare_small.hits > 0 && c.spare_small.swaps > 0, "the hot spare serves and swaps: %zu hits, %zu swaps", c.spare_small.hits, c.spare_small.swaps);
    s.check(c.n_restores == 0 && !c.released, "no release without resize");
}

// runs the steps with --moe-cache-save; returns the residents of a start from that file: its experts in order
// (the residents at exit first, then the others by score) until the slots of each group are full
static std::set<int32_t> make_hot_file(scenario_ctx & s, const std::string & path) {
    std::set<int32_t> serving;
    llama_moe_cache_counters c;
    {
        llama_context_params cparams = hot_spare_params(s.fx, 12);
        cparams.moe_cache_save = path.c_str();
        llama_context_ptr lctx = make_ctx(s.fx, cparams);
        run_steps(s.fx.model.get(), lctx.get());
        serving = residents(lctx.get(), true);
        c = get_counters(lctx.get());
    }
    std::set<int32_t> first, start;
    std::vector<int32_t> n_taken(c.groups.size(), 0);
    size_t n_lines = 0;
    std::ifstream f(path);
    for (std::string line; std::getline(f, line); ) {
        int il, e;
        if (line.empty() || line[0] == '#' || sscanf(line.c_str(), "%d %d", &il, &e) != 2) {
            continue;
        }
        if (n_lines++ < serving.size()) {
            first.insert(il*FIX_N_EXPERT + e);
        }
        for (size_t ig = 0; ig < c.groups.size(); ig++) {
            const auto & layers = c.groups[ig].layers;
            if (std::find(layers.begin(), layers.end(), il) != layers.end() && n_taken[ig] < c.groups[ig].n_slots) {
                n_taken[ig]++;
                start.insert(il*FIX_N_EXPERT + e);
            }
        }
    }
    s.check(!serving.empty() && first == serving, "the file written at exit lists the %zu serving residents first (%zu experts in all)", serving.size(), n_lines);
    return start;
}

static std::string hot_file_path(const scenario_ctx & s) {
    return s.tmp_dir + "/test-moe-cache-" + llama_arch_name(s.fx.arch) + "-" + std::to_string(self_pid()) + ".hot";
}

// the residents written at exit are the residents of the next start, and with no half-life they stay
static void scn_hot_file(scenario_ctx & s) {
    const std::string path = hot_file_path(s);
    const std::set<int32_t> start = make_hot_file(s, path);

    llama_context_params cparams = hot_spare_params(s.fx, 12);
    cparams.moe_cache_hot       = path.c_str();
    cparams.moe_cache_half_life = 0.0f;
    cparams.moe_cache_refill    = 0;
    llama_context_ptr lctx = make_ctx(s.fx, cparams);
    s.check(residents(lctx.get()) == start, "a start from the file has its first experts in the slots (%zu)", start.size());
    const run_result got = run_steps(s.fx.model.get(), lctx.get());
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    check_hot_spare_counters(s.check, s.fx, c, got);
    s.check(residents(lctx.get()) == start && c.spare_small.swaps == 0 && c.spare_large.swaps == 0, "without a half-life the residents never change (%zu swaps)",
        c.spare_small.swaps + c.spare_large.swaps);
    std::remove(path.c_str());
}

// --moe-cache-resize with --moe-cache-batch 64: the batch of 200 tokens runs with the cache released
static void scn_resize(scenario_ctx & s) {
    llama_context_params cparams = hot_spare_params(s.fx, 12);
    cparams.moe_cache_resize = true;
    cparams.moe_cache_batch  = 64;
    llama_context_ptr lctx = make_ctx(s.fx, cparams);
    const run_result got = run_steps(s.fx.model.get(), lctx.get());
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    check_hot_spare_counters(s.check, s.fx, c, got);
    // the first reserve is the large one: the cache comes for the first small batch, and again after the batch of 200
    s.check(c.n_restores == 2 && !c.released && !residents(lctx.get()).empty(), "released for the large batch and restored: %zu restores, %zu residents at the end",
        c.n_restores, residents(lctx.get()).size());
    s.check(c.spare_small.hits > 0, "the hot spare serves: %zu hits in small ubatches", c.spare_small.hits);
}

// the same from a hot file: a restore uploads the residents again
static void scn_resize_warm(scenario_ctx & s) {
    const std::string path = hot_file_path(s);
    const std::set<int32_t> start = make_hot_file(s, path);

    llama_context_params cparams = hot_spare_params(s.fx, 12);
    cparams.moe_cache_hot       = path.c_str();
    cparams.moe_cache_half_life = 0.0f;
    cparams.moe_cache_refill    = 0;
    cparams.moe_cache_resize    = true;
    cparams.moe_cache_batch     = 64;
    llama_context_ptr lctx = make_ctx(s.fx, cparams);
    const run_result got = run_steps(s.fx.model.get(), lctx.get());
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    check_hot_spare_counters(s.check, s.fx, c, got);
    s.check(c.n_restores == 2 && c.restore_bytes > 0 && !c.released, "released for the large batch and restored: %zu restores, %.2f MiB uploaded again",
        c.n_restores, c.restore_bytes/1048576.0);
    s.check(residents(lctx.get()) == start, "the residents after the restore are the file's first experts (%zu)", start.size());
    std::remove(path.c_str());
}

// the device takes no buffer of a bank's size (GGML_VK_FORCE_MAX_BUFFER_SIZE): the layers of a type go to several groups
static void scn_small_buffers(scenario_ctx & s) {
    if (strncmp(ggml_backend_dev_name(s.fx.devs[0]), "Vulkan", 6) != 0) {
        throw std::runtime_error("skip: the switch is the Vulkan backend's");
    }
    if (s.fx.arch == LLM_ARCH_GLM5_NEXT) {
        // 2 MiB is too small for its other buffers and its banks fit 2.5 MiB
        throw std::runtime_error("skip: this fixture's banks are no larger than its other buffers");
    }
    // no large batch: with buffers this small the backend's own staging buffer for its copies does not fit
    llama_context_ptr lctx = make_ctx(s.fx, hot_spare_params(s.fx, 40));
    const run_result got = run_steps(s.fx.model.get(), lctx.get(), 4);
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    check_hot_spare_counters(s.check, s.fx, c, got);
    std::map<std::pair<int32_t, int32_t>, int32_t> n_groups; // by type and expert count
    bool split = false;
    for (const auto & g : c.groups) {
        split = split || ++n_groups[{ (int32_t) g.type, (int32_t) g.layers.size() > 0 ? g.n_expert : 0 }] > 1;
    }
    s.check(split && c.groups.size() > 3, "layers of one type are in several groups (%zu groups)", c.groups.size());
    s.check(c.spare_small.hits > 0, "the hot spare serves: %zu hits in small ubatches", c.spare_small.hits);
}

// a cache on each device for the layers it has
static void scn_two_devices(scenario_ctx & s) {
    llama_context_params cparams = hot_spare_params(s.fx, 16);
    const size_t sizes[2] = { mib(8), mib(8) };
    cparams.moe_cache_sizes   = sizes;
    cparams.n_moe_cache_sizes = 2;
    cparams.moe_cache_resize  = true;
    cparams.moe_cache_batch   = 32;
    // up to the batch of 64 tokens and the answer after it: the second device may be a software one
    llama_context_ptr lctx = make_ctx(s.fx, cparams);
    const run_result got = run_steps(s.fx.model.get(), lctx.get(), 6);
    check_logits(s.check, s.fx.arch, s.ref, got);
    const llama_moe_cache_counters c = get_counters(lctx.get());
    check_hot_spare_counters(s.check, s.fx, c, got);
    int32_t n_cached[2] = { 0, 0 };
    int32_t n_res[2]    = { 0, 0 };
    for (const auto & g : c.groups) {
        if (g.device >= 0 && g.device < 2) {
            n_cached[g.device] += g.n_slots > 0 ? (int32_t) g.layers.size() : 0;
            n_res[g.device]    += g.n_resident;
        }
    }
    s.check(n_cached[0] > 0 && n_cached[1] > 0 && n_res[0] > 0 && n_res[1] > 0, "each device caches its layers: %d and %d layers, %d and %d residents",
        n_cached[0], n_cached[1], n_res[0], n_res[1]);
    s.check(c.n_restores == 2 && !c.released && c.spare_small.hits > 0, "released for the large batch and restored on both: %zu restores, %zu hits in small ubatches",
        c.n_restores, c.spare_small.hits);
}

// the MTP block drafts for the target with the hot spare and resize, as a server with --spec-type draft-mtp: the
// loop of examples/speculative-simple, but each token is the one of an answer made without cache or drafts, and
// the row that predicts it is compared with that answer's. The drafts of a random head are nearly all wrong:
// each one is a rollback of the target's memory (recurrent state included)
static void scn_mtp(scenario_ctx & s) {
    const int32_t n_prompt = 24, n_gen = 48, n_max = 3;
    llama_model * model = s.fx.model.get();
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const std::vector<llama_token> prompt = get_tokens(n_prompt, n_vocab, FIX_SEED);
    const auto top = [&](const float * l) {
        return (llama_token) (std::max_element(l, l + n_vocab) - l);
    };

    run_result base; // the rows that predict each token of the answer
    std::vector<llama_token> answer;
    {
        llama_context_ptr lctx = make_ctx(s.fx, s.fx.ctx_params());
        common_batch batch(lctx.get());
        for (int32_t i = 0; i < n_prompt; i++) {
            batch.add(prompt[i], i, 0, i == n_prompt - 1);
        }
        for (int32_t k = 0; k < n_gen; k++) {
            if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                throw std::runtime_error("failed to decode the answer without drafts");
            }
            const float * l = llama_get_logits_ith(lctx.get(), -1);
            base.logits.insert(base.logits.end(), l, l + n_vocab);
            base.row_batch.push_back(n_max + 1); // compared with rows of verify batches
            answer.push_back(top(l));
            batch.clear();
            batch.add(answer.back(), n_prompt + k, 0, true);
        }
        base.n_vocab = n_vocab;
    }

    llama_context_params cparams = hot_spare_params(s.fx, 12);
    cparams.moe_cache_resize = true;
    cparams.moe_cache_batch  = 64;
    cparams.n_rs_seq         = n_max;
    llama_context_ptr ctx_tgt = make_ctx(s.fx, cparams);

    llama_context_params cparams_dft = s.fx.ctx_params();
    cparams_dft.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
    cparams_dft.n_rs_seq  = 0;
    cparams_dft.ctx_other = ctx_tgt.get();
    llama_context_ptr ctx_dft = make_ctx(s.fx, cparams_dft);

    common_params_speculative sparams;
    sparams.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sparams.draft.ctx_tgt = ctx_tgt.get();
    sparams.draft.ctx_dft = ctx_dft.get();
    sparams.draft.n_max   = n_max;
    sparams.draft.n_min   = 0;
    sparams.draft.p_min   = 0.0f;
    common_speculative_ptr spec(common_speculative_init(sparams, 1));
    if (!spec) {
        throw std::runtime_error("failed to initialize the draft-mtp head");
    }

    common_batch batch(ctx_tgt.get());
    for (int32_t i = 0; i < n_prompt - 1; i++) {
        batch.add(prompt[i], i, 0, false);
    }
    if (llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 || !common_speculative_process(spec.get(), batch)) {
        throw std::runtime_error("failed to process the prompt");
    }
    llama_token id_last = prompt.back();
    llama_tokens prompt_tgt(prompt.begin(), prompt.end() - 1);
    llama_pos n_past = n_prompt - 1;
    common_speculative_begin(spec.get(), 0, prompt_tgt);

    run_result got;
    got.n_vocab = n_vocab;
    size_t n_passes = 0, n_drafted = 0, n_accepted = 0, n_rollbacks = 0;
    while ((int32_t) got.n_rows() < n_gen) {
        const int32_t k = (int32_t) got.n_rows();
        llama_tokens draft;
        common_speculative_draft_params & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true;
        dp.n_max    = std::min(n_max, n_gen - k - 1);
        dp.pos0     = n_past;
        dp.id_last  = id_last;
        dp.prompt   = &prompt_tgt;
        dp.result   = &draft;
        common_speculative_draft(spec.get());
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, n_past, -1);
        if ((int32_t) draft.size() > n_gen - k - 1) {
            draft.resize(n_gen - k - 1);
        }

        batch.clear();
        batch.add(id_last, n_past++, 0, true);
        for (size_t i = 0; i < draft.size(); i++) {
            batch.add(draft[i], n_past + i, 0, true);
        }
        if (llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 || !common_speculative_process(spec.get(), batch)) {
            throw std::runtime_error("failed to verify a draft");
        }

        llama_tokens ids;
        for (size_t i = 0; i <= draft.size(); i++) {
            const float * l = llama_get_logits_ith(ctx_tgt.get(), i);
            got.logits.insert(got.logits.end(), l, l + n_vocab);
            got.row_batch.push_back(n_max + 1);
            ids.push_back(answer[k + i]);
            if (i == draft.size() || draft[i] != ids.back()) {
                break;
            }
        }
        common_speculative_accept(spec.get(), 0, ids.size() - 1);
        n_past += ids.size() - 1;
        for (const llama_token id : ids) {
            prompt_tgt.push_back(id_last);
            id_last = id;
        }
        llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), 0, n_past, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), 0, n_past, -1);

        n_passes++;
        n_drafted   += draft.size();
        n_accepted  += ids.size() - 1;
        n_rollbacks += ids.size() - 1 < draft.size() ? 1 : 0;
    }

    check_logits(s.check, s.fx.arch, base, got);
    s.check(n_drafted > 0 && n_rollbacks > 0, "%zu passes, %zu tokens drafted, %zu accepted, %zu passes rolled back", n_passes, n_drafted, n_accepted, n_rollbacks);
    const llama_moe_cache_counters c = get_counters(ctx_tgt.get());
    s.check(cached_layers(c) == (int32_t) s.fx.n_moe_layers() && c.spare_small.hits > 0 && !c.released,
        "the target's hot spare serves next to the draft context: %s, %zu hits", groups_str(c).c_str(), c.spare_small.hits);
}

static const std::vector<scenario> SCENARIOS = {
    { "lru",           "upstream's cache: the selected experts of small ubatches from a device buffer", 1, false, {},
        [](scenario_ctx & s) { scn_lru(s, 0.0f); } },
    { "lru-cpu-share", "the same with half of the misses computed on the CPU", 1, false, {},
        [](scenario_ctx & s) { scn_lru(s, 0.5f); } },
    { "hot-spare",     "resident experts on the device, the rest on the CPU, swaps by decayed scores", 1, false, {}, scn_hot_spare },
    { "hot-file",      "--moe-cache-save and --moe-cache-hot, residents that never change", 1, false, {}, scn_hot_file },
    { "resize",        "--moe-cache-resize: release for a large batch, restore after", 1, false, {}, scn_resize },
    { "resize-warm",   "the same from a hot file: the restore uploads its residents again", 1, false, {}, scn_resize_warm },
    { "small-buffers", "banks larger than the device's buffers: the layers split into more groups", 1, false,
        { { "GGML_VK_FORCE_MAX_BUFFER_SIZE", "8388608" }, { "GGML_VK_SUBALLOCATION_BLOCK_SIZE", "8388608" } }, scn_small_buffers },
    { "two-devices",   "layers split between two devices, a cache on each, with resize", 2, false, {}, scn_two_devices },
    { "mtp",           "drafts of the MTP block verified and rolled back, the target with the hot spare and resize", 1, true, {}, scn_mtp },
};

static void usage(const char * argv0) {
    printf("usage: %s --arch ARCH --scenario NAME [--ref-dir DIR] [-v]\n       %s --arch ARCH --make-ref --ref-dir DIR\n       %s --list\n", argv0, argv0, argv0);
}

int main(int argc, char ** argv) {
    std::string arch_name = "qwen4exp";
    std::string scenario_name;
    std::string ref_dir;
    std::string write_ref;
    int write_ref_devices = 1;
    bool make_ref = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--list") {
            for (const scenario & s : SCENARIOS) {
                printf("%-18s %s\n", s.name, s.what);
            }
            return 0;
        } else if (a == "-v") {
            g_verbose = true;
        } else if (a == "--arch" && i + 1 < argc) {
            arch_name = argv[++i];
        } else if (a == "--scenario" && i + 1 < argc) {
            scenario_name = argv[++i];
        } else if (a == "--ref-dir" && i + 1 < argc) {
            ref_dir = argv[++i];
        } else if (a == "--make-ref") {
            make_ref = true;
        } else if (a == "--write-ref" && i + 2 < argc) {
            write_ref = argv[++i];
            write_ref_devices = atoi(argv[++i]);
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    const scenario * scn = nullptr;
    for (const scenario & s : SCENARIOS) {
        if (scenario_name == s.name) {
            scn = &s;
        }
    }
    const auto ref_path = [&](const std::string & dir, int n_dev) {
        return dir + "/test-moe-cache-ref-" + arch_name + "-" + std::to_string(n_dev) + "dev.bin";
    };
    if (make_ref && !ref_dir.empty()) {
        // the second device may not exist: the scenarios that need it skip too
        for (int n_dev = 1; n_dev <= 2; n_dev++) {
            std::remove(ref_path(ref_dir, n_dev).c_str());
            const int rc = run_self({ "--arch", arch_name, "--write-ref", ref_path(ref_dir, n_dev), std::to_string(n_dev) }, {});
            printf("reference of %s on %d device(s): %s\n", arch_name.c_str(), n_dev, rc == 0 ? "written" : rc == EXIT_SKIP ? "skipped" : "FAILED");
            if (rc != 0 && (rc != EXIT_SKIP || n_dev == 1)) {
                return rc;
            }
        }
        return 0;
    }
    if (write_ref.empty() && scn == nullptr) {
        usage(argv[0]);
        return 1;
    }

    // before anything reads them: the switches are read once a process
    if (!write_ref.empty()) {
        clean_env();
    } else {
        for (const auto & kv : scn->env) {
            if (getenv(kv.first) == nullptr) { // a value from outside stands, to try another one
                setenv(kv.first, kv.second, 1);
            }
        }
    }
    const int n_devices = write_ref.empty() ? scn->n_devices : write_ref_devices;

    llama_log_set(log_callback, nullptr);
    llama_backend_init();
    init_normal_pool(FIX_SEED, FIX_STDEV);

    const llm_arch arch = llama_arch_from_string(arch_name);
    if (arch == LLM_ARCH_UNKNOWN) {
        fprintf(stderr, "unknown architecture %s\n", arch_name.c_str());
        return 1;
    }

    std::vector<ggml_backend_dev_t> devs = gpu_devices();
    if ((int) devs.size() < n_devices) {
        // a software Vulkan device can stand in: the Vulkan backend leaves it out unless it is asked for by index
        if (getenv("GGML_VK_VISIBLE_DEVICES") == nullptr) {
            std::string list;
            for (int i = 0; i < n_devices; i++) {
                list += (i > 0 ? "," : "") + std::to_string(i);
            }
            const int rc = run_self(std::vector<std::string>(argv + 1, argv + argc), { { "GGML_VK_VISIBLE_DEVICES", list } });
            if (rc >= 0 && rc != 127) {
                return rc;
            }
        }
        printf("SKIP: %d GPU device(s) needed, %zu found\n", n_devices, devs.size());
        return EXIT_SKIP;
    }
    devs.resize(n_devices);

    try {
        if (!write_ref.empty()) {
            fixture fx(arch, devs, false);
            llama_context_ptr lctx = make_ctx(fx, fx.ctx_params());
            // the scenarios on two devices stop after the batch of 64 tokens
            const run_result r = run_steps(fx.model.get(), lctx.get(), n_devices == 2 ? 6 : STEPS.size());
            return ref_save(r, write_ref) ? 0 : 1;
        }

        // the files a scenario writes: next to the references when ctest runs it
        const std::string tmp_dir = ref_dir.empty() ? std::filesystem::temp_directory_path().string() : ref_dir;
        std::string dev_names;
        for (ggml_backend_dev_t dev : devs) {
            dev_names += std::string(dev_names.empty() ? "" : " + ") + ggml_backend_dev_description(dev);
        }
        printf("%s, %s: %s\n  on %s\n", arch_name.c_str(), scn->name, scn->what, dev_names.c_str());

        // the reference: the same model and placement without a cache, from a process without any of the switches
        run_result ref;
        if (!ref_dir.empty()) {
            if (!ref_load(ref, ref_path(ref_dir, n_devices))) {
                fprintf(stderr, "no reference %s: run --make-ref first\n", ref_path(ref_dir, n_devices).c_str());
                return 1;
            }
        } else {
            const std::string path = tmp_dir + "/test-moe-cache-ref-" + std::to_string(self_pid()) + ".bin";
            const int rc = run_self({ "--arch", arch_name, "--write-ref", path, std::to_string(n_devices) }, {});
            const bool ok = rc == 0 && ref_load(ref, path);
            std::remove(path.c_str());
            if (!ok) {
                fprintf(stderr, "failed to make the reference (exit code %d)\n", rc);
                return 1;
            }
        }

        check_list check;
        {
            fixture fx(arch, devs, scn->mtp);
            scenario_ctx sc = { fx, ref, check, tmp_dir };
            scn->run(sc);
        }
        if (check.n_failed > 0) {
            printf("FAILED: %d check(s)\n%s", check.n_failed, g_log.c_str());
            return 1;
        }
        printf("PASSED\n");
        return 0;
    } catch (const std::exception & err) {
        if (strncmp(err.what(), "skip: ", 6) == 0) {
            printf("SKIP: %s\n", err.what() + 6);
            return EXIT_SKIP;
        }
        printf("FAILED: %s\n%s", err.what(), g_log.c_str());
        return 1;
    }
}
