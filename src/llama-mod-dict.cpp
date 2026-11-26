// MoD, the fork's Mixture of Dictionaries: a DSpark drafter drafts over a dictionary of tokens (see llama.h).
// The dictionary's rows of the output head (the target's, borrowed) and of the Markov head's vocabulary matrix are
// copied as they are into tensors of their own on the head's device, K = the fixed tokens + reserve slots; the graph
// (dflash.cpp build_logits, dspark.cpp's Markov head) multiplies by those and scatters the logits back to the
// vocabulary through the slots' token ids, -inf elsewhere, so sampling and the host see the whole vocabulary as before.
// DSpark's Markov chain stays in slot space (its TOP_K / ARGMAX over the K slots, the picks mapped to token ids through
// mod_ids_f) and scatters once, for the sampler. Free slots are -inf (mod_mask) and scatter to a sink row past the
// vocabulary that every scatter drops.
// Slots: [0, n_fixed) the fixed tokens; then a ring of single tokens (the prompt's and the answer's tokens outside the
// dictionary, the oldest overwritten when it is full); then, with families (llama_dspark_dict_families), blocks of a
// family's tokens outside the fixed ones, loaded when the answer meets a token of the family and evicted least recently
// used first (a token of a loaded family that is used again keeps its block). A token is in one slot at most.
// Free slots hold a copy of slot 0's rows, -inf in the mask and the sink as their id. One dictionary a drafter model.

#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-vocab.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_mod_dict {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    const ggml_tensor * src_head = nullptr; // the rows are copied from these
    const ggml_tensor * src_w2   = nullptr;
    const ggml_tensor * src_w2_b = nullptr;

    int32_t n_fixed = 0;
    int32_t n_slots = 0;
    // the ring of single tokens: slots [ring_beg, ring_beg + ring_n), the next write at ring_pos
    int32_t ring_beg = 0, ring_n = 0, ring_pos = 0;
    std::vector<llama_token> ring_tok;
    // the family blocks: slots [fam_beg, fam_beg + n_blocks * block); a unit is a family's tokens outside the fixed
    // ones, cut into pieces of at most block tokens
    int32_t fam_beg = 0, block = 0, n_blocks = 0;
    std::vector<int32_t>                  unit_of;     // token -> unit, -1 none
    std::vector<std::vector<llama_token>> unit_tokens;
    std::vector<int32_t>                  unit_block;  // unit -> block, -1 not loaded
    std::vector<int32_t>                  block_unit;  // block -> unit, -1 free
    std::vector<llama_token>              block_tok;   // block * block + i -> token, -1 free
    std::vector<uint64_t>                 block_tick;  // last use
    uint64_t tick    = 0;
    int32_t  n_loads = 0;
    llama_token sink = 0; // the row of a free slot: one past the vocabulary, every scatter has that row and drops it

    std::unordered_map<llama_token, int32_t> slot_of;
    std::vector<uint8_t> row0_head, row0_w2; // slot 0's rows, to clear the reserve
    // host copies of the head's and the Markov head's rows (read once at init), so an add is one upload a tensor
    std::vector<uint8_t> host_head, host_w2;
    std::vector<float>   host_b;
    float row0_b = 0.0f;

    ~llama_mod_dict() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static const ggml_tensor * mod_head_of(llama_context * ctx, const llama_model & model) {
    if (model.output) {
        return model.output;
    }
    llama_context * other = ctx->get_cparams().ctx_other;
    return other ? llama_get_model(other)->output : nullptr;
}

// one row (ne[0] elements) of a 2-D tensor: its bytes
static void mod_get_row(const ggml_tensor * t, int64_t row, std::vector<uint8_t> & out) {
    out.resize(t->nb[1]);
    ggml_backend_tensor_get(t, out.data(), row * t->nb[1], t->nb[1]);
}

static void mod_set_slot(llama_mod_dict & d, const llama_model & m, int32_t slot, llama_token tok,
        const uint8_t * head_row, const uint8_t * w2_row, float b) {
    ggml_backend_tensor_set(m.mod_head, head_row, (size_t) slot * m.mod_head->nb[1], m.mod_head->nb[1]);
    ggml_backend_tensor_set(m.mod_w2,   w2_row,   (size_t) slot * m.mod_w2->nb[1],   m.mod_w2->nb[1]);
    if (m.mod_w2_b) {
        ggml_backend_tensor_set(m.mod_w2_b, &b, (size_t) slot * sizeof(float), sizeof(float));
    }
    const int64_t id = tok;
    ggml_backend_tensor_set(m.mod_ids, &id, (size_t) slot * sizeof(int64_t), sizeof(int64_t));
    const float idf = (float) tok;
    ggml_backend_tensor_set(m.mod_ids_f, &idf, (size_t) slot * sizeof(float), sizeof(float));
    const float zero = 0.0f;
    ggml_backend_tensor_set(m.mod_mask, &zero, (size_t) slot * sizeof(float), sizeof(float));
    GGML_UNUSED(d);
}

int32_t llama_dspark_dict_init(llama_context * ctx, const llama_token * tokens, int32_t n_tokens, int32_t n_reserve) {
    llama_model & m = const_cast<llama_model &>(ctx->get_model());
    const ggml_tensor * head = mod_head_of(ctx, m);
    if (!m.dspark_markov_w2 || m.d2t || !head) {
        LLAMA_LOG_WARN("%s: not a DSpark drafter with a head of its target and no d2t: no dictionary\n", __func__);
        return -1;
    }
    const ggml_tensor * w2 = m.dspark_markov_w2;
    const int64_t n_vocab = std::min<int64_t>({ (int64_t) m.vocab.n_tokens(), head->ne[1], w2->ne[1] });
    if (m.dspark_markov_w2_b && m.dspark_markov_w2_b->type != GGML_TYPE_F32) {
        LLAMA_LOG_WARN("%s: the Markov head's bias is not F32: no dictionary\n", __func__);
        return -1;
    }

    // the fixed tokens, in range and once each
    std::vector<llama_token> fixed;
    std::unordered_map<llama_token, int32_t> slot_of;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const llama_token t = tokens[i];
        if (t >= 0 && t < n_vocab && slot_of.emplace(t, (int32_t) fixed.size()).second) {
            fixed.push_back(t);
        }
    }
    if (fixed.empty()) {
        return -1;
    }
    const int32_t K = (int32_t) fixed.size() + std::max(0, n_reserve);
    // free slots map to a sink row one past the vocabulary: every scatter writes into n_vocab + 1 rows and keeps the
    // first n_vocab, so a free slot never shares a row with a real token (with -inf in a free slot and a logit in
    // slot 0, which write lands on a shared row is up to the GPU)
    const llama_token sink = (llama_token) m.vocab.n_tokens();

    auto d = std::make_shared<llama_mod_dict>();
    ggml_init_params ip = { 7 * ggml_tensor_overhead(), nullptr, true };
    d->ctx = ggml_init(ip);
    ggml_tensor * t_head = ggml_new_tensor_2d(d->ctx, head->type, head->ne[0], K);
    ggml_tensor * t_w2   = ggml_new_tensor_2d(d->ctx, w2->type, w2->ne[0], K);
    ggml_tensor * t_b    = m.dspark_markov_w2_b ? ggml_new_tensor_1d(d->ctx, GGML_TYPE_F32, K) : nullptr;
    ggml_tensor * t_ids  = ggml_new_tensor_1d(d->ctx, GGML_TYPE_I64, K);
    ggml_tensor * t_idf  = ggml_new_tensor_2d(d->ctx, GGML_TYPE_F32, 1, K);
    ggml_tensor * t_mask = ggml_new_tensor_1d(d->ctx, GGML_TYPE_F32, K);
    ggml_set_name(t_head, "mod_head");
    ggml_set_name(t_w2, "mod_w2");
    if (t_b) {
        ggml_set_name(t_b, "mod_w2_b");
    }
    ggml_set_name(t_ids, "mod_ids");
    ggml_set_name(t_idf, "mod_ids_f");
    ggml_set_name(t_mask, "mod_mask");
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(head->buffer);
    d->buf = ggml_backend_alloc_ctx_tensors_from_buft(d->ctx, buft);
    if (!d->buf) {
        LLAMA_LOG_WARN("%s: no memory for a dictionary of %d slots\n", __func__, K);
        return -1;
    }
    ggml_backend_buffer_set_usage(d->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // the rows, read once from the head and the Markov head (the whole tensors: one transfer each)
    std::vector<uint8_t> all_head(ggml_nbytes(head)), all_w2(ggml_nbytes(w2));
    ggml_backend_tensor_get(head, all_head.data(), 0, all_head.size());
    ggml_backend_tensor_get(w2,   all_w2.data(),   0, all_w2.size());
    std::vector<float> all_b;
    if (t_b) {
        all_b.resize(ggml_nelements(m.dspark_markov_w2_b));
        ggml_backend_tensor_get(m.dspark_markov_w2_b, all_b.data(), 0, all_b.size() * sizeof(float));
    }
    std::vector<uint8_t> g_head((size_t) K * t_head->nb[1]), g_w2((size_t) K * t_w2->nb[1]);
    std::vector<float>   g_b(t_b ? K : 0);
    std::vector<int64_t> g_ids(K);
    std::vector<float>   g_idf(K);
    std::vector<float>   g_mask(K, 0.0f);
    for (int32_t s = 0; s < K; ++s) {
        const llama_token t = fixed[s < (int32_t) fixed.size() ? s : 0]; // the reserve: copies of slot 0
        memcpy(g_head.data() + (size_t) s * t_head->nb[1], all_head.data() + (size_t) t * head->nb[1], t_head->nb[1]);
        memcpy(g_w2.data()   + (size_t) s * t_w2->nb[1],   all_w2.data()   + (size_t) t * w2->nb[1],   t_w2->nb[1]);
        if (t_b) {
            g_b[s] = all_b[t];
        }
        g_ids[s] = s < (int32_t) fixed.size() ? t : sink;
        g_idf[s] = (float) g_ids[s];
        g_mask[s] = s < (int32_t) fixed.size() ? 0.0f : -INFINITY;
    }
    ggml_backend_tensor_set(t_head, g_head.data(), 0, g_head.size());
    ggml_backend_tensor_set(t_w2,   g_w2.data(),   0, g_w2.size());
    if (t_b) {
        ggml_backend_tensor_set(t_b, g_b.data(), 0, g_b.size() * sizeof(float));
    }
    ggml_backend_tensor_set(t_ids, g_ids.data(), 0, g_ids.size() * sizeof(int64_t));
    ggml_backend_tensor_set(t_idf, g_idf.data(), 0, g_idf.size() * sizeof(float));
    ggml_backend_tensor_set(t_mask, g_mask.data(), 0, g_mask.size() * sizeof(float));

    d->src_head = head;
    d->src_w2   = w2;
    d->src_w2_b = m.dspark_markov_w2_b;
    d->n_fixed  = (int32_t) fixed.size();
    d->n_slots  = K;
    d->ring_beg = d->n_fixed;
    d->ring_n   = K - d->n_fixed;
    d->fam_beg  = K;
    d->ring_tok.assign(d->ring_n, -1);
    d->sink     = sink;
    d->slot_of  = std::move(slot_of);
    d->row0_head.assign(g_head.begin(), g_head.begin() + t_head->nb[1]);
    d->row0_w2.assign(g_w2.begin(), g_w2.begin() + t_w2->nb[1]);
    d->row0_b = t_b ? g_b[0] : 0.0f;
    d->host_head = std::move(all_head);
    d->host_w2   = std::move(all_w2);
    d->host_b    = std::move(all_b);

    m.mod_head = t_head;
    m.mod_w2   = t_w2;
    m.mod_w2_b = t_b;
    m.mod_ids  = t_ids;
    m.mod_ids_f = t_idf;
    m.mod_mask  = t_mask;
    m.mod      = d;

    ctx->graph_invalidate();

    LLAMA_LOG_INFO("%s: MoD: %d fixed tokens + %d reserve slots of %lld, %.1f MiB on %s\n", __func__, d->n_fixed,
        K - d->n_fixed, (long long) n_vocab, ggml_backend_buffer_get_size(d->buf) / 1048576.0, ggml_backend_buft_name(buft));
    return d->n_fixed;
}

// slots [s0, s0 + k) to the tokens toks (-1: a free slot), one upload a tensor from the host copies of the rows
static void mod_write(llama_mod_dict & d, const llama_model & m, int32_t s0, const llama_token * toks, int32_t k) {
    if (k <= 0) {
        return;
    }
    const size_t rh = m.mod_head->nb[1], rw = m.mod_w2->nb[1];
    std::vector<uint8_t> h((size_t) k * rh), w((size_t) k * rw);
    std::vector<float>   b(k), idf(k), mask(k);
    std::vector<int64_t> ids(k);
    for (int32_t i = 0; i < k; ++i) {
        const llama_token t = toks[i];
        if (t >= 0) {
            memcpy(h.data() + (size_t) i * rh, d.host_head.data() + (size_t) t * d.src_head->nb[1], rh);
            memcpy(w.data() + (size_t) i * rw, d.host_w2.data()   + (size_t) t * d.src_w2->nb[1],   rw);
            b[i]    = d.host_b.empty() ? 0.0f : d.host_b[t];
            ids[i]  = t;
            mask[i] = 0.0f;
        } else {
            memcpy(h.data() + (size_t) i * rh, d.row0_head.data(), rh);
            memcpy(w.data() + (size_t) i * rw, d.row0_w2.data(),   rw);
            b[i]    = d.row0_b;
            ids[i]  = d.sink;
            mask[i] = -INFINITY;
        }
        idf[i] = (float) ids[i];
    }
    const size_t o = (size_t) s0;
    ggml_backend_tensor_set(m.mod_head, h.data(), o * rh, h.size());
    ggml_backend_tensor_set(m.mod_w2,   w.data(), o * rw, w.size());
    if (m.mod_w2_b) {
        ggml_backend_tensor_set(m.mod_w2_b, b.data(), o * sizeof(float), b.size() * sizeof(float));
    }
    ggml_backend_tensor_set(m.mod_ids,   ids.data(),  o * sizeof(int64_t), ids.size() * sizeof(int64_t));
    ggml_backend_tensor_set(m.mod_ids_f, idf.data(),  o * sizeof(float),   idf.size() * sizeof(float));
    ggml_backend_tensor_set(m.mod_mask,  mask.data(), o * sizeof(float),   mask.size() * sizeof(float));
}

static void mod_drop(llama_mod_dict & d, llama_token t, int32_t slot) {
    if (t < 0) {
        return;
    }
    auto it = d.slot_of.find(t);
    if (it != d.slot_of.end() && it->second == slot) {
        d.slot_of.erase(it);
    }
}

// the tokens into the ring, the oldest out: at most two uploads a tensor (the ring wraps once at most)
static int32_t mod_ring_add(llama_mod_dict & d, const llama_model & m, std::vector<llama_token> & add) {
    if (d.ring_n <= 0 || add.empty()) {
        return 0;
    }
    if ((int32_t) add.size() > d.ring_n) {
        add.erase(add.begin(), add.end() - d.ring_n); // the latest ring_n
    }
    const int32_t p0 = d.ring_pos, k = (int32_t) add.size();
    for (int32_t i = 0; i < k; ++i) {
        const int32_t p = (p0 + i) % d.ring_n, slot = d.ring_beg + p;
        mod_drop(d, d.ring_tok[p], slot);
        d.ring_tok[p] = add[i];
        d.slot_of[add[i]] = slot;
    }
    d.ring_pos = (p0 + k) % d.ring_n;
    const int32_t k1 = std::min(k, d.ring_n - p0);
    mod_write(d, m, d.ring_beg + p0, d.ring_tok.data() + p0, k1);
    mod_write(d, m, d.ring_beg, d.ring_tok.data(), k - k1);
    return k;
}

// unit u into the least recently used block (its old unit out); tokens in slots already or waiting for the ring stay
// where they are
static int32_t mod_load_unit(llama_mod_dict & d, const llama_model & m, int32_t u, const std::unordered_set<llama_token> & pending) {
    int32_t b = 0;
    for (int32_t i = 1; i < d.n_blocks; ++i) {
        if (d.block_tick[i] < d.block_tick[b]) {
            b = i;
        }
    }
    llama_token * bt = d.block_tok.data() + (size_t) b * d.block;
    const int32_t s0 = d.fam_beg + b * d.block;
    if (d.block_unit[b] >= 0) {
        d.unit_block[d.block_unit[b]] = -1;
    }
    for (int32_t i = 0; i < d.block; ++i) {
        mod_drop(d, bt[i], s0 + i);
        bt[i] = -1;
    }
    int32_t k = 0;
    for (const llama_token t : d.unit_tokens[u]) {
        if (!d.slot_of.count(t) && !pending.count(t)) {
            bt[k] = t;
            d.slot_of[t] = s0 + k;
            ++k;
        }
    }
    mod_write(d, m, s0, bt, d.block);
    d.block_unit[b] = u;
    d.unit_block[u] = b;
    d.block_tick[b] = ++d.tick;
    ++d.n_loads;
    return k;
}

int32_t llama_dspark_dict_add(llama_context * ctx, const llama_token * tokens, int32_t n_tokens, bool families) {
    llama_model & m = const_cast<llama_model &>(ctx->get_model());
    if (!m.mod) {
        return -1;
    }
    llama_mod_dict & d = *m.mod;
    const int64_t n_vocab = std::min<int64_t>({ (int64_t) m.vocab.n_tokens(), d.src_head->ne[1], d.src_w2->ne[1] });
    std::vector<llama_token> ring;
    std::unordered_set<llama_token> pending;
    int32_t n_new = 0;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const llama_token t = tokens[i];
        if (t < 0 || t >= n_vocab) {
            continue;
        }
        auto it = d.slot_of.find(t);
        if (it != d.slot_of.end()) {
            if (it->second >= d.fam_beg && d.n_blocks > 0) {
                d.block_tick[(it->second - d.fam_beg) / d.block] = ++d.tick; // its family is in use
            }
            continue;
        }
        if (pending.count(t)) {
            continue;
        }
        if (families && d.n_blocks > 0 && (size_t) t < d.unit_of.size() && d.unit_of[t] >= 0 && d.unit_block[d.unit_of[t]] < 0) {
            n_new += mod_load_unit(d, m, d.unit_of[t], pending);
            if (d.slot_of.count(t)) {
                continue;
            }
        }
        pending.insert(t);
        ring.push_back(t);
    }
    n_new += mod_ring_add(d, m, ring);
    static const bool check = getenv("LLAMA_MOD_CHECK") != nullptr;
    if (check) {
        // DIAG: the slots on the device against the host's map: every token in one slot at most, free slots masked
        std::vector<int64_t> ids(d.n_slots);
        std::vector<float>   mask(d.n_slots), idf(d.n_slots);
        ggml_backend_tensor_get(m.mod_ids,   ids.data(),  0, ids.size() * sizeof(int64_t));
        ggml_backend_tensor_get(m.mod_mask,  mask.data(), 0, mask.size() * sizeof(float));
        ggml_backend_tensor_get(m.mod_ids_f, idf.data(),  0, idf.size() * sizeof(float));
        std::unordered_set<int64_t> seen;
        size_t n_live = 0;
        for (int32_t s = 0; s < d.n_slots; ++s) {
            const bool live = ids[s] != d.sink;
            GGML_ASSERT(live == (mask[s] == 0.0f) && (float) ids[s] == idf[s]);
            if (live) {
                GGML_ASSERT(seen.insert(ids[s]).second && "MoD: a token in two slots");
                auto it = d.slot_of.find((llama_token) ids[s]);
                GGML_ASSERT(it != d.slot_of.end() && it->second == s);
                ++n_live;
            }
        }
        GGML_ASSERT(n_live == d.slot_of.size());
        static int n_checks = 0;
        if (++n_checks % 200 == 0) {
            LLAMA_LOG_INFO("%s: MoD check %d: %zu live slots of %d, %d family loads\n", __func__, n_checks, n_live, d.n_slots, d.n_loads);
        }
    }
    return n_new;
}

int32_t llama_dspark_dict_families(llama_context * ctx, const int32_t * fam_of, int32_t n_fam_of, int32_t n_slots, int32_t block) {
    llama_model & m = const_cast<llama_model &>(ctx->get_model());
    if (!m.mod || block <= 0) {
        return -1;
    }
    llama_mod_dict & d = *m.mod;
    llama_dspark_dict_reset(ctx);
    const int32_t n_res = d.n_slots - d.n_fixed;
    const int32_t n_blocks = std::min(n_slots, n_res) / block;
    if (n_blocks <= 0) {
        return -1;
    }
    // the units: each family's tokens outside the fixed ones, in pieces of at most block tokens
    int32_t n_fam = 0;
    for (int32_t t = 0; t < n_fam_of; ++t) {
        n_fam = std::max(n_fam, fam_of[t] + 1);
    }
    std::vector<std::vector<llama_token>> fam(n_fam);
    for (int32_t t = 0; t < n_fam_of; ++t) {
        if (fam_of[t] >= 0 && !d.slot_of.count(t)) {
            fam[fam_of[t]].push_back(t);
        }
    }
    d.unit_of.assign(n_fam_of, -1);
    d.unit_tokens.clear();
    for (const auto & f : fam) {
        for (size_t i = 0; i < f.size(); i += block) {
            d.unit_tokens.emplace_back(f.begin() + i, f.begin() + std::min(f.size(), i + block));
            for (const llama_token t : d.unit_tokens.back()) {
                d.unit_of[t] = (int32_t) d.unit_tokens.size() - 1;
            }
        }
    }
    d.block      = block;
    d.n_blocks   = n_blocks;
    d.ring_n     = n_res - n_blocks * block;
    d.fam_beg    = d.ring_beg + d.ring_n;
    d.ring_pos   = 0;
    d.ring_tok.assign(d.ring_n, -1);
    d.unit_block.assign(d.unit_tokens.size(), -1);
    d.block_unit.assign(n_blocks, -1);
    d.block_tok.assign((size_t) n_blocks * block, -1);
    d.block_tick.assign(n_blocks, 0);
    LLAMA_LOG_INFO("%s: MoD: %d families in %zu units, %d blocks of %d slots, %d slots for single tokens\n", __func__,
        n_fam, d.unit_tokens.size(), n_blocks, block, d.ring_n);
    return (int32_t) d.unit_tokens.size();
}

int32_t llama_dspark_dict_n_loads(const llama_context * ctx) {
    const llama_model & m = ctx->get_model();
    return m.mod ? m.mod->n_loads : -1;
}

void llama_dspark_dict_reset(llama_context * ctx) {
    llama_model & m = const_cast<llama_model &>(ctx->get_model());
    if (!m.mod) {
        return;
    }
    llama_mod_dict & d = *m.mod;
    // every slot past the fixed tokens free, and out of the map
    for (auto it = d.slot_of.begin(); it != d.slot_of.end(); ) {
        if (it->second >= d.n_fixed) {
            it = d.slot_of.erase(it);
        } else {
            ++it;
        }
    }
    std::vector<llama_token> none(d.n_slots - d.n_fixed, -1);
    mod_write(d, m, d.n_fixed, none.data(), (int32_t) none.size());
    d.ring_pos = 0;
    std::fill(d.ring_tok.begin(), d.ring_tok.end(), -1);
    std::fill(d.unit_block.begin(), d.unit_block.end(), -1);
    std::fill(d.block_unit.begin(), d.block_unit.end(), -1);
    std::fill(d.block_tok.begin(), d.block_tok.end(), -1);
    std::fill(d.block_tick.begin(), d.block_tick.end(), 0);
}
