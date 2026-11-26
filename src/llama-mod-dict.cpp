// MoD, the fork's Mixture of Dictionaries: a DSpark drafter drafts over a dictionary of tokens (see llama.h).
// The dictionary's rows of the output head (the target's, borrowed) and of the Markov head's vocabulary matrix are
// copied as they are into tensors of their own on the head's device, K = the fixed tokens + reserve slots; the graph
// (dflash.cpp build_logits, dspark.cpp's Markov head) multiplies by those and scatters the logits back to the
// vocabulary through the slots' token ids, -inf elsewhere, so sampling and the host see the whole vocabulary as before.
// DSpark's Markov chain stays in slot space (its TOP_K / ARGMAX over the K slots, the picks mapped to token ids through
// mod_ids_f) and scatters once, for the sampler. Free slots are -inf (mod_mask) and scatter to a sink row past the
// vocabulary that every scatter drops.
// Free reserve slots hold a copy of slot 0 (the same token, the same logit). One dictionary a drafter model.

#include "llama-impl.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-vocab.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

struct llama_mod_dict {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    const ggml_tensor * src_head = nullptr; // the rows are copied from these
    const ggml_tensor * src_w2   = nullptr;
    const ggml_tensor * src_w2_b = nullptr;

    int32_t n_fixed = 0;
    int32_t n_slots = 0;
    int32_t n_used  = 0;
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
    d->n_used   = d->n_fixed;
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

int32_t llama_dspark_dict_add(llama_context * ctx, const llama_token * tokens, int32_t n_tokens) {
    llama_model & m = const_cast<llama_model &>(ctx->get_model());
    if (!m.mod) {
        return -1;
    }
    llama_mod_dict & d = *m.mod;
    const int64_t n_vocab = std::min<int64_t>({ (int64_t) m.vocab.n_tokens(), d.src_head->ne[1], d.src_w2->ne[1] });
    // the new tokens, once each, into the next free slots (consecutive): one upload a tensor from the host copies
    std::vector<llama_token> add;
    for (int32_t i = 0; i < n_tokens; ++i) {
        const llama_token t = tokens[i];
        if (t < 0 || t >= n_vocab || d.slot_of.count(t)) {
            continue;
        }
        if (d.n_used + (int32_t) add.size() >= d.n_slots) {
            static bool warned = false;
            if (!warned) {
                LLAMA_LOG_WARN("%s: MoD: the %d reserve slots are full, later tokens stay out until a reset\n", __func__, d.n_slots - d.n_fixed);
                warned = true;
            }
            break;
        }
        d.slot_of.emplace(t, d.n_used + (int32_t) add.size());
        add.push_back(t);
    }
    const int32_t k = (int32_t) add.size();
    if (k == 0) {
        return 0;
    }
    const size_t rh = m.mod_head->nb[1], rw = m.mod_w2->nb[1];
    std::vector<uint8_t> h((size_t) k * rh), w((size_t) k * rw);
    std::vector<float>   b(k), idf(k), mask(k, 0.0f);
    std::vector<int64_t> ids(k);
    for (int32_t i = 0; i < k; ++i) {
        const llama_token t = add[i];
        memcpy(h.data() + (size_t) i * rh, d.host_head.data() + (size_t) t * d.src_head->nb[1], rh);
        memcpy(w.data() + (size_t) i * rw, d.host_w2.data()   + (size_t) t * d.src_w2->nb[1],   rw);
        b[i]   = d.host_b.empty() ? 0.0f : d.host_b[t];
        ids[i] = t;
        idf[i] = (float) t;
    }
    const size_t s0 = (size_t) d.n_used;
    ggml_backend_tensor_set(m.mod_head, h.data(), s0 * rh, h.size());
    ggml_backend_tensor_set(m.mod_w2,   w.data(), s0 * rw, w.size());
    if (m.mod_w2_b) {
        ggml_backend_tensor_set(m.mod_w2_b, b.data(), s0 * sizeof(float), b.size() * sizeof(float));
    }
    ggml_backend_tensor_set(m.mod_ids,   ids.data(),  s0 * sizeof(int64_t), ids.size() * sizeof(int64_t));
    ggml_backend_tensor_set(m.mod_ids_f, idf.data(),  s0 * sizeof(float),   idf.size() * sizeof(float));
    ggml_backend_tensor_set(m.mod_mask,  mask.data(), s0 * sizeof(float),   mask.size() * sizeof(float));
    d.n_used += k;
    return k;
}

void llama_dspark_dict_reset(llama_context * ctx) {
    llama_model & m = const_cast<llama_model &>(ctx->get_model());
    if (!m.mod) {
        return;
    }
    llama_mod_dict & d = *m.mod;
    if (d.n_used == d.n_fixed) {
        return;
    }
    // the used reserve slots back to copies of slot 0, and out of the map
    const int64_t id0 = d.sink;
    for (auto it = d.slot_of.begin(); it != d.slot_of.end(); ) {
        if (it->second >= d.n_fixed) {
            it = d.slot_of.erase(it);
        } else {
            ++it;
        }
    }
    const int32_t n = d.n_used - d.n_fixed;
    std::vector<uint8_t> h((size_t) n * m.mod_head->nb[1]), w((size_t) n * m.mod_w2->nb[1]);
    std::vector<float>   b(n, d.row0_b);
    std::vector<int64_t> ids(n, id0);
    for (int32_t i = 0; i < n; ++i) {
        memcpy(h.data() + (size_t) i * m.mod_head->nb[1], d.row0_head.data(), m.mod_head->nb[1]);
        memcpy(w.data() + (size_t) i * m.mod_w2->nb[1],   d.row0_w2.data(),   m.mod_w2->nb[1]);
    }
    ggml_backend_tensor_set(m.mod_head, h.data(), (size_t) d.n_fixed * m.mod_head->nb[1], h.size());
    ggml_backend_tensor_set(m.mod_w2,   w.data(), (size_t) d.n_fixed * m.mod_w2->nb[1],   w.size());
    if (m.mod_w2_b) {
        ggml_backend_tensor_set(m.mod_w2_b, b.data(), (size_t) d.n_fixed * sizeof(float), b.size() * sizeof(float));
    }
    ggml_backend_tensor_set(m.mod_ids, ids.data(), (size_t) d.n_fixed * sizeof(int64_t), ids.size() * sizeof(int64_t));
    std::vector<float> idf(n, (float) id0);
    ggml_backend_tensor_set(m.mod_ids_f, idf.data(), (size_t) d.n_fixed * sizeof(float), idf.size() * sizeof(float));
    std::vector<float> mask(n, -INFINITY);
    ggml_backend_tensor_set(m.mod_mask, mask.data(), (size_t) d.n_fixed * sizeof(float), mask.size() * sizeof(float));
    d.n_used = d.n_fixed;
}
