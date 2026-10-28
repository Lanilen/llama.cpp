// records the experts selected by the router of every MoE layer for every token of a prompt and a generation
// output: <out>.npy (int16 [n_tokens, n_moe_layers, n_expert_used]) and <out>.json (model and run info, the tokens)
// LLAMA_MOE_TRACE_HIDDEN=16,32 also saves the inputs of those layers (float16) for every
// LLAMA_MOE_TRACE_HIDDEN_STRIDE-th prompt token (default 16) and around each generated segment's start:
// <out>.hid.npy [n_rows, n_layers, n_embd], the rows' positions in the json's "hidden"

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "../../src/llama-ext.h" // staging API: llama_set_embeddings_layer_inp / llama_get_embeddings_layer_inp

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct moe_layer_info {
    int64_t n_expert = 0;
    std::map<std::string, size_t> expert_bytes; // expert tensor -> bytes per expert
    std::map<std::string, std::string> types;
    std::vector<int32_t> ids;                   // [n_tokens, n_expert_used]
};

struct moe_trace_data {
    bool enabled = true;
    int64_t n_expert_used = 0;
    std::map<int, moe_layer_info> layers;
    std::vector<int32_t> buf;
};

// parse the layer index of a "name-il" tensor
static bool moe_trace_layer(const char * name, const char * prefix, int & il) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') {
        return false;
    }
    char * end = nullptr;
    il = (int) strtol(name + n + 1, &end, 10);
    return end != name + n + 1 && *end == '\0';
}

static bool moe_trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = (moe_trace_data *) user_data;
    if (!data->enabled) {
        return !ask;
    }

    int il = -1;
    const bool is_topk = moe_trace_layer(t->name, "ffn_moe_topk", il);

    if (ask) {
        // the expert tensors are seen when the graph is planned, their data is not needed
        // with --moe-cache-mib they may be views of the cache, so the number of experts comes from the router
        int l = -1;
        if (moe_trace_layer(t->name, "ffn_moe_probs", l)) {
            data->layers[l].n_expert = t->ne[0];
        } else if (t->op == GGML_OP_MUL_MAT_ID && strncmp(t->name, "ffn_moe_", 8) == 0) {
            const char * dash = strrchr(t->name, '-');
            if (dash != nullptr) {
                const ggml_tensor * w = t->src[0];
                // large batches may read a scheduler copy, "<backend>#<name>#<n>"
                std::string name = w->name;
                const size_t hash = name.find('#');
                if (hash != std::string::npos) {
                    name = name.substr(hash + 1, name.find('#', hash + 1) - hash - 1);
                }
                if (name.rfind("moe_cache.", 0) == 0) {
                    name = name.substr(strlen("moe_cache."));
                }
                auto & info = data->layers[atoi(dash + 1)];
                info.expert_bytes[name] = w->nb[2];
                info.types[name] = ggml_type_name(w->type);
            }
        }
        return is_topk;
    }

    if (!is_topk) {
        return true;
    }

    GGML_ASSERT(t->type == GGML_TYPE_I32);
    const int64_t n_used   = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    if (data->n_expert_used == 0) {
        data->n_expert_used = n_used;
    }
    GGML_ASSERT(data->n_expert_used == n_used);

    // the selection is a view of the argsort rows, so read it row by row
    auto & ids = data->layers[il].ids;
    const size_t start = ids.size();
    ids.resize(start + n_tokens*n_used);
    for (int64_t i = 0; i < n_tokens; ++i) {
        ggml_backend_tensor_get(t, ids.data() + start + i*n_used, i*t->nb[1], n_used*sizeof(int32_t));
    }

    return true;
}

static bool write_npy(const std::string & path, const char * descr, const void * data, size_t elem_size, const std::vector<size_t> & shape) {
    size_t n = 1;
    std::string shape_str = "(";
    for (size_t s : shape) {
        shape_str += std::to_string(s) + ", ";
    }
    shape_str += ")";
    for (size_t s : shape) {
        n *= s;
    }
    std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape_str + ", }";
    // magic (6) + version (2) + header length (2) + header, padded with spaces to a multiple of 64 and ended by a newline
    const size_t total = 10 + header.size() + 1;
    header += std::string((64 - total % 64) % 64, ' ') + "\n";

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const uint8_t prefix[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    const uint16_t hlen = (uint16_t) header.size();
    fwrite(prefix, 1, sizeof(prefix), f);
    fwrite(&hlen, 1, sizeof(hlen), f);
    fwrite(header.data(), 1, header.size(), f);
    fwrite(data, elem_size, n, f);
    return fclose(f) == 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MOE_TRACE)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // LLAMA_MOE_TRACE_OFF=1 runs the same generation without observing the graph, to check that tracing does not change it and to time it
    moe_trace_data data;
    data.enabled = getenv("LLAMA_MOE_TRACE_OFF") == nullptr;

    // without tracing the callback is not set: with one, the scheduler waits for each split to finish
    if (data.enabled) {
        params.cb_eval = moe_trace_cb_eval;
        params.cb_eval_user_data = &data;
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

    std::vector<int> hid_layers;
    int hid_stride = 16;
    if (const char * env = getenv("LLAMA_MOE_TRACE_HIDDEN")) {
        std::stringstream ss(env);
        for (std::string item; std::getline(ss, item, ',');) {
            const int il = atoi(item.c_str());
            if (il < 0 || il >= llama_model_n_layer(model)) {
                LOG_ERR("%s: LLAMA_MOE_TRACE_HIDDEN: no layer %d\n", __func__, il);
                return 1;
            }
            hid_layers.push_back(il);
            llama_set_embeddings_layer_inp(ctx, (uint32_t) il, true);
        }
        if (const char * st = getenv("LLAMA_MOE_TRACE_HIDDEN_STRIDE")) {
            hid_stride = std::max(1, atoi(st));
        }
    }
    const int64_t n_embd = llama_model_n_embd(model);
    std::vector<ggml_fp16_t> hid;
    std::vector<int> hid_pos;
    std::set<int> hid_extra;   // the last prompt token before and the first token of each generated segment

    std::string prompt = params.prompt;
    auto chat_templates = common_chat_templates_init(model, params.chat_template);
    const bool use_chat = params.conversation_mode != COMMON_CONVERSATION_MODE_DISABLED && params.enable_chat_template &&
        llama_model_chat_template(model, nullptr) != nullptr;
    if (use_chat) {
        common_chat_templates_inputs inputs;
        inputs.use_jinja = params.use_jinja;
        if (!params.system_prompt.empty()) {
            common_chat_msg msg;
            msg.role    = "system";
            msg.content = params.system_prompt;
            inputs.messages.push_back(msg);
        }
        common_chat_msg msg;
        msg.role    = "user";
        msg.content = params.prompt;
        inputs.messages.push_back(msg);
        inputs.add_generation_prompt = true;
        prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt.str();
    }

    // --conversation: the messages are traced as one sequence, each message a segment of prompt or generated tokens
    const bool conversation = !params.moe_trace_conversation.empty();
    json segments = json::array();
    std::vector<llama_token> tokens;
    if (conversation) {
        std::ifstream fconv(params.moe_trace_conversation);
        if (!fconv) {
            LOG_ERR("%s: failed to read %s\n", __func__, params.moe_trace_conversation.c_str());
            return 1;
        }
        const json conv = json::parse(fconv);
        const json & msgs = conv.is_object() ? conv.at("messages") : conv;
        std::vector<common_chat_msg> all;
        if (!params.system_prompt.empty()) {
            common_chat_msg msg;
            msg.role    = "system";
            msg.content = params.system_prompt;
            all.push_back(msg);
        }
        for (const auto & m : msgs) {
            common_chat_msg msg;
            msg.role    = m.at("role").get<std::string>();
            msg.content = m.at("content").get<std::string>();
            all.push_back(msg);
        }
        auto render = [&](size_t n, bool add_generation_prompt) {
            common_chat_templates_inputs inputs;
            inputs.use_jinja = params.use_jinja;
            inputs.messages.assign(all.begin(), all.begin() + n);
            inputs.add_generation_prompt = add_generation_prompt;
            return common_tokenize(ctx, common_chat_templates_apply(chat_templates.get(), inputs).prompt.str(), true, true);
        };
        tokens = render(all.size(), false);

        // a message starts where the prompt before it ends, for an assistant message after the generation prompt
        std::vector<size_t> starts;
        for (size_t k = 0; k < all.size(); ++k) {
            const bool assistant = all[k].role == "assistant";
            if (k == 0 && !assistant) {
                starts.push_back(0);
                continue;
            }
            const auto prefix = render(k, assistant);
            if (prefix.size() > tokens.size() || !std::equal(prefix.begin(), prefix.end(), tokens.begin()) ||
                    (!starts.empty() && prefix.size() < starts.back())) {
                LOG_WRN("%s: the template renders message %zu differently in the whole conversation, its segment is approximate\n", __func__, k);
            }
            starts.push_back(std::min(prefix.size(), tokens.size()));
        }
        for (size_t k = 0; k < all.size(); ++k) {
            const size_t end = k + 1 < all.size() ? starts[k + 1] : tokens.size();
            segments.push_back({ starts[k], std::max(end, starts[k]), all[k].role == "assistant" ? "generated" : "prompt" });
            if (all[k].role == "assistant" && starts[k] > 0) {
                hid_extra.insert((int) starts[k] - 1);
                hid_extra.insert((int) starts[k]);
            }
        }
    } else {
        tokens = common_tokenize(ctx, prompt, true, true);
    }
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt, use -p or -f\n", __func__);
        return 1;
    }
    const int n_prompt = (int) tokens.size();
    if (conversation) {
        params.n_predict = 0;
    }
    if (n_prompt + std::max(params.n_predict, 0) > (int) llama_n_ctx(ctx)) {
        LOG_ERR("%s: %d prompt tokens + %d to predict do not fit in the context of %d, use -c\n", __func__, n_prompt, params.n_predict, llama_n_ctx(ctx));
        return 1;
    }
    LOG_INF("%s: %d %s tokens, chat template %s\n", __func__, n_prompt, conversation ? "conversation" : "prompt", use_chat || conversation ? "applied" : "not applied");

    common_sampler_ptr smpl(common_sampler_init(model, params.sampling));
    for (llama_token id : tokens) {
        common_sampler_accept(smpl.get(), id, false);
    }

    const int64_t t_start = ggml_time_us();

    // only the last token of each batch is an output, the layers that skip the other prompt tokens have rows only for these
    std::vector<int> output_rows;
    for (int i = 0; i < n_prompt; i += params.n_batch) {
        const int n = std::min(params.n_batch, n_prompt - i);
        output_rows.push_back(i + n - 1);
        common_batch batch = common_batch_get_one(ctx, tokens.data() + i, n);
        // a conversation has every token as an output, so no layer skips one and each token has all its selections
        for (int j = 0; conversation && j < n; ++j) {
            batch.set_output(j, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: failed to process the prompt\n", __func__);
            return 1;
        }
        // the layer inputs cover every token of the batch, outputs or not
        for (int j = 0; !hid_layers.empty() && j < n; ++j) {
            if ((i + j) % hid_stride != 0 && hid_extra.count(i + j) == 0) {
                continue;
            }
            hid_pos.push_back(i + j);
            for (int il : hid_layers) {
                const float * src = llama_get_embeddings_layer_inp(ctx, (uint32_t) il);
                GGML_ASSERT(src != nullptr);
                hid.resize(hid.size() + n_embd);
                ggml_fp32_to_fp16_row(src + (size_t) j*n_embd, hid.data() + hid.size() - n_embd, n_embd);
            }
        }
    }

    const int64_t t_prompt = ggml_time_us();

    // every generated token is processed once, so it gets a row of the trace too
    std::vector<llama_token> generated;
    std::string text;
    bool eog = false;
    for (int i = 0; i < params.n_predict; ++i) {
        const llama_token id = common_sampler_sample(smpl.get(), ctx, -1);
        common_sampler_accept(smpl.get(), id, true);
        if (llama_vocab_is_eog(vocab, id)) {
            eog = true;
            break;
        }
        generated.push_back(id);
        text += common_token_to_piece(ctx, id);

        common_batch batch = common_batch_get_one(ctx, &id, 1);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            LOG_ERR("%s: failed to process token %d\n", __func__, i);
            return 1;
        }
    }

    const int64_t t_end = ggml_time_us();
    const int n_gen = (int) generated.size();
    const int n_tokens = n_prompt + n_gen;
    for (int i = n_prompt; i < n_tokens; ++i) {
        output_rows.push_back(i);
    }

    LOG_INF("%s: prompt %.2f s, generated %d tokens in %.2f s (%.2f t/s)%s\n", __func__,
        (t_prompt - t_start)/1e6, n_gen, (t_end - t_prompt)/1e6, n_gen/((t_end - t_prompt)/1e6), eog ? ", stopped at end of generation" : "");
    LOG("%s\n", text.c_str());

    std::vector<int> moe_layers;
    for (const auto & [il, info] : data.layers) {
        if (!info.ids.empty()) {
            moe_layers.push_back(il);
        }
    }

    json meta;
    char buf[512];
    meta["model_path"] = params.model.path;
    meta["model_name"] = llama_model_meta_val_str(model, "general.name", buf, sizeof(buf)) >= 0 ? std::string(buf) : std::string();
    llama_model_desc(model, buf, sizeof(buf));
    meta["model_desc"] = buf;
    meta["n_layer"] = llama_model_n_layer(model);
    meta["n_expert_used"] = data.n_expert_used;
    meta["n_prompt"] = n_prompt;
    if (conversation) {
        meta["conversation"] = params.moe_trace_conversation;
        meta["segments"] = segments;
    }
    meta["n_generated"] = n_gen;
    meta["stopped_at_eog"] = eog;
    meta["rows"] = conversation ?
        "row i is the token at position i of the conversation, segments give [start, end, prompt|generated] per message" :
        "row i is the token processed at position i; rows >= n_prompt are generated tokens; -1 where a layer skipped a prompt token (output pruning)";
    meta["chat_template"] = use_chat;
    meta["n_batch"] = params.n_batch;
    meta["n_ubatch"] = params.n_ubatch;
    meta["traced"] = data.enabled;
    meta["prompt_seconds"] = (t_prompt - t_start)/1e6;
    meta["generation_seconds"] = (t_end - t_prompt)/1e6;
    std::string args;
    for (int i = 1; i < argc; ++i) {
        args += (i > 1 ? " " : "") + std::string(argv[i]);
    }
    meta["args"] = args;
    meta["moe_layers"] = moe_layers;
    json layers = json::array();
    for (int il : moe_layers) {
        const auto & info = data.layers.at(il);
        size_t bytes = 0;
        json tensors = json::object();
        for (const auto & [name, b] : info.expert_bytes) {
            bytes += b;
            tensors[name] = { { "type", info.types.at(name) }, { "bytes_per_expert", b } };
        }
        layers.push_back({ { "il", il }, { "n_expert", info.n_expert }, { "expert_bytes", bytes }, { "tensors", tensors } });
    }
    meta["layers"] = layers;
    meta["generated_tokens"] = generated;
    meta["tokens"] = tokens;
    if (!hid_layers.empty()) {
        meta["hidden"] = { { "layers", hid_layers }, { "stride", hid_stride }, { "n_embd", n_embd }, { "positions", hid_pos },
                           { "rows", "layer inputs (the mean of the hyper-connection streams where the model has them), float16" } };
        if (!write_npy(params.out_file + ".hid.npy", "<f2", hid.data(), sizeof(ggml_fp16_t), { hid_pos.size(), hid_layers.size(), (size_t) n_embd })) {
            LOG_ERR("%s: failed to write %s.hid.npy\n", __func__, params.out_file.c_str());
            return 1;
        }
    }
    meta["generated_text"] = text;

    if (data.enabled) {
        const size_t n_used = data.n_expert_used;
        std::vector<int16_t> out(n_tokens*moe_layers.size()*n_used, -1);
        json pruned = json::array();
        for (size_t l = 0; l < moe_layers.size(); ++l) {
            const auto & ids = data.layers.at(moe_layers[l]).ids;
            const size_t n_rows = ids.size()/n_used;
            if (n_rows != (size_t) n_tokens && n_rows != output_rows.size()) {
                LOG_ERR("%s: layer %d has %zu selections, expected %d or %zu\n", __func__, moe_layers[l], n_rows, n_tokens, output_rows.size());
                return 1;
            }
            if (n_rows != (size_t) n_tokens) {
                pruned.push_back(moe_layers[l]);
            }
            for (size_t r = 0; r < n_rows; ++r) {
                const size_t t = n_rows == (size_t) n_tokens ? r : output_rows[r];
                for (size_t k = 0; k < n_used; ++k) {
                    out[(t*moe_layers.size() + l)*n_used + k] = (int16_t) ids[r*n_used + k];
                }
            }
        }
        meta["pruned_layers"] = pruned;
        if (!write_npy(params.out_file + ".npy", "<i2", out.data(), sizeof(int16_t), { (size_t) n_tokens, moe_layers.size(), n_used })) {
            LOG_ERR("%s: failed to write %s.npy\n", __func__, params.out_file.c_str());
            return 1;
        }
    }

    FILE * f = fopen((params.out_file + ".json").c_str(), "w");
    if (!f) {
        LOG_ERR("%s: failed to write %s.json\n", __func__, params.out_file.c_str());
        return 1;
    }
    fprintf(f, "%s\n", meta.dump(2).c_str());
    fclose(f);

    if (data.enabled) {
        LOG_INF("%s: wrote %s.npy and %s.json: %d tokens, %zu MoE layers, %lld experts used\n", __func__,
            params.out_file.c_str(), params.out_file.c_str(), n_tokens, moe_layers.size(), (long long) data.n_expert_used);
    } else {
        LOG_INF("%s: tracing is off, wrote only %s.json\n", __func__, params.out_file.c_str());
    }

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
