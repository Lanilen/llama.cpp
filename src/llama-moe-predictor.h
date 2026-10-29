#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// a learned predictor of the experts a MoE model selects next, for the hot spare (--moe-cache-predictor)
//
// It comes from a file of its own (a GGUF with general.architecture = "moe-predictor", written by an exporter outside
// llama.cpp), so it can be retrained and replaced without code changes. The file says what it reads: the inputs of
// some layers (the residual stream) at the newest token, and count maps of the selections so far (see map_kind); and
// what it predicts: for each horizon H, the distribution of each layer's selections over the next H tokens. It runs
// as a small ggml graph on a backend of its own, the CPU by default.
class llama_moe_predictor {
public:
    // the count maps a predictor file can ask for, each [n_layer][n_expert] selections:
    enum map_kind {
        MAP_LIVE64,      // decayed with a half-life of 64 tokens, over all tokens
        MAP_CTX512,      // the same with a half-life of 512
        MAP_CONTEXT,     // all tokens before the answer
        MAP_TURN,        // the prompt of the current turn (the tokens since the previous answer)
        MAP_PREV_ANSWER, // the previous answer
        MAP_ANSWER,      // the answer so far
        MAP_ANSWER32,    // the answer so far, decayed with a half-life of 32
        MAP_ANSWER8,     // the same with a half-life of 8
        MAP_PRIOR,       // the file's own prior (the caller passes nothing for it)
        MAP_COUNT,
    };
    static const char * map_name(map_kind k);

    // nullptr if the file can't be used for a model of n_layer layers of n_expert experts and n_embd embeddings
    // (logged); device: name of the backend device to run on, empty = CPU
    static std::unique_ptr<llama_moe_predictor> load(const std::string & path, const std::string & device,
                                                     int32_t n_layer, int32_t n_expert, int32_t n_embd);
    ~llama_moe_predictor();

    const std::vector<int32_t>  & hidden_layers() const; // the layers whose input it reads, at the newest token
    const std::vector<map_kind> & maps() const;          // the count maps it reads, in this order
    const std::vector<int32_t>  & horizons() const;      // tokens, one prediction each
    int32_t every() const;                               // generated tokens between predictions

    // maps: one pointer per entry of maps() to n_layer*n_expert counts (layer-major), nullptr for MAP_PRIOR
    // hidden: one pointer per entry of hidden_layers() to the n_embd inputs of that layer at the newest token
    // t: tokens of the answer generated so far (0: right after the prompt)
    // out: per horizon, n_layer*n_expert probabilities (each layer's sum to 1); false if the graph failed
    bool predict(const std::vector<const float *> & maps, const std::vector<const float *> & hidden, int64_t t,
                 std::vector<std::vector<float>> & out);

private:
    llama_moe_predictor();
    struct impl;
    std::unique_ptr<impl> pimpl;
};
