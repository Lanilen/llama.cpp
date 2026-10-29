// checks a predictor file (--moe-cache-predictor, --moe-cache-predictor-start) against the predictions of the model
// it was exported from:
//   llama-moe-predictor-check PREDICTOR.gguf VECTORS [DEVICE]
// VECTORS (written by the exporter): int32 header [n, n_layer, n_expert, n_maps, n_embd, n_hidden_layers, turn,
// n_horizons], then per vector int32 t, f32 maps [n_maps, n_layer, n_expert] (the map named prior is ignored), f32
// hidden [n_hidden_layers * n_embd], f32 turn embedding [n_embd] if turn, and per horizon f32 probabilities
// [n_layer, n_expert]. Prints the largest differences and how the top experts agree.
#include "../../src/llama-moe-predictor.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s PREDICTOR.gguf VECTORS [DEVICE]\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    FILE * f = fopen(argv[2], "rb");
    int32_t hdr[8];
    if (!f || fread(hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    const int n = hdr[0], NL = hdr[1], NE = hdr[2], NF = hdr[3], NEMB = hdr[4], NHL = hdr[5], TURN = hdr[6], NHZ = hdr[7];
    auto p = llama_moe_predictor::load(argv[1], argc > 3 ? argv[3] : "", NL, NE, NEMB);
    if (!p) {
        return 1;
    }
    if ((int) p->maps().size() != NF || (int) p->horizons().size() != NHZ || (int) p->hidden_layers().size() != NHL ||
            p->reads_turn_embd() != (TURN != 0)) {
        fprintf(stderr, "the vectors don't match the predictor file\n");
        return 1;
    }
    double max_abs = 0.0, max_rel = 0.0, agree = 0.0, t_us = 0.0;
    size_t n_agree = 0;
    for (int v = 0; v < n; ++v) {
        int32_t t;
        std::vector<float> maps((size_t) NF*NL*NE), hid((size_t) NHL*NEMB), turn(TURN ? NEMB : 0);
        std::vector<std::vector<float>> want(NHZ, std::vector<float>((size_t) NL*NE)), got;
        bool ok = fread(&t, 4, 1, f) == 1 && fread(maps.data(), 4, maps.size(), f) == maps.size() &&
                  fread(hid.data(), 4, hid.size(), f) == hid.size() && fread(turn.data(), 4, turn.size(), f) == turn.size();
        for (auto & w : want) {
            ok = ok && fread(w.data(), 4, w.size(), f) == w.size();
        }
        if (!ok) {
            fprintf(stderr, "short vectors file\n");
            return 1;
        }
        std::vector<const float *> mp, hp;
        for (int i = 0; i < NF; ++i) {
            mp.push_back(p->maps()[i] == llama_moe_predictor::MAP_PRIOR ? nullptr : maps.data() + (size_t) i*NL*NE);
        }
        for (int i = 0; i < NHL; ++i) {
            hp.push_back(hid.data() + (size_t) i*NEMB);
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (!p->predict(mp, hp, t, got, TURN ? turn.data() : nullptr)) {
            fprintf(stderr, "predict failed\n");
            return 1;
        }
        t_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        for (int h = 0; h < NHZ; ++h) {
            for (int l = 0; l < NL; ++l) {
                const float * a = got[h].data() + (size_t) l*NE;
                const float * b = want[h].data() + (size_t) l*NE;
                for (int e = 0; e < NE; ++e) {
                    max_abs = std::max(max_abs, (double) std::fabs(a[e] - b[e]));
                    max_rel = std::max(max_rel, (double) std::fabs(a[e] - b[e]) / std::max(b[e], 1e-6f));
                }
                // the top 164 of each layer: how many both pick
                std::vector<int> ia(NE), ib(NE);
                std::iota(ia.begin(), ia.end(), 0); std::iota(ib.begin(), ib.end(), 0);
                const int S = std::min(164, NE);
                std::partial_sort(ia.begin(), ia.begin() + S, ia.end(), [&](int x, int y) { return a[x] > a[y]; });
                std::partial_sort(ib.begin(), ib.begin() + S, ib.end(), [&](int x, int y) { return b[x] > b[y]; });
                std::sort(ia.begin(), ia.begin() + S); std::sort(ib.begin(), ib.begin() + S);
                std::vector<int> both;
                std::set_intersection(ia.begin(), ia.begin() + S, ib.begin(), ib.begin() + S, std::back_inserter(both));
                agree += (double) both.size() / S; n_agree++;
            }
        }
    }
    fclose(f);
    printf("%d vectors: largest difference %.3g (relative %.3g), top-164 sets agree on %.2f%% of their experts, %.2f ms per prediction\n",
        n, max_abs, max_rel, 100.0*agree/n_agree, t_us/n/1000.0);
    return max_abs < 1e-3 ? 0 : 2;
}
