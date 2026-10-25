// measures host to device upload speed for each GPU: copy sizes, pinned or not, synchronous, async or in the background
// usage: llama-upload-bench [repetitions]

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

static const size_t MiB = 1024*1024;

// best of n runs of f, in GB/s for bytes per run
static double bench(int n, size_t bytes, const std::function<void()> & f) {
    f(); // warm up
    int64_t best = INT64_MAX;
    for (int i = 0; i < n; ++i) {
        const int64_t t0 = ggml_time_us();
        f();
        best = std::min(best, ggml_time_us() - t0);
    }
    return bytes/1e3/std::max<int64_t>(best, 1);
}

int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 10;
    const size_t sizes[] = { 2*MiB, 16*MiB, 128*MiB };
    const size_t max_size = 128*MiB;

    ggml_time_init();
    ggml_backend_load_all();

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto dev_pinned = (ggml_backend_dev_pinned_buffer_type_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_pinned_buffer_type");
        auto upload_bg        = (ggml_backend_upload_bg_t)        ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_bg");
        auto upload_bg_submit = (ggml_backend_upload_bg_submit_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_bg_submit");
        auto upload_bg_done   = (ggml_backend_upload_bg_done_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_upload_bg_done");

        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, max_size);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        std::vector<uint8_t> unpinned(max_size, 1);
        ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(dev);
        ggml_backend_buffer_t host = buft_host ? ggml_backend_buft_alloc_buffer(buft_host, max_size) : nullptr;
        ggml_backend_buffer_t pinned = dev_pinned ? ggml_backend_buft_alloc_buffer(dev_pinned(dev), max_size) : nullptr;
        uint8_t * host_ptr   = host   ? (uint8_t *) ggml_backend_buffer_get_base(host)   : nullptr;
        uint8_t * pinned_ptr = pinned ? (uint8_t *) ggml_backend_buffer_get_base(pinned) : nullptr;
        if (host_ptr)   { memset(host_ptr,   1, max_size); }
        if (pinned_ptr) { memset(pinned_ptr, 1, max_size); }

        printf("\n## %s (%s), GB/s, best of %d\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), reps);
        printf("host buffer type: %s, device pinned buffer type: %s\n\n",
            host ? ggml_backend_buft_name(ggml_backend_buffer_get_type(host)) : "none",
            pinned ? ggml_backend_buft_name(ggml_backend_buffer_get_type(pinned)) : "none");
        printf("| copy | unpinned, sync | host buffer, sync | device pinned, sync | device pinned, async | device pinned, background |\n");
        printf("|---|---|---|---|---|---|\n");

        for (size_t size : sizes) {
            const double r_unpinned = bench(reps, size, [&]() { ggml_backend_tensor_set(t, unpinned.data(), 0, size); });
            const double r_host = host_ptr ? bench(reps, size, [&]() { ggml_backend_tensor_set(t, host_ptr, 0, size); }) : 0.0;
            const double r_pinned = pinned_ptr ? bench(reps, size, [&]() { ggml_backend_tensor_set(t, pinned_ptr, 0, size); }) : 0.0;
            const double r_async = pinned_ptr ? bench(reps, size, [&]() {
                ggml_backend_tensor_set_async(backend, t, pinned_ptr, 0, size);
                ggml_backend_synchronize(backend);
            }) : 0.0;
            double r_bg = 0.0;
            if (pinned_ptr && upload_bg && upload_bg(backend, t, pinned_ptr, 0, 0 + 1)) {
                upload_bg_done(backend, upload_bg_submit(backend), true);
                r_bg = bench(reps, size, [&]() {
                    upload_bg(backend, t, pinned_ptr, 0, size);
                    upload_bg_done(backend, upload_bg_submit(backend), true);
                });
            }
            printf("| %3zu MiB | %.2f | %.2f | %.2f | %.2f | %.2f |\n", size/MiB, r_unpinned, r_host, r_pinned, r_async, r_bg);
        }

        // expert sized copies, one batch
        if (pinned_ptr && upload_bg) {
            const size_t n = 16;
            const size_t size = 2*MiB;
            const double r = bench(reps, n*size, [&]() {
                for (size_t j = 0; j < n; ++j) {
                    upload_bg(backend, t, pinned_ptr + j*size, j*size, size);
                }
                upload_bg_done(backend, upload_bg_submit(backend), true);
            });
            printf("| %zu x %zu MiB, one background batch | | | | | %.2f |\n", n, size/MiB, r);
        }

        if (pinned) { ggml_backend_buffer_free(pinned); }
        if (host)   { ggml_backend_buffer_free(host); }
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    return 0;
}
