#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include "Common.h"    // where all the macros are defined
#include "Intersect.h" // where the launch function is declared
#include "Ops.h"       // a collection of all gsplat operators

namespace gsplat {

namespace {

struct CudaEventTimer {
    cudaEvent_t event;

    CudaEventTimer() {
        C10_CUDA_CHECK(cudaEventCreate(&event));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(event);
    }

    void record() {
        C10_CUDA_CHECK(cudaEventRecord(event, at::cuda::getCurrentCUDAStream()));
    }

    float elapsed_ms(const CudaEventTimer &end) const {
        C10_CUDA_CHECK(cudaEventSynchronize(end.event));
        float elapsed = 0.0f;
        C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed, event, end.event));
        return elapsed;
    }
};

at::Tensor make_intersect_profile_tensor(
    float count_ms,
    float cumsum_ms,
    float emit_ms,
    float sort_ms,
    float total_ms
) {
    at::Tensor timings = at::empty(
        {5}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU)
    );
    float *ptr = timings.data_ptr<float>();
    ptr[0] = count_ms;
    ptr[1] = cumsum_ms;
    ptr[2] = emit_ms;
    ptr[3] = sort_ms;
    ptr[4] = total_ms;
    return timings;
}

void check_ellipse_tile_filter_inputs(
    const at::Tensor means2d,
    const at::Tensor depths,
    const at::optional<at::Tensor> conics,
    const at::optional<at::Tensor> opacities,
    const bool ellipse_tile_filter
) {
    if (!ellipse_tile_filter) {
        return;
    }
    TORCH_CHECK(
        conics.has_value() && opacities.has_value(),
        "ellipse_tile_filter requires conics and opacities."
    );
    CHECK_INPUT(conics.value());
    CHECK_INPUT(opacities.value());
    TORCH_CHECK(
        conics.value().numel() == means2d.numel() / 2 * 3,
        "conics must match means2d with shape [..., 3]."
    );
    TORCH_CHECK(
        opacities.value().numel() == depths.numel(),
        "opacities must match depths."
    );
    TORCH_CHECK(
        conics.value().scalar_type() == means2d.scalar_type(),
        "conics must have the same dtype as means2d."
    );
    TORCH_CHECK(
        opacities.value().scalar_type() == means2d.scalar_type(),
        "opacities must have the same dtype as means2d."
    );
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented,
    const bool ellipse_tile_filter
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    check_ellipse_tile_filter_inputs(
        means2d, depths, conics, opacities, ellipse_tile_filter
    );

    auto opt = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            image_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, image_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(image_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the image id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t image_n_bits = std::bit_width(I);
    uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    assert(image_n_bits + tile_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    at::Tensor offsets;
    if (n_elements) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            ellipse_tile_filter ? conics : c10::nullopt,
            ellipse_tile_filter ? opacities : c10::nullopt,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            ellipse_tile_filter,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
        if (segmented) {
            // offsets in the isect_ids and flatten_ids
            offsets = at::cumsum(
                at::sum(tiles_per_gauss, -1).view({-1}), 0, at::kLong
            );
            offsets = at::cat(
                {at::tensor({0}, opt.dtype(at::kLong)),
                offsets}
            );
        }
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if (n_isects) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            ellipse_tile_filter ? conics : c10::nullopt,
            ellipse_tile_filter ? opacities : c10::nullopt,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            ellipse_tile_filter,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        if (segmented) {
            segmented_radix_sort_double_buffer(
                n_isects,
                I,
                image_n_bits,
                tile_n_bits,
                offsets,
                isect_ids,
                flatten_ids,
                isect_ids_sorted,
                flatten_ids_sorted
            );
        } else {
            radix_sort_double_buffer(
                n_isects,
                image_n_bits,
                tile_n_bits,
                isect_ids,
                flatten_ids,
                isect_ids_sorted, 
                flatten_ids_sorted
            );
        }
        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> intersect_tile_profiled(
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented,
    const bool ellipse_tile_filter
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    check_ellipse_tile_filter_inputs(
        means2d, depths, conics, opacities, ellipse_tile_filter
    );

    auto opt = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            image_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, image_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(image_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t image_n_bits = (uint32_t)floor(log2(I)) + 1;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    assert(image_n_bits + tile_n_bits <= 32);

    CudaEventTimer total_start, total_stop;
    CudaEventTimer count_start, count_stop;
    CudaEventTimer cumsum_start, cumsum_stop;
    CudaEventTimer emit_start, emit_stop;
    CudaEventTimer sort_start, sort_stop;

    float count_ms = 0.0f;
    float cumsum_ms = 0.0f;
    float emit_ms = 0.0f;
    float sort_ms = 0.0f;
    float total_ms = 0.0f;
    float segmented_cumsum_ms = 0.0f;

    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    at::Tensor offsets;

    total_start.record();
    if (n_elements) {
        count_start.record();
        launch_intersect_tile_kernel(
            means2d,
            radii,
            depths,
            ellipse_tile_filter ? conics : c10::nullopt,
            ellipse_tile_filter ? opacities : c10::nullopt,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            ellipse_tile_filter,
            c10::nullopt,
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt,
            c10::nullopt
        );
        count_stop.record();

        cumsum_start.record();
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        cumsum_stop.record();
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
        if (segmented) {
            CudaEventTimer offsets_start, offsets_stop;
            offsets_start.record();
            offsets = at::cumsum(
                at::sum(tiles_per_gauss, -1).view({-1}), 0, at::kLong
            );
            offsets = at::cat(
                {at::tensor({0}, opt.dtype(at::kLong)),
                offsets}
            );
            offsets_stop.record();
            segmented_cumsum_ms = offsets_start.elapsed_ms(offsets_stop);
        }
    } else {
        n_isects = 0;
    }

    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if (n_isects) {
        emit_start.record();
        launch_intersect_tile_kernel(
            means2d,
            radii,
            depths,
            ellipse_tile_filter ? conics : c10::nullopt,
            ellipse_tile_filter ? opacities : c10::nullopt,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            ellipse_tile_filter,
            cum_tiles_per_gauss,
            c10::nullopt,
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );
        emit_stop.record();
    }

    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        sort_start.record();
        if (segmented) {
            segmented_radix_sort_double_buffer(
                n_isects,
                I,
                image_n_bits,
                tile_n_bits,
                offsets,
                isect_ids,
                flatten_ids,
                isect_ids_sorted,
                flatten_ids_sorted
            );
        } else {
            radix_sort_double_buffer(
                n_isects,
                image_n_bits,
                tile_n_bits,
                isect_ids,
                flatten_ids,
                isect_ids_sorted,
                flatten_ids_sorted
            );
        }
        sort_stop.record();
        total_stop.record();

        count_ms = count_start.elapsed_ms(count_stop);
        cumsum_ms = cumsum_start.elapsed_ms(cumsum_stop) + segmented_cumsum_ms;
        emit_ms = emit_start.elapsed_ms(emit_stop);
        sort_ms = sort_start.elapsed_ms(sort_stop);
        total_ms = total_start.elapsed_ms(total_stop);
        return std::make_tuple(
            tiles_per_gauss,
            isect_ids_sorted,
            flatten_ids_sorted,
            make_intersect_profile_tensor(
                count_ms, cumsum_ms, emit_ms, sort_ms, total_ms
            )
        );
    }

    total_stop.record();
    if (n_elements) {
        count_ms = count_start.elapsed_ms(count_stop);
        cumsum_ms = cumsum_start.elapsed_ms(cumsum_stop) + segmented_cumsum_ms;
    }
    if (n_isects) {
        emit_ms = emit_start.elapsed_ms(emit_stop);
    }
    total_ms = total_start.elapsed_ms(total_stop);
    return std::make_tuple(
        tiles_per_gauss,
        isect_ids,
        flatten_ids,
        make_intersect_profile_tensor(count_ms, cumsum_ms, emit_ms, sort_ms, total_ms)
    );
}

at::Tensor intersect_offset(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    auto opt = isect_ids.options();
    at::Tensor offsets = at::empty(
        {I, tile_height, tile_width}, opt.dtype(at::kInt)
    );
    launch_intersect_offset_kernel(
        isect_ids, I, tile_width, tile_height, offsets
    );
    return offsets;
}

} // namespace gsplat
