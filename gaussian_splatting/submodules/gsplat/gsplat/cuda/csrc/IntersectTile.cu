#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <tuple>

// for CUB_WRAPPER
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Intersect.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

__device__ __forceinline__ float clamp_float(
    const float value,
    const float lower,
    const float upper
) {
    return fminf(fmaxf(value, lower), upper);
}

__device__ __forceinline__ float eval_conic_q(
    const float a,
    const float b,
    const float c,
    const float x,
    const float y
) {
    return a * x * x + 2.f * b * x * y + c * y * y;
}

__device__ __forceinline__ float min_conic_q_over_rect(
    const float a,
    const float b,
    const float c,
    const float dx_min,
    const float dx_max,
    const float dy_min,
    const float dy_max
) {
    if (dx_min <= 0.f && 0.f <= dx_max && dy_min <= 0.f && 0.f <= dy_max) {
        return 0.f;
    }

    const float eps = 1e-20f;
    float best = 1e30f;

    float x = dx_min;
    float y = fabsf(c) > eps ? clamp_float(-b * x / c, dy_min, dy_max)
                             : clamp_float(0.f, dy_min, dy_max);
    best = fminf(best, eval_conic_q(a, b, c, x, y));

    x = dx_max;
    y = fabsf(c) > eps ? clamp_float(-b * x / c, dy_min, dy_max)
                       : clamp_float(0.f, dy_min, dy_max);
    best = fminf(best, eval_conic_q(a, b, c, x, y));

    y = dy_min;
    x = fabsf(a) > eps ? clamp_float(-b * y / a, dx_min, dx_max)
                       : clamp_float(0.f, dx_min, dx_max);
    best = fminf(best, eval_conic_q(a, b, c, x, y));

    y = dy_max;
    x = fabsf(a) > eps ? clamp_float(-b * y / a, dx_min, dx_max)
                       : clamp_float(0.f, dx_min, dx_max);
    best = fminf(best, eval_conic_q(a, b, c, x, y));

    return fmaxf(best, 0.f);
}

template <typename scalar_t>
__device__ __forceinline__ bool ellipse_tile_overlaps(
    const scalar_t *__restrict__ conics,
    const scalar_t *__restrict__ opacities,
    const uint32_t idx,
    const vec2 mean2d,
    const int32_t tile_x,
    const int32_t tile_y,
    const uint32_t tile_size,
    const uint32_t image_width,
    const uint32_t image_height
) {
    const float opacity = static_cast<float>(opacities[idx]);
    if (opacity <= ALPHA_THRESHOLD) {
        return false;
    }

    const float threshold = 2.f * __logf(opacity / ALPHA_THRESHOLD);
    const int32_t tile_size_i = static_cast<int32_t>(tile_size);
    const int32_t tile_px_end =
        min((tile_x + 1) * tile_size_i, static_cast<int32_t>(image_width));
    const int32_t tile_py_end =
        min((tile_y + 1) * tile_size_i, static_cast<int32_t>(image_height));
    const float px_min = static_cast<float>(tile_x * tile_size_i) + 0.5f;
    const float px_max = static_cast<float>(tile_px_end) - 0.5f;
    const float py_min = static_cast<float>(tile_y * tile_size_i) + 0.5f;
    const float py_max = static_cast<float>(tile_py_end) - 0.5f;

    if (px_max < px_min || py_max < py_min) {
        return false;
    }

    const float dx_min = mean2d.x - px_max;
    const float dx_max = mean2d.x - px_min;
    const float dy_min = mean2d.y - py_max;
    const float dy_max = mean2d.y - py_min;
    const float a = static_cast<float>(conics[idx * 3]);
    const float b = static_cast<float>(conics[idx * 3 + 1]);
    const float c = static_cast<float>(conics[idx * 3 + 2]);
    const float min_q =
        min_conic_q_over_rect(a, b, c, dx_min, dx_max, dy_min, dy_max);
    return min_q <= threshold + 1e-4f;
}

template <typename scalar_t>
__global__ void intersect_tile_kernel(
    // if the data is [...,  N, ...] or [nnz, ...] (packed)
    const bool packed,
    // parallelize over I * N, only used if packed is False
    const uint32_t I,
    const uint32_t N,
    // parallelize over nnz, only used if packed is True
    const uint32_t nnz,
    const int64_t *__restrict__ image_ids,    // [nnz] optional
    const int64_t *__restrict__ gaussian_ids, // [nnz] optional
    // data
    const scalar_t *__restrict__ means2d,            // [..., N, 2] or [nnz, 2]
    const int32_t *__restrict__ radii,               // [..., N, 2] or [nnz, 2]
    const scalar_t *__restrict__ depths,             // [..., N] or [nnz]
    const scalar_t *__restrict__ conics,             // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ opacities,          // [..., N] or [nnz]
    const int64_t *__restrict__ cum_tiles_per_gauss, // [..., N] or [nnz]
    const bool ellipse_tile_filter,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_n_bits,
    const uint32_t image_n_bits,
    int32_t *__restrict__ tiles_per_gauss, // [..., N] or [nnz]
    int64_t *__restrict__ isect_ids,       // [n_isects]
    int32_t *__restrict__ flatten_ids      // [n_isects]
) {
    // parallelize over I * N.
    uint32_t idx = cg::this_grid().thread_rank();
    bool first_pass = cum_tiles_per_gauss == nullptr;
    if (idx >= (packed ? nnz : I * N)) {
        return;
    }

    const float radius_x = radii[idx * 2];
    const float radius_y = radii[idx * 2 + 1];
    if (radius_x <= 0 || radius_y <= 0) {
        if (first_pass) {
            tiles_per_gauss[idx] = 0;
        }
        return;
    }

    vec2 mean2d = glm::make_vec2(means2d + 2 * idx);

    float tile_radius_x = radius_x / static_cast<float>(tile_size);
    float tile_radius_y = radius_y / static_cast<float>(tile_size);
    float tile_x = mean2d.x / static_cast<float>(tile_size);
    float tile_y = mean2d.y / static_cast<float>(tile_size);

    // tile_min is inclusive, tile_max is exclusive
    uint2 tile_min, tile_max;
    tile_min.x = min(max(0, (uint32_t)floor(tile_x - tile_radius_x)), tile_width);
    tile_min.y =
        min(max(0, (uint32_t)floor(tile_y - tile_radius_y)), tile_height);
    tile_max.x = min(max(0, (uint32_t)ceil(tile_x + tile_radius_x)), tile_width);
    tile_max.y = min(max(0, (uint32_t)ceil(tile_y + tile_radius_y)), tile_height);

    if (first_pass) {
        // first pass only writes out tiles_per_gauss
        int32_t tile_count = 0;
        if (ellipse_tile_filter) {
            for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
                for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
                    if (ellipse_tile_overlaps(
                            conics,
                            opacities,
                            idx,
                            mean2d,
                            j,
                            i,
                            tile_size,
                            image_width,
                            image_height
                        )) {
                        ++tile_count;
                    }
                }
            }
        } else {
            tile_count = static_cast<int32_t>(
                (tile_max.y - tile_min.y) * (tile_max.x - tile_min.x)
            );
        }
        tiles_per_gauss[idx] = tile_count;
        return;
    }

    int64_t iid; // image id
    if (packed) {
        // parallelize over nnz
        iid = image_ids[idx];
    } else {
        // parallelize over I * N
        iid = idx / N;
    }
    const int64_t iid_enc = iid << (32 + tile_n_bits);

    // tolerance for negative depth
    int32_t depth_i32 = *(int32_t *)&(depths[idx]);  // Bit-level reinterpret
    int64_t depth_id_enc = static_cast<uint32_t>(depth_i32);  // Zero-extend to 64-bit
    // int64_t depth_id_enc = (int64_t) * (int32_t *)&(depths[idx]);
    
    int64_t cur_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    for (int32_t i = tile_min.y; i < tile_max.y; ++i) {
        for (int32_t j = tile_min.x; j < tile_max.x; ++j) {
            if (ellipse_tile_filter &&
                !ellipse_tile_overlaps(
                    conics,
                    opacities,
                    idx,
                    mean2d,
                    j,
                    i,
                    tile_size,
                    image_width,
                    image_height
                )) {
                continue;
            }
            int64_t tile_id = i * tile_width + j;
            // e.g. tile_n_bits = 22:
            // image id (10 bits) | tile id (22 bits) | depth (32 bits)
            isect_ids[cur_idx] = iid_enc | (tile_id << 32) | depth_id_enc;
            // the flatten index in [I * N] or [nnz]
            flatten_ids[cur_idx] = static_cast<int32_t>(idx);
            ++cur_idx;
        }
    }
}

void launch_intersect_tile_kernel(
    // inputs
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
    const bool ellipse_tile_filter,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N, nnz;
    int64_t n_elements;
    if (packed) {
        nnz = means2d.size(0); // total number of gaussians
        n_elements = nnz;
    } else {
        N = means2d.size(-2); // number of gaussians per image
        n_elements = I * N;
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

    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "intersect_tile_kernel",
        [&]() {
            intersect_tile_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    packed,
                    I,
                    N,
                    nnz,
                    image_ids.has_value()
                        ? image_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    gaussian_ids.has_value()
                        ? gaussian_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    means2d.data_ptr<scalar_t>(),
                    radii.data_ptr<int32_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.has_value()
                        ? conics.value().data_ptr<scalar_t>()
                        : nullptr,
                    opacities.has_value()
                        ? opacities.value().data_ptr<scalar_t>()
                        : nullptr,
                    cum_tiles_per_gauss.has_value()
                        ? cum_tiles_per_gauss.value().data_ptr<int64_t>()
                        : nullptr,
                    ellipse_tile_filter,
                    image_width,
                    image_height,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_n_bits,
                    image_n_bits,
                    tiles_per_gauss.has_value()
                        ? tiles_per_gauss.value().data_ptr<int32_t>()
                        : nullptr,
                    isect_ids.has_value()
                        ? isect_ids.value().data_ptr<int64_t>()
                        : nullptr,
                    flatten_ids.has_value()
                        ? flatten_ids.value().data_ptr<int32_t>()
                        : nullptr
                );
        }
    );
}

__global__ void intersect_offset_kernel(
    const uint32_t n_isects,
    const int64_t *__restrict__ isect_ids,
    const uint32_t I,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [I, n_tiles]
) {
    // e.g., ids: [1, 1, 1, 3, 3], n_tiles = 6
    // counts: [0, 3, 0, 2, 0, 0]
    // cumsum: [0, 3, 3, 5, 5, 5]
    // offsets: [0, 0, 3, 3, 5, 5]
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects)
        return;

    uint32_t image_n_bits = (uint32_t)floor(log2f(float(I))) + 1;

    int64_t isect_id_curr = isect_ids[idx] >> 32;
    int64_t iid_curr = isect_id_curr >> (tile_n_bits);
    int64_t tid_curr = isect_id_curr & ((1 << tile_n_bits) - 1);
    int64_t id_curr = iid_curr * n_tiles + tid_curr;

    if (idx == 0) {
        // write out the offsets until the first valid tile (inclusive)
        for (uint32_t i = 0; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
    if (idx == n_isects - 1) {
        // write out the rest of the offsets
        for (uint32_t i = id_curr + 1; i < I * n_tiles; ++i)
            offsets[i] = static_cast<int32_t>(n_isects);
    }

    if (idx > 0) {
        // visit the current and previous isect_id and check if the (bid, cid,
        // tile_id) tuple changes.
        int64_t isect_id_prev = isect_ids[idx - 1] >> 32; // shift out the depth
        if (isect_id_prev == isect_id_curr)
            return;

        // write out the offsets between the previous and current tiles
        int64_t iid_prev = isect_id_prev >> (tile_n_bits);
        int64_t tid_prev = isect_id_prev & ((1 << tile_n_bits) - 1);
        int64_t id_prev = iid_prev * n_tiles + tid_prev;
        for (uint32_t i = id_prev + 1; i < id_curr + 1; ++i)
            offsets[i] = static_cast<int32_t>(idx);
    }
}

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [I, tile_height, tile_width]
) {
    int64_t n_elements = isect_ids.size(0); // total number of intersections
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        offsets.fill_(0);
        return;
    }

    uint32_t n_tiles = tile_width * tile_height;
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    intersect_offset_kernel<<<
        grid,
        threads,
        shmem_size,
        at::cuda::getCurrentCUDAStream()>>>(
        n_elements,
        isect_ids.data_ptr<int64_t>(),
        I,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        0,
        32 + tile_n_bits + image_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
// DoubleBuffer reduce the auxiliary memory usage from O(N+P) to O(P)
void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
) {
    if (n_isects <= 0) {
        return;
    }

    // Create a set of DoubleBuffers to wrap pairs of device pointers
    cub::DoubleBuffer<int64_t> d_keys(
        isect_ids.data_ptr<int64_t>(), isect_ids_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int32_t> d_values(
        flatten_ids.data_ptr<int32_t>(), flatten_ids_sorted.data_ptr<int32_t>()
    );
    // image dimensions are contiguous in the isect_ids, 
    // so we can use DeviceSegmentedRadixSort to only sort the lower 
    // (tile_n_bits + 32) bits
    CUB_WRAPPER(
        cub::DeviceSegmentedRadixSort::SortPairs,
        d_keys,
        d_values,
        n_isects,
        n_segments, // number of segments
        offsets.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>() + 1,
        0,
        32 + tile_n_bits,
        at::cuda::getCurrentCUDAStream()
    );
    switch (d_keys.selector) {
    case 0: // sorted items are stored in isect_ids
        isect_ids_sorted.set_(isect_ids);
        break;
    case 1: // sorted items are stored in isect_ids_sorted
        break;
    }
    switch (d_values.selector) {
    case 0: // sorted items are stored in flatten_ids
        flatten_ids_sorted.set_(flatten_ids);
        break;
    case 1: // sorted items are stored in flatten_ids_sorted
        break;
    }
}

} // namespace gsplat
