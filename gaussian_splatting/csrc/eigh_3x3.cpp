#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/GradMode.h>
#include <torch/csrc/utils/pybind.h>
#include <cusolverDn.h>
#include <limits>
#include <tuple>

#if CUDA_VERSION < 13020
#error "Boba requires the CUDA 13.2 or newer toolkit from phystwin-cu132"
#endif

namespace {
void check(cusolverStatus_t status) {
  TORCH_CHECK(status == CUSOLVER_STATUS_SUCCESS,
              "Boba cuSOLVER syevjBatched failed (status ", int(status), ")");
}

struct JacobiParams {
  syevjInfo_t value = nullptr;
  JacobiParams() { check(cusolverDnCreateSyevjInfo(&value)); }
  ~JacobiParams() { if (value) cusolverDnDestroySyevjInfo(value); }
  JacobiParams(const JacobiParams&) = delete;
  JacobiParams& operator=(const JacobiParams&) = delete;
};

std::tuple<at::Tensor, at::Tensor> eigh_3x3(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda() && input.dim() == 3 && input.size(1) == 3 &&
              input.size(2) == 3, "Expected CUDA matrices with shape (N, 3, 3)");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble,
              "syevjBatched supports float32 and float64");
  TORCH_CHECK(!c10::GradMode::is_enabled() || !input.requires_grad(),
              "Use the Python eigh_3x3 wrapper for autograd");
  TORCH_CHECK(input.size(0) <= std::numeric_limits<int>::max(),
              "syevjBatched batch size exceeds its integer API limit");
  const c10::cuda::CUDAGuard guard(input.device());
  const int batch = static_cast<int>(input.size(0));
  auto values = at::empty({batch, 3}, input.options());
  // cuSOLVER overwrites A with eigenvectors. Keep the caller's Gram matrix
  // intact using the same column-major output layout as torch.linalg.eigh.
  auto vectors = at::empty_strided({batch, 3, 3}, {9, 1, 3}, input.options());
  if (!batch) return {values, vectors};
  vectors.copy_(input);
  auto info = at::empty({batch}, input.options().dtype(at::kInt));
  // PyTorch owns the handle and binds it to this thread/device/current stream.
  auto handle = at::cuda::getCurrentCUDASolverDnHandle();
  JacobiParams params;
  check(cusolverDnXsyevjSetSortEig(params.value, 1));
  check(cusolverDnXsyevjSetMaxSweeps(params.value, 100));
  // Leave tolerance at cuSOLVER's default (machine precision).
  int lwork = 0;
  if (input.scalar_type() == at::kFloat) {
    auto* a = vectors.data_ptr<float>();
    auto* w = values.data_ptr<float>();
    check(cusolverDnSsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, 3, a, 3, w, &lwork, params.value, batch));
    TORCH_CHECK(lwork >= 0, "cuSOLVER returned an invalid workspace size");
    auto work = at::empty({lwork}, input.options());
    check(cusolverDnSsyevjBatched(handle, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, 3, a, 3, w, work.data_ptr<float>(), lwork,
        info.data_ptr<int>(), params.value, batch));
  } else {
    auto* a = vectors.data_ptr<double>();
    auto* w = values.data_ptr<double>();
    check(cusolverDnDsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, 3, a, 3, w, &lwork, params.value, batch));
    TORCH_CHECK(lwork >= 0, "cuSOLVER returned an invalid workspace size");
    auto work = at::empty({lwork}, input.options());
    check(cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, 3, a, 3, w, work.data_ptr<double>(), lwork,
        info.data_ptr<int>(), params.value, batch));
  }
  // Preserve PyTorch's convergence diagnostics and batch indices. Like eigh,
  // this synchronizes with the CPU; no unchecked asynchronous errors escape.
  at::_linalg_check_errors(info, "linalg.eigh", false);
  return {values, vectors};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("eigh_3x3", &eigh_3x3, pybind11::call_guard<pybind11::gil_scoped_release>());
}
