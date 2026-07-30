// clang-format off
// Decode-only Q8_0 QMV plus DeepSeek-V4 hC-post kernel instantiations.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/kq_quantized.h"
#include "mlx/backend/metal/kernels/kq_ds4_hc_post.h"

instantiate_kernel(
    "kquant_q8_0_qmv_fast_hc_post_float",
    kq_q8_0_qmv_fast_hc_post,
    float,
    true)
instantiate_kernel(
    "kquant_q8_0_qmv_fast_hc_post_bfloat16_t",
    kq_q8_0_qmv_fast_hc_post,
    bfloat16_t,
    false)
instantiate_kernel(
    "kquant_q8_0_qmv_fast_add_hc_post_bfloat16_t",
    kq_q8_0_qmv_fast_add_hc_post,
    bfloat16_t)
    // clang-format on
