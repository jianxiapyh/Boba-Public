#pragma once

#include "Common.h"

namespace gsplat {

template <typename scalar_t>
inline __device__ void sh_coeffs_to_color_fast(
    const uint32_t degree,
    const uint32_t c,
    const vec3 &dir,
    const scalar_t *coeffs,
    scalar_t *colors
) {
    float result = 0.2820947917738781f * coeffs[c];
    if (degree >= 1) {
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x = dir.x * inorm;
        float y = dir.y * inorm;
        float z = dir.z * inorm;

        result +=
            0.48860251190292f * (-y * coeffs[1 * 3 + c] +
                                 z * coeffs[2 * 3 + c] -
                                 x * coeffs[3 * 3 + c]);
        if (degree >= 2) {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1 = x * x - y * y;
            float fS1 = 2.f * x * y;
            float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7 = fTmp0B * x;
            float pSH5 = fTmp0B * y;
            float pSH8 = 0.5462742152960395f * fC1;
            float pSH4 = 0.5462742152960395f * fS1;

            result += pSH4 * coeffs[4 * 3 + c] +
                      pSH5 * coeffs[5 * 3 + c] +
                      pSH6 * coeffs[6 * 3 + c] +
                      pSH7 * coeffs[7 * 3 + c] +
                      pSH8 * coeffs[8 * 3 + c];
            if (degree >= 3) {
                float fTmp0C =
                    -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2 = x * fC1 - y * fS1;
                float fS2 = x * fS1 + y * fC1;
                float pSH12 =
                    z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9 = -0.5900435899266435f * fS2;

                result += pSH9 * coeffs[9 * 3 + c] +
                          pSH10 * coeffs[10 * 3 + c] +
                          pSH11 * coeffs[11 * 3 + c] +
                          pSH12 * coeffs[12 * 3 + c] +
                          pSH13 * coeffs[13 * 3 + c] +
                          pSH14 * coeffs[14 * 3 + c] +
                          pSH15 * coeffs[15 * 3 + c];

                if (degree >= 4) {
                    float fTmp0D =
                        z * (-4.683325804901025f * z2 +
                             2.007139630671868f);
                    float fTmp1C =
                        3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 =
                        (1.984313483298443f * z * pSH12 -
                         1.006230589874905f * pSH6);
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    result += pSH16 * coeffs[16 * 3 + c] +
                              pSH17 * coeffs[17 * 3 + c] +
                              pSH18 * coeffs[18 * 3 + c] +
                              pSH19 * coeffs[19 * 3 + c] +
                              pSH20 * coeffs[20 * 3 + c] +
                              pSH21 * coeffs[21 * 3 + c] +
                              pSH22 * coeffs[22 * 3 + c] +
                              pSH23 * coeffs[23 * 3 + c] +
                              pSH24 * coeffs[24 * 3 + c];
                }
            }
        }
    }

    colors[c] = result;
}

} // namespace gsplat
