#include <cuda_runtime.h>
#include <surface_types.h>

namespace app {
namespace {

__device__ __forceinline__ unsigned char toU8(float v) {
    v = fminf(fmaxf(v, 0.f), 1.f);
    return static_cast<unsigned char>(v * 255.f + 0.5f);
}

/// Inria planar 输出与当前窗口预览坐标一致，直接按 y 拷贝（不翻转）。
__global__ void planarRgbToRgbaSurface(cudaSurfaceObject_t surf, const float* planar, int W, int H) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= W || y >= H) return;
    const size_t pix = static_cast<size_t>(y) * static_cast<size_t>(W) + static_cast<size_t>(x);
    const size_t plane = static_cast<size_t>(W) * static_cast<size_t>(H);
    const uchar4 rgba = make_uchar4(toU8(planar[pix]), toU8(planar[plane + pix]), toU8(planar[2 * plane + pix]),
                                    255);
    surf2Dwrite(rgba, surf, x * static_cast<int>(sizeof(uchar4)), y);
}

}  // namespace

bool copyPlanarRgbToCudaArray(cudaArray_t array, const float* d_planar_rgb, int width, int height) {
    if (!array || !d_planar_rgb || width <= 0 || height <= 0) return false;
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = array;
    cudaSurfaceObject_t surf = 0;
    if (cudaCreateSurfaceObject(&surf, &res_desc) != cudaSuccess) return false;
    const dim3 block(16, 16);
    const dim3 grid((static_cast<unsigned>(width) + block.x - 1) / block.x,
                    (static_cast<unsigned>(height) + block.y - 1) / block.y);
    planarRgbToRgbaSurface<<<grid, block>>>(surf, d_planar_rgb, width, height);
    const cudaError_t err = cudaGetLastError();
    cudaDestroySurfaceObject(surf);
    return err == cudaSuccess;
}

}  // namespace app
