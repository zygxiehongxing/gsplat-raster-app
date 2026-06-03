#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <cstddef>
#include <string>
#include <vector>

namespace app {

/// App 侧 PLY 读取参数（不依赖 SDK 的 PLY 接口）。
struct PlyLoadOptions {
    size_t max_points = 0;
    float min_opacity = 0.005f;
    bool drop_sh_rest = false;
    bool stride_subsample = true;
};

/// 从 3DGS 风格 PLY 读取高斯数组。
bool loadGaussianPlyLocal(const std::string& path, std::vector<gsplat::Gaussian>& out,
                          const PlyLoadOptions& opts);

}  // namespace app
