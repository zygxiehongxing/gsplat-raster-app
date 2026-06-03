#pragma once

#include <gsplat_raster/gsplat_raster.h>

#include <string>

/// 真值包：记录单帧 OSG 输入矩阵与实际送入 SDK 的相机参数。
struct CameraTruth {
    int width = 0;
    int height = 0;
    int mat_mode = -1;
    bool proj_mul_pv = false;
    float scale_modifier = 0.02f;
    bool dc_only = true;
    bool drop_sh_rest = true;
    float min_opacity = 0.002f;
    int visible = 0;
    double osg_view[16] = {};
    double osg_proj[16] = {};
    gsplat::Camera sdk{};
};

/// 将真值包写入 JSON 文件。
bool saveCameraTruth(const std::string& path, const CameraTruth& truth);
/// 从 JSON 文件读取真值包。
bool loadCameraTruth(const std::string& path, CameraTruth& truth);

/// 将真值包里的渲染相关字段应用到 raster 设置。
void applyTruthRenderSettings(gsplat::Rasterizer& raster, const CameraTruth& truth);

/// 打印真值包摘要（分辨率、模式、可见数量等）。
void printCameraTruthSummary(const CameraTruth& truth, const char* label);
/// 计算并打印两组 4x4 行主序矩阵的最大绝对差。
void logMatrixDiff(const char* name_a, const double a[16], const char* name_b, const double b[16]);
