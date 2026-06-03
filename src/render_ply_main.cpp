#include <gsplat_raster/gsplat_raster.h>

#include "camera_truth.h"
#include "ply_loader_local.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultMaxPoints = 120000;

/// 估算高斯点云包围球中心与半径。
void boundsCenterRadius(const std::vector<gsplat::Gaussian>& g, double c[3], double& radius) {
    if (g.empty()) {
        c[0] = c[1] = c[2] = 0;
        radius = 1;
        return;
    }
    const size_t step = std::max(size_t(1), g.size() / 10000);
    double minx = g[0].x, maxx = g[0].x;
    double miny = g[0].y, maxy = g[0].y;
    double minz = g[0].z, maxz = g[0].z;
    for (size_t i = 0; i < g.size(); i += step) {
        const auto& p = g[i];
        minx = std::min(minx, static_cast<double>(p.x));
        maxx = std::max(maxx, static_cast<double>(p.x));
        miny = std::min(miny, static_cast<double>(p.y));
        maxy = std::max(maxy, static_cast<double>(p.y));
        minz = std::min(minz, static_cast<double>(p.z));
        maxz = std::max(maxz, static_cast<double>(p.z));
    }
    c[0] = 0.5 * (minx + maxx);
    c[1] = 0.5 * (miny + maxy);
    c[2] = 0.5 * (minz + maxz);
    const double dx = maxx - c[0], dy = maxy - c[1], dz = maxz - c[2];
    radius = std::sqrt(dx * dx + dy * dy + dz * dz);
    radius = std::max(radius, 0.5);
}

/// 读取整个文本文件到字符串。
std::string readAllText(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// 从字符串中提取所有数字并转为 float 列表。
bool parseNumberList(const std::string& s, std::vector<float>& out) {
    out.clear();
    const char* p = s.c_str();
    const char* end = p + s.size();
    while (p < end) {
        while (p < end &&
               !std::isdigit(static_cast<unsigned char>(*p)) &&
               *p != '-' && *p != '+' && *p != '.') {
            ++p;
        }
        if (p >= end) break;
        char* next = nullptr;
        const float v = std::strtof(p, &next);
        if (next == p) {
            ++p;
            continue;
        }
        out.push_back(v);
        p = next;
    }
    return !out.empty();
}

/// 从 JSON 中读取数组字段。
bool extractKeyArray(const std::string& json, const std::string& key, std::vector<float>& out) {
    const std::string token = "\"" + key + "\"";
    const size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return false;
    const size_t lb = json.find('[', key_pos);
    if (lb == std::string::npos) return false;
    int depth = 0;
    for (size_t i = lb; i < json.size(); ++i) {
        if (json[i] == '[') ++depth;
        else if (json[i] == ']') {
            --depth;
            if (depth == 0) {
                return parseNumberList(json.substr(lb, i - lb + 1), out);
            }
        }
    }
    return false;
}

/// 从 JSON 中读取数值字段。
bool extractKeyNumber(const std::string& json, const std::string& key, float& out) {
    const std::string token = "\"" + key + "\"";
    const size_t key_pos = json.find(token);
    if (key_pos == std::string::npos) return false;
    const size_t colon = json.find(':', key_pos);
    if (colon == std::string::npos) return false;
    const char* p = json.c_str() + colon + 1;
    char* next = nullptr;
    out = std::strtof(p, &next);
    return next != p;
}

/// 加载 camera.json 到 SDK Camera 结构。
bool loadCameraJson(const std::string& path, gsplat::Camera& cam) {
    const std::string json = readAllText(path);
    if (json.empty()) return false;

    std::vector<float> view_v;
    std::vector<float> proj_v;
    std::vector<float> cam_pos_v;
    float tan_fovx = 0.f;
    float tan_fovy = 0.f;

    if (!extractKeyArray(json, "view", view_v) || view_v.size() < 16) return false;
    if (!extractKeyArray(json, "proj", proj_v) || proj_v.size() < 16) return false;
    if (!extractKeyArray(json, "cam_pos", cam_pos_v) || cam_pos_v.size() < 3) return false;
    if (!extractKeyNumber(json, "tan_fovx", tan_fovx)) return false;
    if (!extractKeyNumber(json, "tan_fovy", tan_fovy)) return false;

    for (int i = 0; i < 16; ++i) {
        cam.view[i] = view_v[static_cast<size_t>(i)];
        cam.proj[i] = proj_v[static_cast<size_t>(i)];
    }
    cam.cam_pos[0] = cam_pos_v[0];
    cam.cam_pos[1] = cam_pos_v[1];
    cam.cam_pos[2] = cam_pos_v[2];
    cam.tan_fovx = tan_fovx;
    cam.tan_fovy = tan_fovy;
    return cam.tan_fovx > 0.f && cam.tan_fovy > 0.f;
}

/// 判断字符串是否为纯无符号整数。
bool isUnsignedInt(const std::string& s) {
    if (s.empty()) return false;
    for (char ch : s) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) return false;
    }
    return true;
}

}  // namespace

/// render_ply CLI 主入口：加载高斯、构建相机并调用 SDK 输出 PNG。
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.ply> <output.png> [width] [height] [max_points]"
                  << " [--camera camera.json | --truth camera_truth.json]\n"
                  << "  max_points defaults to " << kDefaultMaxPoints
                  << " (stride subsample). Use 0 for all points (large VRAM).\n"
                  << "  --truth loads OSG+SDK bundle written by gsplat_osg_app (R/T keys).\n"
                  << "  --camera requires keys: view[16], proj[16], cam_pos[3], tan_fovx, tan_fovy.\n";
        return 1;
    }

    const std::string ply = argv[1];
    const std::string png = argv[2];
    int width = 1280;
    int height = 720;
    size_t max_points = kDefaultMaxPoints;
    std::string camera_json_path;
    std::string truth_json_path;

    int argi = 3;
    if (argi < argc && isUnsignedInt(argv[argi])) width = std::stoi(argv[argi++]);
    if (argi < argc && isUnsignedInt(argv[argi])) height = std::stoi(argv[argi++]);
    if (argi < argc && isUnsignedInt(argv[argi])) max_points = static_cast<size_t>(std::stoull(argv[argi++]));
    while (argi < argc) {
        const std::string a = argv[argi++];
        if (a == "--camera" && argi < argc) {
            camera_json_path = argv[argi++];
        } else if (a == "--truth" && argi < argc) {
            truth_json_path = argv[argi++];
        }
    }

    app::PlyLoadOptions load_opts;
    load_opts.max_points = max_points;
    load_opts.stride_subsample = true;
    load_opts.drop_sh_rest = true;
    if (!truth_json_path.empty()) {
        load_opts.stride_subsample = false;
        load_opts.min_opacity = 0.002f;
    }

    std::vector<gsplat::Gaussian> gaussians;
    std::cout << "Loading " << ply << " (max " << load_opts.max_points << " points) ...\n"
              << std::flush;
    if (!app::loadGaussianPlyLocal(ply, gaussians, load_opts)) {
        std::cerr << "PLY load failed\n";
        return 3;
    }
    std::cout << "Gaussians: " << gaussians.size() << "\n";
    if (!gsplat::isCudaAvailable()) {
        std::cerr << "CUDA not available. Build with CUDA toolkit and diff-gaussian-rasterization.\n";
        return 2;
    }

    double center[3], radius;
    boundsCenterRadius(gaussians, center, radius);
    std::cout << "Scene center (" << center[0] << ", " << center[1] << ", " << center[2]
              << ") radius ~" << radius << "\n";

    const double dist = radius * 8.0;
    const double eye[3] = {center[0], center[1] - dist, center[2] + dist * 0.35};
    const double up[3] = {0, 0, 1};
    const double aspect = static_cast<double>(width) / static_cast<double>(height);

    gsplat::Rasterizer raster;
    gsplat::RenderSettings settings;
    settings.scale_modifier = 0.12f;
    settings.dc_only = true;
    CameraTruth truth;
    if (!truth_json_path.empty()) {
        if (!loadCameraTruth(truth_json_path, truth)) {
            std::cerr << "Failed to parse truth json: " << truth_json_path << "\n";
            return 6;
        }
        width = truth.width > 0 ? truth.width : width;
        height = truth.height > 0 ? truth.height : height;
        settings.scale_modifier = truth.scale_modifier;
        settings.dc_only = truth.dc_only;
        load_opts.drop_sh_rest = truth.drop_sh_rest;
        load_opts.min_opacity = truth.min_opacity;
        load_opts.stride_subsample = false;
        printCameraTruthSummary(truth, truth_json_path.c_str());
    }
    raster.setSettings(settings);
    if (raster.setGaussians(std::move(gaussians)) != gsplat::Status::Ok) {
        std::cerr << "Upload failed\n";
        return 4;
    }

    gsplat::Camera cam;
    if (!truth_json_path.empty()) {
        cam = truth.sdk;
        std::cout << "Using truth camera (exact SDK matrices) from: " << truth_json_path << "\n";
        std::cout << "  recorded visible=" << truth.visible << " mat_mode=" << truth.mat_mode << "\n";
    } else if (!camera_json_path.empty()) {
        if (!loadCameraJson(camera_json_path, cam)) {
            std::cerr << "Failed to parse camera json: " << camera_json_path << "\n";
            return 6;
        }
        std::cout << "Using camera from: " << camera_json_path << "\n";
    } else {
        gsplat::buildCameraLookAt(eye, center, up, 60.0, aspect, 0.01, 10000.0, center, cam, &raster);
    }

    const int frustum = raster.countFrustumPass(cam);
    std::cout << "Frustum pass (sampled): ~" << frustum << "\n";

    int render_w = width;
    int render_h = height;
    const int n = raster.numGaussians();
    if (n > 100000) {
        const int cap_w = 960;
        if (render_w > cap_w) {
            render_h = std::max(1, render_h * cap_w / render_w);
            render_w = cap_w;
            std::cout << "Lowering resolution to " << render_w << "x" << render_h
                      << " for GPU memory (" << n << " splats)\n";
        }
    }

    std::cout << "Rendering " << render_w << "x" << render_h << " -> " << png << " ...\n" << std::flush;
    const gsplat::Status st = raster.renderToPng(cam, render_w, render_h, png);
    if (st != gsplat::Status::Ok) {
        std::cerr << "Render failed: " << gsplat::statusString(st) << "\n";
        gsplat::resetCudaDevice();
        return 5;
    }
    const int vis = raster.lastVisibleCount();
    std::cout << "Visible splats: " << vis << " / " << raster.numGaussians() << "\n";
    if (!truth_json_path.empty() && truth.visible > 0) {
        const int delta = vis - truth.visible;
        std::cout << "Truth visible delta (cli - recorded): " << delta << "\n";
    }
    std::cout << "Wrote " << png << "\n";
    gsplat::resetCudaDevice();
    return 0;
}
