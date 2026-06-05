#include <gsplat_raster/gsplat_raster.h>

#include "camera_truth.h"
#include "gl_screen_cache.h"
#include "ply_loader_local.h"

#include <osg/BlendFunc>

#include <osg/GL>

#include <osg/Camera>

#include <osg/DisplaySettings>

#include <osg/Drawable>

#include <osg/Geode>

#include <osg/Geometry>

#include <osg/Group>

#include <osg/Image>

#include <osg/PrimitiveSet>

#include <osg/Point>

#include <osg/State>

#include <osg/Texture2D>

#include <osg/Timer>

#include <osgGA/GUIEventHandler>

#include <osgGA/TrackballManipulator>

#include <osgViewer/Viewer>


#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_runtime.h>
#endif



#include <algorithm>

#include <cctype>

#include <cmath>

#include <cstring>

#include <iostream>

#include <limits>

#include <string>

#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <osgViewer/api/Win32/GraphicsWindowWin32>
#include <imm.h>

/// 3D 视图无文本输入：解除 Win32 IME 关联，避免中文输入法嵌套消息循环导致 OSG 卡死。
void disableWindowsImeOnViewer(osgViewer::Viewer& viewer) {
    osgViewer::GraphicsWindow* gw =
        dynamic_cast<osgViewer::GraphicsWindow*>(viewer.getCamera()->getGraphicsContext());
    auto* gw32 = dynamic_cast<osgViewer::GraphicsWindowWin32*>(gw);
    if (!gw32) {
        return;
    }
    const HWND hwnd = gw32->getHWND();
    if (!hwnd) {
        return;
    }
    ImmAssociateContext(hwnd, nullptr);
    std::cout << "[OSG APP] Win32 IME disabled on GL window (avoids CJK IME freeze)\n";
}
#endif

namespace {



constexpr float kShC0 = 0.28209479177387814f;

constexpr double kMinRenderIntervalSec = 0.12;



/// 将颜色值裁剪到 [0,1]。
float clamp01(float v) {

    if (v < 0.f) return 0.f;

    if (v > 1.f) return 1.f;

    return v;

}



/// 把 osg::Matrixd 按行主序拷贝到 double[16]。
void osgMatrixToArray(const osg::Matrixd& m, double out[16]) {

    for (int r = 0; r < 4; ++r) {

        for (int c = 0; c < 4; ++c) {

            out[r * 4 + c] = m(r, c);

        }

    }

}

/// 比较两组 4x4 行主序矩阵是否完全相等。
bool matricesEqual(const double a[16], const double b[16]) {

    return std::memcmp(a, b, 16 * sizeof(double)) == 0;

}

/// 将场景中心转换为 C 风格数组，供 SDK 接口使用。
void sceneRefCenter(const osg::Vec3d& scene_center, double ref_center[3]) {
    ref_center[0] = scene_center.x();
    ref_center[1] = scene_center.y();
    ref_center[2] = scene_center.z();
}

/// 从 view 矩阵反解 eye/center/up（用于 look-at 相机构建兜底）。
bool extractLookAtFromView(const osg::Matrixd& view, osg::Vec3d& eye, osg::Vec3d& center, osg::Vec3d& up) {
    osg::Matrixd inv;
    if (!inv.invert(view)) return false;
    eye.set(inv(3, 0), inv(3, 1), inv(3, 2));

    osg::Vec3d forward(-view(2, 0), -view(2, 1), -view(2, 2));
    if (forward.length2() < 1e-12) return false;
    forward.normalize();
    center = eye + forward;

    up.set(view(1, 0), view(1, 1), view(1, 2));
    if (up.length2() < 1e-12) up.set(0.0, 0.0, 1.0);
    up.normalize();
    return true;
}

/// 从 PLY 子集采样世界坐标，供 OSG→CUDA 矩阵模式探测。
void fillCameraProbes(const std::vector<gsplat::Gaussian>* cloud, std::vector<float>& out_xyz) {
    out_xyz.clear();
    if (!cloud || cloud->empty()) return;
    const int n = static_cast<int>(cloud->size());
    const int step = std::max(1, n / 2048);
    out_xyz.reserve(static_cast<size_t>((n + step - 1) / step) * 3u);
    for (int i = 0; i < n; i += step) {
        const auto& g = cloud->at(static_cast<size_t>(i));
        out_xyz.push_back(g.x);
        out_xyz.push_back(g.y);
        out_xyz.push_back(g.z);
    }
}

/// 使用当前帧 OSG view/proj 构建 SDK 相机（与 GL SSBO pass 同一套矩阵）。
void buildCameraFromSnapshot(const osg::Matrixd& view, const osg::Matrixd& proj,
                             const osg::Vec3d& scene_center, gsplat::Camera& out, int* mat_mode_out = nullptr,
                             const std::vector<gsplat::Gaussian>* probe_cloud = nullptr) {
    double view_a[16];
    double proj_a[16];
    osgMatrixToArray(view, view_a);
    osgMatrixToArray(proj, proj_a);
    double ref_center[3];
    sceneRefCenter(scene_center, ref_center);
    static thread_local std::vector<float> probe_xyz;
    fillCameraProbes(probe_cloud, probe_xyz);
    gsplat::buildCameraFromOsg(view_a, proj_a, ref_center, out, mat_mode_out,
                               probe_xyz.empty() ? nullptr : probe_xyz.data(),
                               static_cast<int>(probe_xyz.size() / 3));
}

/// 步骤 1：对比 SSBO 光栅相机 vs 同帧 OSG snapshot（前几次自动打印，或 GSPLAT_AUDIT_CAMERA=1）。
void auditRenderCamera(const gsplat::Camera& render_cam, bool used_ssbo_meta, const osg::Matrixd& view,
                       const osg::Matrixd& proj, const osg::Vec3d& scene_center,
                       const std::vector<gsplat::Gaussian>* probe_cloud) {
    static int audit_left = 3;
    static int env_cached = -1;
    if (env_cached < 0) {
        const char* env = std::getenv("GSPLAT_AUDIT_CAMERA");
        env_cached = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    if (audit_left <= 0 && env_cached == 0) {
        return;
    }
    if (audit_left > 0) {
        --audit_left;
    }

    gsplat::Camera snap{};
    buildCameraFromSnapshot(view, proj, scene_center, snap, nullptr, probe_cloud);

    float max_view = 0.f;
    float max_proj = 0.f;
    int max_view_i = 0;
    int max_proj_i = 0;
    for (int i = 0; i < 16; ++i) {
        const float dv = std::abs(render_cam.view[i] - snap.view[i]);
        const float dp = std::abs(render_cam.proj[i] - snap.proj[i]);
        if (dv > max_view) {
            max_view = dv;
            max_view_i = i;
        }
        if (dp > max_proj) {
            max_proj = dp;
            max_proj_i = i;
        }
    }
    const float d_tan_x = std::abs(render_cam.tan_fovx - snap.tan_fovx);
    const float d_tan_y = std::abs(render_cam.tan_fovy - snap.tan_fovy);
    const float d_pos = std::max({std::abs(render_cam.cam_pos[0] - snap.cam_pos[0]),
                                  std::abs(render_cam.cam_pos[1] - snap.cam_pos[1]),
                                  std::abs(render_cam.cam_pos[2] - snap.cam_pos[2])});

    std::cout << "[CAMERA-AUDIT] ssbo_meta=" << (used_ssbo_meta ? 1 : 0)
              << " |render-snap_view| max=" << max_view << " at i=" << max_view_i
              << " |render-snap_proj| max=" << max_proj << " at i=" << max_proj_i
              << " |tan_fovx|=" << d_tan_x << " |tan_fovy|=" << d_tan_y << " |cam_pos|=" << d_pos
              << " render_tan=(" << render_cam.tan_fovx << "," << render_cam.tan_fovy << ")"
              << " snap_tan=(" << snap.tan_fovx << "," << snap.tan_fovy << ")\n";
}

/// 优先使用 SSBO 头里记录的 GL 矩阵（与视锥收集 pass 逐位一致），否则回退 OSG 快照。
bool buildRenderCamera(app::GlScreenGaussianCache& cache, const osg::Matrixd& view, const osg::Matrixd& proj,
                       const osg::Vec3d& scene_center, gsplat::Camera& out, int* mat_mode_out,
                       const std::vector<gsplat::Gaussian>* probe_cloud) {
    if (cache.hasSsboCamera()) {
        out = cache.ssboCamera();
        if (mat_mode_out) {
            *mat_mode_out = 17;
        }
        return true;
    }
    buildCameraFromSnapshot(view, proj, scene_center, out, mat_mode_out, probe_cloud);
    return false;
}

/// GL/SSBO/SDK：主相机 view/proj。
bool getDrawMatrices(osg::RenderInfo& renderInfo, osg::Camera* cam, osg::Matrixd& V, osg::Matrixd& P,
                     int& vp_w, int& vp_h) {
    vp_w = 1280;
    vp_h = 720;
    if (!cam) return false;

    V = cam->getViewMatrix();
    P = cam->getProjectionMatrix();

    osg::State* osg_state = renderInfo.getState();
    if (osg_state) {
        const osg::Viewport* vp = osg_state->getCurrentViewport();
        if (vp && vp->width() > 0 && vp->height() > 0) {
            vp_w = std::max(1, static_cast<int>(vp->width()));
            vp_h = std::max(1, static_cast<int>(vp->height()));
            return true;
        }
    }
    const osg::Viewport* vp = cam->getViewport();
    if (vp && vp->width() > 0 && vp->height() > 0) {
        vp_w = std::max(1, static_cast<int>(vp->width()));
        vp_h = std::max(1, static_cast<int>(vp->height()));
    }
    return true;
}

struct PreviewState;

/// 执行一次真值导出/截图流程（truth json + capture + replay）。
void runTruthCapture(PreviewState* state, osg::State* gl_state, const osg::Matrixd& V, const osg::Matrixd& P,
                     int vp_w, int vp_h, bool do_png, const gsplat::Camera* frozen_cam = nullptr,
                     int frozen_w = 0, int frozen_h = 0, bool reuse_screen_buffers = false);

/// 截图/真值渲染：scale 过大时自动减半重试，避免 binning OOM 导致无 PNG。
gsplat::Status renderToPngWithScaleFallback(gsplat::Rasterizer* raster, const gsplat::Camera& cam, int w,
                                            int h, const gsplat::DeviceGaussianBuffers& buf,
                                            const std::string& path, float start_scale, float& out_used_scale) {
    if (!raster) return gsplat::Status::ErrorRenderFailed;
    float try_scale = std::clamp(start_scale, 0.005f, 2.0f);
    for (int attempt = 0; attempt < 6; ++attempt) {
        gsplat::RenderSettings rs = raster->settings();
        rs.scale_modifier = try_scale;
        raster->setSettings(rs);
        const gsplat::Status st = raster->renderToPng(cam, w, h, buf, path);
        if (st == gsplat::Status::Ok) {
            out_used_scale = try_scale;
            if (attempt > 0) {
                std::cout << "[OSG APP] capture ok at scale_modifier=" << try_scale << " (reduced from "
                          << start_scale << ")\n";
            }
            return st;
        }
        if (try_scale <= 0.03f + 1e-5f) {
            break;
        }
        try_scale = std::max(0.03f, try_scale * 0.5f);
        std::cout << "[OSG APP] capture retry scale_modifier=" << try_scale << "\n";
    }
    out_used_scale = try_scale;
    return gsplat::Status::ErrorRenderFailed;
}

/// 将交互调节后的 scale_modifier 应用到 raster 设置。
void updateScaleForSnapshot(const osg::Matrixd& view, const osg::Vec3d& scene_center,
                            gsplat::Rasterizer* raster, float scale_modifier) {
    (void)view;
    (void)scene_center;
    if (!raster) return;
    gsplat::RenderSettings rs = raster->settings();
    rs.scale_modifier = std::clamp(scale_modifier, 0.005f, 2.0f);
    raster->setSettings(rs);
}



/// 采样估算场景中心，用于相机参考点与 home 位置。
osg::Vec3d computeSceneCenter(const std::vector<gsplat::Gaussian>& g) {

    if (g.empty()) return osg::Vec3d(0, 0, 0);

    double sx = 0, sy = 0, sz = 0;

    const size_t n = std::min(g.size(), size_t(20000));

    const size_t step = std::max(size_t(1), g.size() / n);

    size_t count = 0;

    for (size_t i = 0; i < g.size(); i += step) {

        sx += g[i].x;

        sy += g[i].y;

        sz += g[i].z;

        ++count;

    }

    if (count == 0) return osg::Vec3d(0, 0, 0);

    const double inv = 1.0 / static_cast<double>(count);

    return osg::Vec3d(sx * inv, sy * inv, sz * inv);

}



/// SDK 预览/截图分辨率：与 OSG 当前视口 1:1（不再按点数强制缩到 640）。
void previewRenderDims(int vp_w, int vp_h, int /*num_gaussians*/, int& out_w, int& out_h) {
    out_w = std::max(1, vp_w);
    out_h = std::max(1, vp_h);
}

/// PostDraw：SSBO 已在主相机绘制时写好；CUDA-GL interop map 后在 GPU 上 unpack（无 CPU 回读）。
gsplat::Status unpackGlScreenCache(app::GlScreenGaussianCache& cache) {
    if (cache.sourceCount() <= 0) return gsplat::Status::ErrorNoGaussians;
    if (!gsplat::isCudaAvailable()) return gsplat::Status::ErrorNoCuda;
    return cache.unpackSsboToDevice();
}

/// 预览运行时状态（纹理、相机、交互标志、缓存帧）。
struct PreviewState {

    gsplat::Rasterizer* raster = nullptr;

    const std::vector<gsplat::Gaussian>* source_cloud = nullptr;

    app::GlScreenGaussianCache gl_screen_cache;

    osg::Vec3d scene_center;
    double scene_radius = 10.0;

    osg::ref_ptr<osg::Texture2D> texture;

    osg::ref_ptr<osg::Image> image;



    std::vector<uint8_t> last_rgb;

    std::vector<uint8_t> texture_rgb;

    int last_w = 0;

    int last_h = 0;

    double last_view[16] = {};

    double last_proj[16] = {};

    bool have_last_mats = false;

    double last_render_time_s = 0.0;



    bool preview_enabled = true;

    bool interacting = false;

    bool force_render = false;

    int warmup_frames = 0;

    bool logged_first_render = false;
    bool logged_prep_fail = false;
    int zero_visible_streak = 0;

    int capture_idx = 0;
    bool pending_capture = false;
    bool pending_truth_dump = false;
    float scale_modifier = 0.03f;

    gsplat::Camera last_gcam;
    bool have_last_gcam = false;
    gsplat::Camera last_ssbo_gcam;
    bool have_last_ssbo_gcam = false;
    int last_raster_visible = 0;
    int last_mat_mode = -1;
};

/// 一次 glDrawArrays：深度测试赢家片元写入 W×H SSBO 栅格。
class ScreenSsboDrawDrawable : public osg::Drawable {
public:
    ScreenSsboDrawDrawable() {
        setSupportsDisplayList(false);
        setUseVertexBufferObjects(false);
    }

    void setCache(app::GlScreenGaussianCache* cache) { cache_ = cache; }

    void drawImplementation(osg::RenderInfo& renderInfo) const override {
        if (!cache_ || cache_->sourceCount() <= 0) return;
        osg::State* st = renderInfo.getState();
        if (!st) return;
        osg::Matrixd V;
        osg::Matrixd P;
        int w = 1280;
        int h = 720;
        if (!getDrawMatrices(renderInfo, renderInfo.getCurrentCamera(), V, P, w, h)) {
            return;
        }
        cache_->drawPointsAndSsbo(st, V, P, w, h);
    }

    osg::Object* cloneType() const override { return new ScreenSsboDrawDrawable(); }
    osg::Object* clone(const osg::CopyOp& copyop) const override {
        auto* d = new ScreenSsboDrawDrawable();
        d->cache_ = cache_;
        (void)copyop;
        return d;
    }
    bool isSameKindAs(const osg::Object* obj) const override {
        return dynamic_cast<const ScreenSsboDrawDrawable*>(obj) != nullptr;
    }
    const char* libraryName() const override { return "gsplat_osg_app"; }
    const char* className() const override { return "ScreenSsboDrawDrawable"; }

protected:
    ~ScreenSsboDrawDrawable() override = default;

private:
    app::GlScreenGaussianCache* cache_ = nullptr;
};

osg::ref_ptr<osg::Node> buildScreenSsboSceneNode(app::GlScreenGaussianCache& cache) {
    osg::ref_ptr<ScreenSsboDrawDrawable> draw = new ScreenSsboDrawDrawable();
    draw->setCache(&cache);
    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(draw.get());
    osg::StateSet* ss = geode->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);
    return geode;
}

/// 导出真值与截图：保存参数、渲染 PNG、做回放对比。
void runTruthCapture(PreviewState* state, osg::State* gl_state, const osg::Matrixd& V, const osg::Matrixd& P,
                     int vp_w, int vp_h, bool do_png, const gsplat::Camera* frozen_cam, int frozen_w,
                     int frozen_h, bool reuse_screen_buffers) {
    if (!state || !state->raster || !gl_state) return;

    const osg::Matrixd& P_use = P;

    double view_a[16], proj_a[16];
    osgMatrixToArray(V, view_a);
    osgMatrixToArray(P_use, proj_a);

    gsplat::Camera gcam;
    int cap_w = std::max(1, vp_w);
    int cap_h = std::max(1, vp_h);
    int snap_mat_mode = state->last_mat_mode;
    double view_gl[16];
    double proj_gl[16];
    if (frozen_cam && frozen_w > 0 && frozen_h > 0) {
        gcam = *frozen_cam;
        cap_w = frozen_w;
        cap_h = frozen_h;
        if (state->have_last_mats) {
            std::memcpy(view_gl, state->last_view, sizeof(view_gl));
            std::memcpy(proj_gl, state->last_proj, sizeof(proj_gl));
        } else {
            std::memcpy(view_gl, view_a, sizeof(view_gl));
            std::memcpy(proj_gl, proj_a, sizeof(proj_gl));
        }
        std::cout << "[OSG APP] capture uses preview camera " << cap_w << "x" << cap_h << "\n";
    } else {
        updateScaleForSnapshot(V, state->scene_center, state->raster, state->scale_modifier);
        buildRenderCamera(state->gl_screen_cache, V, P_use, state->scene_center, gcam, &snap_mat_mode,
                          state->source_cloud);
        previewRenderDims(vp_w, vp_h, state->gl_screen_cache.sourceCount(), cap_w, cap_h);
        std::memcpy(view_gl, view_a, sizeof(view_gl));
        std::memcpy(proj_gl, proj_a, sizeof(proj_gl));
    }

    const int idx = state->capture_idx++;
    const std::string truth_path = "camera_truth_" + std::to_string(idx) + ".json";
    const std::string capture_path = "capture_" + std::to_string(idx) + ".png";
    const std::string cli_path = "truth_cli_" + std::to_string(idx) + ".png";

    int visible = 0;
    const bool can_reuse = reuse_screen_buffers && state->gl_screen_cache.ssboReady() &&
                           state->gl_screen_cache.filledCount() > 0;
    gsplat::Status prep_st = gsplat::Status::Ok;
    if (!can_reuse || state->gl_screen_cache.deviceBuffers().count <= 0) {
        prep_st = unpackGlScreenCache(state->gl_screen_cache);
    }
    if (prep_st != gsplat::Status::Ok) {
        std::cout << "[OSG APP] capture unpack failed: " << gsplat::statusString(prep_st) << "\n";
    }
    if (can_reuse) {
        std::cout << "[OSG APP] capture reuses screen SSBO (" << cap_w << "x" << cap_h
                  << ") filled=" << state->gl_screen_cache.filledCount() << "\n";
    }
    const gsplat::DeviceGaussianBuffers& screen_buf = state->gl_screen_cache.deviceBuffers();
    float capture_scale = state->scale_modifier;
    if (do_png) {
        const gsplat::Status cap_st =
            (prep_st == gsplat::Status::Ok)
                ? renderToPngWithScaleFallback(state->raster, gcam, cap_w, cap_h, screen_buf, capture_path,
                                               state->scale_modifier, capture_scale)
                : prep_st;
        visible = (cap_st == gsplat::Status::Ok) ? state->raster->lastVisibleCount() : 0;
        if (cap_st == gsplat::Status::Ok) {
            std::cout << "[OSG APP] wrote " << capture_path << " (" << cap_w << "x" << cap_h
                      << ") splats=" << screen_buf.count << " raster_visible=" << visible
                      << " scale=" << capture_scale << (can_reuse ? " (ssbo gpu)" : "") << "\n";
        } else {
            std::cout << "[OSG APP] capture failed: " << gsplat::statusString(cap_st)
                      << " (try 0 to reset scale, then R)\n";
        }
    } else {
        std::vector<uint8_t> rgb;
        const gsplat::Status st =
            (prep_st == gsplat::Status::Ok)
                ? state->raster->render(gcam, cap_w, cap_h, screen_buf, rgb)
                : prep_st;
        visible = (st == gsplat::Status::Ok) ? state->raster->lastVisibleCount() : 0;
        if (st != gsplat::Status::Ok) {
            std::cout << "[OSG APP] truth render failed: " << gsplat::statusString(st) << "\n";
        }
    }

    CameraTruth truth;
    truth.width = cap_w;
    truth.height = cap_h;
    truth.mat_mode = snap_mat_mode;
    truth.proj_mul_pv = false;
    truth.scale_modifier = capture_scale;
    truth.dc_only = state->raster->settings().dc_only;
    truth.drop_sh_rest = true;
    truth.min_opacity = 0.002f;
    truth.visible = screen_buf.count;
    std::memcpy(truth.osg_view, view_a, sizeof(view_a));
    std::memcpy(truth.osg_proj, proj_a, sizeof(proj_a));
    truth.sdk = gcam;

    if (!saveCameraTruth(truth_path, truth)) {
        std::cout << "[OSG APP] failed to write " << truth_path << "\n";
    } else {
        printCameraTruthSummary(truth, truth_path.c_str());
    }

    gsplat::Camera rebuilt;
    buildCameraFromSnapshot(V, P_use, state->scene_center, rebuilt, nullptr, state->source_cloud);
    double sdk_view[16], sdk_proj[16], reb_view[16], reb_proj[16];
    for (int i = 0; i < 16; ++i) {
        sdk_view[i] = static_cast<double>(gcam.view[i]);
        sdk_proj[i] = static_cast<double>(gcam.proj[i]);
        reb_view[i] = static_cast<double>(rebuilt.view[i]);
        reb_proj[i] = static_cast<double>(rebuilt.proj[i]);
    }
    logMatrixDiff("sdk_view", sdk_view, "rebuilt_view", reb_view);
    logMatrixDiff("sdk_proj", sdk_proj, "rebuilt_proj", reb_proj);

    CameraTruth loaded;
    if (loadCameraTruth(truth_path, loaded)) {
        applyTruthRenderSettings(*state->raster, loaded);
        const gsplat::Status cli_prep = unpackGlScreenCache(state->gl_screen_cache);
        float cli_scale = loaded.scale_modifier;
        const gsplat::Status cli_st =
            (cli_prep == gsplat::Status::Ok)
                ? renderToPngWithScaleFallback(state->raster, loaded.sdk, cap_w, cap_h,
                                               state->gl_screen_cache.deviceBuffers(), cli_path,
                                               loaded.scale_modifier, cli_scale)
                : cli_prep;
        if (cli_st == gsplat::Status::Ok) {
            std::cout << "[OSG APP] truth_cli replay -> " << cli_path << " visible="
                      << state->raster->lastVisibleCount() << "\n";
        } else {
            std::cout << "[OSG APP] truth_cli replay failed: " << gsplat::statusString(cli_st) << "\n";
        }
        updateScaleForSnapshot(V, state->scene_center, state->raster, state->scale_modifier);
    }

    std::cout << "[TRUTH] offline check:\n"
              << "  gsplat_render_ply <ply> " << cli_path << " " << cap_w << " " << cap_h
              << " <max_points> --truth " << truth_path << "\n";
}

/// 创建左下角 HUD 相机，用于显示 SDK 预览贴图。
osg::ref_ptr<osg::Camera> createHudCamera(osg::Texture2D* tex) {

    auto* hud = new osg::Camera();

    hud->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    hud->setViewMatrix(osg::Matrix::identity());

    hud->setProjectionMatrix(osg::Matrix::ortho2D(0.0, 1.0, 0.0, 1.0));

    hud->setRenderOrder(osg::Camera::POST_RENDER);

    hud->setClearMask(GL_DEPTH_BUFFER_BIT);

    hud->setAllowEventFocus(false);



    auto* geode = new osg::Geode();

    osg::ref_ptr<osg::Geometry> quad =

        osg::createTexturedQuadGeometry(osg::Vec3(0.02f, 0.02f, 0.0f), osg::Vec3(0.46f, 0.0f, 0.0f),

                                      osg::Vec3(0.0f, 0.46f, 0.0f));

    auto* ss = quad->getOrCreateStateSet();

    ss->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);

    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    ss->setMode(GL_BLEND, osg::StateAttribute::ON);

    ss->setAttributeAndModes(new osg::BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    geode->addDrawable(quad.get());

    hud->addChild(geode);

    return hud;

}



/// 更新 HUD 纹理内容。
void updatePreviewTexture(PreviewState& st, int w, int h, const std::vector<uint8_t>& rgb) {

    st.texture_rgb = rgb;

    st.image->setImage(w, h, 1, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, st.texture_rgb.data(),

                       osg::Image::NO_DELETE);

    st.texture->setImage(st.image.get());

}

#ifdef _WIN32
/// IME/组合键产生的异常 KEY 事件（在 EventHandler 链中吞掉，减轻误触发）。
static bool isImeOrNonAsciiKeyEvent(const osgGA::GUIEventAdapter& ea) {
    const int key = ea.getKey();
    if (key <= 0) {
        return true;
    }
    if (key > 255) {
        return true;
    }
    if (key == 229) {
        return true;
    }
    return false;
}
#endif

/// 先于 InteractionHandler 注册（OSG 逆序调用）：吞掉 IME 相关按键，避免进入操纵器。
class ImeKeyFilterHandler : public osgGA::GUIEventHandler {
public:
    bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) override {
        (void)aa;
#ifdef _WIN32
        if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN ||
            ea.getEventType() == osgGA::GUIEventAdapter::KEYUP) {
            if (isImeOrNonAsciiKeyEvent(ea)) {
                return true;
            }
        }
#else
        (void)ea;
#endif
        return false;
    }
};

/// 交互事件处理：键盘控制预览与截图、鼠标状态追踪。
class InteractionHandler : public osgGA::GUIEventHandler {

public:

    explicit InteractionHandler(PreviewState* st) : state_(st) {}



    bool handle(const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa) override {

        if (!state_) return false;

        switch (ea.getEventType()) {

        case osgGA::GUIEventAdapter::PUSH:

        case osgGA::GUIEventAdapter::DRAG:

        case osgGA::GUIEventAdapter::SCROLL:

            state_->interacting = true;

            break;

        case osgGA::GUIEventAdapter::RELEASE:

            state_->interacting = false;

            state_->force_render = true;

            break;

        case osgGA::GUIEventAdapter::KEYDOWN: {
#ifdef _WIN32
            if (isImeOrNonAsciiKeyEvent(ea)) {
                return true;
            }
#endif
            const int mod = ea.getModKeyMask();
            if (mod & osgGA::GUIEventAdapter::MODKEY_ALT) {
                return false;
            }
            if (ea.getKey() == 'p' || ea.getKey() == 'P') {

                state_->preview_enabled = !state_->preview_enabled;

                std::cout << "[OSG APP] SDK live preview " << (state_->preview_enabled ? "ON" : "OFF")

                          << "\n";

                return true;

            }

            if (ea.getKey() == 'r' || ea.getKey() == 'R') {
                (void)aa;
                if (state_->scale_modifier > 1.0f) {
                    std::cout << "[OSG APP] warn: scale_modifier=" << state_->scale_modifier
                              << " may OOM; press 0 to reset before capture\n";
                }
                state_->pending_capture = true;
                state_->force_render = true;
                std::cout << "[OSG APP] truth capture queued (next rendered frame)\n";
                return true;
            }
            if (ea.getKey() == 't' || ea.getKey() == 'T') {
                (void)aa;
                state_->pending_truth_dump = true;
                state_->force_render = true;
                std::cout << "[OSG APP] truth dump queued (next rendered frame)\n";
                return true;
            }

            if (ea.getKey() == '[' || ea.getKey() == '-') {
                state_->scale_modifier = std::max(0.005f, state_->scale_modifier / 1.5f);
                state_->force_render = true;
                updateScaleForSnapshot(osg::Matrixd(), state_->scene_center, state_->raster,
                                       state_->scale_modifier);
                std::cout << "[OSG APP] scale_modifier=" << state_->scale_modifier
                          << " applied=" << state_->raster->settings().scale_modifier << "\n";
                return true;
            }
            if (ea.getKey() == ']' || ea.getKey() == '}' || ea.getKey() == '=' || ea.getKey() == '+') {
                state_->scale_modifier = std::min(2.0f, state_->scale_modifier * 1.5f);
                state_->force_render = true;
                updateScaleForSnapshot(osg::Matrixd(), state_->scene_center, state_->raster,
                                       state_->scale_modifier);
                std::cout << "[OSG APP] scale_modifier=" << state_->scale_modifier
                          << " applied=" << state_->raster->settings().scale_modifier << "\n";
                return true;
            }
            if (ea.getKey() == '0') {
                state_->scale_modifier = 0.03f;
                state_->force_render = true;
                updateScaleForSnapshot(osg::Matrixd(), state_->scene_center, state_->raster,
                                       state_->scale_modifier);
                std::cout << "[OSG APP] scale_modifier reset=" << state_->scale_modifier << "\n";
                return true;
            }
            break;
        }

        default:

            break;

        }

        return false;

    }



private:
    PreviewState* state_ = nullptr;

};



/// PostDraw 回调：按当前视角调用 SDK 渲染并刷新 HUD。
class SdkPostDrawCallback : public osg::Camera::DrawCallback {

public:

    explicit SdkPostDrawCallback(PreviewState* st) : state_(st) {}



    void operator()(osg::RenderInfo& renderInfo) const override {

        if (!state_) return;

        if (!state_->raster) return;

        const bool want_capture = state_->pending_capture || state_->pending_truth_dump;
        if (!state_->preview_enabled && !want_capture) return;

        osg::Camera* cam = renderInfo.getCurrentCamera();

        if (!cam) return;

        if (state_->warmup_frames < 2) {

            ++state_->warmup_frames;

            return;

        }

        osg::Matrixd V;
        osg::Matrixd P;
        int vp_w = 1280;
        int vp_h = 720;
        if (!getDrawMatrices(renderInfo, cam, V, P, vp_w, vp_h)) return;

        GLint gl_vp[4] = {0, 0, 0, 0};
        glGetIntegerv(GL_VIEWPORT, gl_vp);
        if (gl_vp[2] > 0 && gl_vp[3] > 0) {
            vp_w = gl_vp[2];
            vp_h = gl_vp[3];
        }

        const bool do_capture_png = state_->pending_capture;
        const bool do_capture_truth = state_->pending_truth_dump;

        double view_a[16], proj_a[16];

        osgMatrixToArray(V, view_a);

        osgMatrixToArray(P, proj_a);

        const bool mats_changed =

            !state_->have_last_mats || !matricesEqual(view_a, state_->last_view) ||

            !matricesEqual(proj_a, state_->last_proj);



        const double now_s = osg::Timer::instance()->time_s();

        const bool throttle =

            !state_->force_render && (now_s - state_->last_render_time_s) < kMinRenderIntervalSec;



        const bool must_capture = do_capture_png || do_capture_truth;

        if (state_->interacting && !state_->force_render && !must_capture) {
            return;
        }

        if (!must_capture && !mats_changed && !state_->force_render && !state_->last_rgb.empty()) {
            return;
        }

        if (!must_capture && throttle && !state_->last_rgb.empty()) {
            return;
        }



        int rw = vp_w;
        int rh = vp_h;
        previewRenderDims(vp_w, vp_h, state_->gl_screen_cache.sourceCount(), rw, rh);

        gsplat::Camera gcam{};
        osg::State* gl_state = renderInfo.getState();

        std::vector<uint8_t> rgb;
        const gsplat::Status prep_st =
            gl_state ? unpackGlScreenCache(state_->gl_screen_cache) : gsplat::Status::ErrorRenderFailed;
        updateScaleForSnapshot(V, state_->scene_center, state_->raster, state_->scale_modifier);
        const bool used_ssbo_cam = buildRenderCamera(state_->gl_screen_cache, V, P, state_->scene_center, gcam,
                                                   &state_->last_mat_mode, state_->source_cloud);
        auditRenderCamera(gcam, used_ssbo_cam, V, P, state_->scene_center, state_->source_cloud);
        const gsplat::DeviceGaussianBuffers& screen_buf = state_->gl_screen_cache.deviceBuffers();
        const gsplat::Status st =
            (prep_st == gsplat::Status::Ok) ? state_->raster->render(gcam, rw, rh, screen_buf, rgb) : prep_st;
        const int ssbo_filled = (prep_st == gsplat::Status::Ok) ? state_->gl_screen_cache.filledCount() : 0;
        const int raster_visible = (st == gsplat::Status::Ok) ? state_->raster->lastVisibleCount() : 0;

        if (st != gsplat::Status::Ok) {
            if (!state_->logged_prep_fail) {
                state_->logged_prep_fail = true;
                std::cerr << "[OSG APP] preview failed: prep=" << gsplat::statusString(prep_st)
                          << " render=" << gsplat::statusString(st) << "\n";
            }
            if (do_capture_png || do_capture_truth) {
                state_->pending_capture = false;
                state_->pending_truth_dump = false;
                runTruthCapture(state_, gl_state, V, P, vp_w, vp_h, do_capture_png, &gcam, rw, rh, false);
            }
            return;
        }
        if (raster_visible <= 0) {
            state_->zero_visible_streak++;
            if (state_->zero_visible_streak == 1) {
                std::cerr << "[OSG APP] CUDA raster visible=0 (ssbo_filled=" << ssbo_filled
                          << " P=" << screen_buf.count << " " << rw << "x" << rh
                          << " scale_mod=" << state_->scale_modifier << " prep="
                          << gsplat::statusString(prep_st) << ")\n";
            }
            if (do_capture_png || do_capture_truth) {
                state_->pending_capture = false;
                state_->pending_truth_dump = false;
                std::cout << "[OSG APP] visible=0, capturing current frame anyway\n";
                runTruthCapture(state_, gl_state, V, P, vp_w, vp_h, do_capture_png, &gcam, rw, rh, false);
                return;
            }
            if (!state_->last_rgb.empty()) {
                return;
            }
        } else {
            state_->zero_visible_streak = 0;
            state_->last_gcam = gcam;
            state_->have_last_gcam = true;
        }

        state_->last_rgb = std::move(rgb);
        state_->last_raster_visible = raster_visible;

        state_->last_w = rw;

        state_->last_h = rh;

        std::memcpy(state_->last_view, view_a, sizeof(view_a));

        std::memcpy(state_->last_proj, proj_a, sizeof(proj_a));

        state_->have_last_mats = true;

        state_->last_render_time_s = now_s;

        state_->force_render = false;



        updatePreviewTexture(*state_, rw, rh, state_->last_rgb);

        if (do_capture_png || do_capture_truth) {
            state_->pending_capture = false;
            state_->pending_truth_dump = false;
            const gsplat::Camera* cap_cam = &gcam;
            const int cap_w = rw;
            const int cap_h = rh;
            if (cap_cam && cap_w > 0 && cap_h > 0) {
                runTruthCapture(state_, gl_state, V, P, vp_w, vp_h, do_capture_png, cap_cam, cap_w, cap_h,
                                true);
            } else {
                std::cout << "[OSG APP] capture skipped: no valid preview camera\n";
            }
        }

        if (!state_->preview_enabled) return;

        if (!state_->logged_first_render) {

            state_->logged_first_render = true;

            std::cout << "[OSG APP] SDK Inria raster " << rw << "x" << rh << " visible=" << raster_visible
                      << " ssbo_gpu=" << screen_buf.count << " filled=" << ssbo_filled << " / "
                      << state_->gl_screen_cache.sourceCount() << " mat_mode=" << state_->last_mat_mode
                      << " (viewport " << vp_w << "x" << vp_h << ")\n";

        }

    }



private:

    PreviewState* state_ = nullptr;

};



/// 不区分大小写判断字符串后缀。
bool endsWithIgnoreCase(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(s[s.size() - n + i])));
        const char b = static_cast<char>(std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) return false;
    }
    return true;
}

/// 判断参数是否像 PLY 文件路径。
bool looksLikePlyPath(const char* arg) {
    if (!arg || !arg[0]) return false;
    const std::string s(arg);
    return endsWithIgnoreCase(s, ".ply") || s.find(".ply") != std::string::npos;
}

/// 解析 max_points 参数（正整数且在合理范围内）。
bool parseMaxPointsArg(const char* arg, size_t& out) {
    if (!arg || !arg[0]) return false;
    try {
        size_t pos = 0;
        const unsigned long long v = std::stoull(arg, &pos);
        if (pos == 0) return false;
        while (arg[pos] != '\0' && std::isspace(static_cast<unsigned char>(arg[pos]))) ++pos;
        if (arg[pos] != '\0') return false;
        if (v == 0 || v > 50000000ULL) return false;
        out = static_cast<size_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

/// 解析命令行：支持 ply 与 max_points 任意顺序。
bool parseCli(int argc, char** argv, std::string& ply_out, size_t& max_points_out) {
    max_points_out = 200000;
    if (argc < 2) return false;

    const char* ply_arg = nullptr;
    const char* max_arg = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (looksLikePlyPath(argv[i])) {
            if (!ply_arg) ply_arg = argv[i];
        } else if (parseMaxPointsArg(argv[i], max_points_out)) {
            max_arg = argv[i];
        }
    }

    if (!ply_arg && argc >= 2) {
        if (looksLikePlyPath(argv[1])) {
            ply_arg = argv[1];
        } else if (argc >= 3 && looksLikePlyPath(argv[2])) {
            ply_arg = argv[2];
            if (!max_arg) parseMaxPointsArg(argv[1], max_points_out);
        }
    }

    if (!ply_arg) return false;
    ply_out = ply_arg;
    if (max_arg) parseMaxPointsArg(max_arg, max_points_out);
    return true;
}

/// 采样计算场景包围球（用于设置相机 home）。
void sceneBounds(const std::vector<gsplat::Gaussian>& g, osg::Vec3d& center, double& radius) {
    if (g.empty()) {
        center.set(0, 0, 0);
        radius = 10.0;
        return;
    }
    const size_t step = std::max(size_t(1), g.size() / 10000);
    double minx = g[0].x, maxx = g[0].x;
    double miny = g[0].y, maxy = g[0].y;
    double minz = g[0].z, maxz = g[0].z;
    for (size_t i = 0; i < g.size(); i += step) {
        minx = std::min(minx, static_cast<double>(g[i].x));
        maxx = std::max(maxx, static_cast<double>(g[i].x));
        miny = std::min(miny, static_cast<double>(g[i].y));
        maxy = std::max(maxy, static_cast<double>(g[i].y));
        minz = std::min(minz, static_cast<double>(g[i].z));
        maxz = std::max(maxz, static_cast<double>(g[i].z));
    }
    center.set(0.5 * (minx + maxx), 0.5 * (miny + maxy), 0.5 * (minz + maxz));
    const double dx = maxx - center.x();
    const double dy = maxy - center.y();
    const double dz = maxz - center.z();
    radius = std::sqrt(dx * dx + dy * dy + dz * dz);
    radius = std::max(radius, 1.0);
}

}  // namespace

/// OSG app 主入口：加载高斯、初始化 viewer，并在 PostDraw 中驱动 SDK 预览。
int main(int argc, char** argv) {
    std::cout << "[OSG APP] starting...\n" << std::flush;
    std::string ply;
    size_t max_points = 1200000;
    if (!parseCli(argc, argv, ply, max_points)) {
        std::cerr << "Usage: " << argv[0] << " <input.ply> [max_points]\n";
        std::cerr << "  Example: " << argv[0]
                  << " C:\\path\\point_cloud.ply 200000\n";
        std::cerr << "  (PLY path must end with .ply; max_points is optional.)\n";
        return 1;
    }



    std::vector<gsplat::Gaussian> gaussians;

    app::PlyLoadOptions opts;

    opts.max_points = max_points;
    // Keep more weak splats for floor/wheel continuity while avoiding heavy haze.
    opts.min_opacity = 0.002f;
    // Random reservoir subsample preserves scene coverage better than stride skipping.
    opts.stride_subsample = false;
    opts.drop_sh_rest = true;

    if (!app::loadGaussianPlyLocal(ply, gaussians, opts)) {

        std::cerr << "PLY load failed: " << ply << "\n";

        return 2;

    }

    std::cout << "[OSG APP] loaded " << gaussians.size() << " gaussians\n";



    if (!gsplat::isCudaAvailable()) {

        std::cerr << "[OSG APP] CUDA not available — SDK preview disabled.\n";
#ifdef GSPLAT_CUDA_ENABLED
        int dev_count = 0;
        const cudaError_t ce = cudaGetDeviceCount(&dev_count);
        const char* ce_msg = cudaGetErrorString(ce);
        const char* ce_name = cudaGetErrorName(ce);
        std::cerr << "[OSG APP] cudaGetDeviceCount: " << (ce_name ? ce_name : "?") << " (" << ce << ") devices="
                  << dev_count << " — " << (ce_msg && ce_msg[0] ? ce_msg : "no message") << "\n";
        if (ce == 35) {
            std::cerr << "[OSG APP] err=35: CUDA 13 runtime vs driver CUDA 12.6 — update NVIDIA driver, or put C:\\cuda118\\bin\\cudart64_110.dll next to exe and rebuild linking cuda118 (see README).\n";
        }
#endif

    }



    gsplat::Rasterizer raster;

    gsplat::RenderSettings rs;

    rs.dc_only = false;
    rs.scale_modifier = 0.03f;

    raster.setSettings(rs);

    PreviewState preview;

    preview.raster = &raster;
    preview.source_cloud = &gaussians;
    preview.scale_modifier = rs.scale_modifier;

    preview.scene_center = computeSceneCenter(gaussians);
    {
        osg::Vec3d bound_center;
        double bound_radius = 10.0;
        sceneBounds(gaussians, bound_center, bound_radius);
        preview.scene_radius = std::max(bound_radius, 1.0);
    }

    if (!preview.gl_screen_cache.uploadSource(gaussians, rs.dc_only)) {
        std::cerr << "GL source VBO upload failed\n";
        return 3;
    }
    preview.texture = new osg::Texture2D();

    preview.image = new osg::Image();

    preview.image->allocateImage(4, 4, 1, GL_RGB, GL_UNSIGNED_BYTE);

    std::memset(preview.image->data(), 0, 4 * 4 * 3);

    preview.texture->setImage(preview.image.get());

    osg::DisplaySettings* ds = osg::DisplaySettings::instance();
    ds->setGLContextVersion("4.3");

    osgViewer::Viewer viewer;

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->addChild(buildScreenSsboSceneNode(preview.gl_screen_cache).get());
    viewer.setSceneData(root.get());

    auto* manip = new osgGA::TrackballManipulator();
    osg::Vec3d bound_center;
    double bound_radius = 10.0;
    sceneBounds(gaussians, bound_center, bound_radius);

    if (bound_radius < 1e-6) bound_radius = 10.0;

    const double dist = bound_radius * 2.5;
    manip->setHomePosition(bound_center + osg::Vec3d(0, -dist, dist * 0.35), bound_center,
                           osg::Vec3d(0, 0, 1));
    viewer.setCameraManipulator(manip);

    viewer.addEventHandler(new InteractionHandler(&preview));
    viewer.addEventHandler(new ImeKeyFilterHandler());

    viewer.setUpViewInWindow(100, 100, 1280, 720);
    viewer.realize();
#ifdef _WIN32
    disableWindowsImeOnViewer(viewer);
#endif
    viewer.home();

    viewer.addSlave(createHudCamera(preview.texture.get()), false);
    viewer.getCamera()->setPostDrawCallback(new SdkPostDrawCallback(&preview));

    std::cout << "[OSG APP] roam: P=preview, R=capture, T=truth json, [ - / ] = + scale, 0=reset.\n";
    std::cout << "[OSG APP] CJK IME is disabled on the 3D window to prevent UI freeze.\n";

    return viewer.run();

}


