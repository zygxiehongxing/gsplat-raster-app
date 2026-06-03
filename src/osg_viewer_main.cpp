#include <gsplat_raster/gsplat_raster.h>

#include "camera_truth.h"
#include "ply_loader_local.h"

#include <osg/BlendFunc>

#include <osg/GL>

#include <osg/Camera>

#include <osg/Geode>

#include <osg/Geometry>

#include <osg/Image>

#include <osg/Point>

#include <osg/Texture2D>

#include <osg/Timer>

#include <osgGA/GUIEventHandler>

#include <osgGA/TrackballManipulator>

#include <osgViewer/Viewer>



#include <algorithm>

#include <cctype>

#include <cmath>

#include <cstring>

#include <iostream>

#include <limits>

#include <string>

#include <vector>



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

/// 4x4 行主序矩阵转置。
void transposeMat4(const double in[16], double out[16]) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) out[r * 4 + c] = in[c * 4 + r];
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

/// 打印投影参数（透视 near/far 或正交 near/far），用于对齐排查。
void logProjNearFar(const char* tag, const osg::Matrixd& proj) {
    double fovy = 0.0, aspect = 1.0, znear = 0.01, zfar = 10000.0;
    if (proj.getPerspective(fovy, aspect, znear, zfar)) {
        std::cout << "[TRUTH] " << tag << " perspective near/far=" << znear << "/" << zfar
                  << " fovy=" << fovy << " aspect=" << aspect << "\n";
        return;
    }
    double l = 0.0, r = 0.0, b = 0.0, t = 0.0, n = 0.0, f = 0.0;
    if (proj.getOrtho(l, r, b, t, n, f)) {
        std::cout << "[TRUTH] " << tag << " ortho near/far=" << n << "/" << f << "\n";
        return;
    }
    std::cout << "[TRUTH] " << tag << " projection decode failed\n";
}

/// 使用当前帧 OSG view/proj 构建 SDK 相机参数。
void buildCameraFromSnapshot(const osg::Matrixd& view, const osg::Matrixd& proj,
                             const osg::Vec3d& scene_center, gsplat::Camera& out) {
    double fovy_deg = 0.0;
    double aspect = 1.0;
    double real_near = 0.01;
    double real_far = 10000.0;
    osg::Vec3d eye, center, up;
    if (proj.getPerspective(fovy_deg, aspect, real_near, real_far) &&
        std::isfinite(fovy_deg) && fovy_deg > 0.01 && std::isfinite(aspect) && aspect > 1e-6 &&
        std::isfinite(real_near) && std::isfinite(real_far) && real_near > 1e-6 && real_far > real_near) {
        // 使用 OSG 官方分解，避免手写矩阵约定误差导致的拉伸/翻转。
        view.getLookAt(eye, center, up);
        const double eye_a[3] = {eye.x(), eye.y(), eye.z()};
        const double center_a[3] = {center.x(), center.y(), center.z()};
        const double up_a[3] = {up.x(), up.y(), up.z()};
        double ref_center[3];
        sceneRefCenter(scene_center, ref_center);
        gsplat::buildCameraLookAt(eye_a, center_a, up_a, fovy_deg, aspect, real_near, real_far, ref_center, out,
                                  nullptr);
        return;
    }

    double view_a[16];
    double proj_a[16];
    osgMatrixToArray(view, view_a);
    osgMatrixToArray(proj, proj_a);
    double ref_center[3];
    sceneRefCenter(scene_center, ref_center);
    // 回退：若非透视投影，仍走矩阵转换路径。
    gsplat::buildCameraFromOsg(view_a, proj_a, ref_center, out);
}

/// 获取当前绘制帧的相机矩阵与视口尺寸。
bool getDrawMatrices(osg::RenderInfo& renderInfo, osg::Camera* cam, osg::Matrixd& V, osg::Matrixd& P,
                     int& vp_w, int& vp_h, const char*& source_tag) {
    vp_w = 1280;
    vp_h = 720;
    source_tag = "none";
    // view 必须来自 camera，避免 State::ModelView 混入模型变换导致姿态错误。
    if (cam) {
        V = cam->getViewMatrix();
        source_tag = "camera";
        // projection 优先用 State（可拿到动态 near/far），否则回退 camera projection。
        osg::State* osg_state = renderInfo.getState();
        if (osg_state) {
            P = osg_state->getProjectionMatrix();
            source_tag = "camera_view+state_proj";
            const osg::Viewport* vp = osg_state->getCurrentViewport();
            if (vp && vp->width() > 0 && vp->height() > 0) {
                vp_w = std::max(1, static_cast<int>(vp->width()));
                vp_h = std::max(1, static_cast<int>(vp->height()));
                return true;
            }
        } else {
            P = cam->getProjectionMatrix();
        }
        const osg::Viewport* vp = cam->getViewport();
        if (vp && vp->width() > 0 && vp->height() > 0) {
            vp_w = std::max(1, static_cast<int>(vp->width()));
            vp_h = std::max(1, static_cast<int>(vp->height()));
        }
        return true;
    }
    // 无 camera 时才回退 State 矩阵。
    osg::State* osg_state = renderInfo.getState();
    if (osg_state) {
        V = osg_state->getModelViewMatrix();
        P = osg_state->getProjectionMatrix();
        source_tag = "state_fallback";
        const osg::Viewport* vp = osg_state->getCurrentViewport();
        if (vp && vp->width() > 0 && vp->height() > 0) {
            vp_w = std::max(1, static_cast<int>(vp->width()));
            vp_h = std::max(1, static_cast<int>(vp->height()));
        }
        return true;
    }
    return false;
}

struct PreviewState;

/// 执行一次真值导出/截图流程（truth json + capture + replay）。
void runTruthCapture(PreviewState* state, const osg::Matrixd& V, const osg::Matrixd& P, int vp_w, int vp_h,
                     bool do_png, const gsplat::Camera* frozen_cam = nullptr, int frozen_w = 0,
                     int frozen_h = 0);

/// 将交互调节后的 scale_modifier 应用到 raster 设置。
void updateScaleForSnapshot(const osg::Matrixd& view, const osg::Vec3d& scene_center,
                            gsplat::Rasterizer* raster, float scale_modifier) {
    (void)view;
    (void)scene_center;
    if (!raster) return;
    gsplat::RenderSettings rs = raster->settings();
    rs.scale_modifier = std::clamp(scale_modifier, 0.005f, 0.5f);
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



/// 构建 OSG 点预览节点（GPU splat 之外的参考视图）。
osg::ref_ptr<osg::Node> buildPointNode(const std::vector<gsplat::Gaussian>& g) {

    auto geode = osg::ref_ptr<osg::Geode>(new osg::Geode());

    auto geom = osg::ref_ptr<osg::Geometry>(new osg::Geometry());



    auto verts = osg::ref_ptr<osg::Vec3Array>(new osg::Vec3Array());

    auto colors = osg::ref_ptr<osg::Vec4Array>(new osg::Vec4Array());

    verts->reserve(g.size());

    colors->reserve(g.size());



    for (const auto& p : g) {

        verts->push_back(osg::Vec3(p.x, p.y, p.z));

        const float r = clamp01(0.5f + kShC0 * p.sh[0]);

        const float gg = clamp01(0.5f + kShC0 * p.sh[1]);

        const float b = clamp01(0.5f + kShC0 * p.sh[2]);

        colors->push_back(osg::Vec4(r, gg, b, 1.f));

    }



    geom->setVertexArray(verts.get());

    geom->setColorArray(colors.get(), osg::Array::BIND_PER_VERTEX);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_POINTS, 0, static_cast<GLsizei>(verts->size())));

    geode->addDrawable(geom.get());



    auto point_state = geode->getOrCreateStateSet();

    auto point = osg::ref_ptr<osg::Point>(new osg::Point());

    point->setSize(2.0f);

    point_state->setAttribute(point.get());

    point_state->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    return geode;

}



/// 预览运行时状态（纹理、相机、交互标志、缓存帧）。
struct PreviewState {

    gsplat::Rasterizer* raster = nullptr;

    osg::Vec3d scene_center;

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
    int zero_visible_streak = 0;

    int capture_idx = 0;
    bool pending_capture = false;
    bool pending_truth_dump = false;
    float scale_modifier = 0.03f;

    gsplat::Camera last_gcam;
    bool have_last_gcam = false;

};

/// 导出真值与截图：保存参数、渲染 PNG、做回放对比。
void runTruthCapture(PreviewState* state, const osg::Matrixd& V, const osg::Matrixd& P, int vp_w, int vp_h,
                     bool do_png, const gsplat::Camera* frozen_cam, int frozen_w, int frozen_h) {
    if (!state || !state->raster) return;

    double view_a[16], proj_a[16];
    osgMatrixToArray(V, view_a);
    osgMatrixToArray(P, proj_a);

    gsplat::Camera gcam;
    int cap_w = std::max(1, vp_w);
    int cap_h = std::max(1, vp_h);
    if (frozen_cam && frozen_w > 0 && frozen_h > 0) {
        gcam = *frozen_cam;
        cap_w = frozen_w;
        cap_h = frozen_h;
        std::cout << "[OSG APP] capture uses preview camera " << cap_w << "x" << cap_h << "\n";
    } else {
        updateScaleForSnapshot(V, state->scene_center, state->raster, state->scale_modifier);
        buildCameraFromSnapshot(V, P, state->scene_center, gcam);
        previewRenderDims(vp_w, vp_h, state->raster->numGaussians(), cap_w, cap_h);
    }

    const int idx = state->capture_idx++;
    const std::string truth_path = "camera_truth_" + std::to_string(idx) + ".json";
    const std::string capture_path = "capture_" + std::to_string(idx) + ".png";
    const std::string cli_path = "truth_cli_" + std::to_string(idx) + ".png";

    int visible = 0;
    if (do_png) {
        const gsplat::Status cap_st = state->raster->renderToPng(gcam, cap_w, cap_h, capture_path);
        visible = (cap_st == gsplat::Status::Ok) ? state->raster->lastVisibleCount() : 0;
        if (cap_st == gsplat::Status::Ok) {
            std::cout << "[OSG APP] wrote " << capture_path << " (" << cap_w << "x" << cap_h
                      << ") visible=" << visible << "\n";
        } else {
            std::cout << "[OSG APP] capture failed: " << gsplat::statusString(cap_st) << "\n";
        }
    } else {
        std::vector<uint8_t> rgb;
        const gsplat::Status st = state->raster->render(gcam, cap_w, cap_h, rgb);
        visible = (st == gsplat::Status::Ok) ? state->raster->lastVisibleCount() : 0;
        if (st != gsplat::Status::Ok) {
            std::cout << "[OSG APP] truth render failed: " << gsplat::statusString(st) << "\n";
        }
    }

    CameraTruth truth;
    truth.width = cap_w;
    truth.height = cap_h;
    truth.mat_mode = -1;
    truth.proj_mul_pv = false;
    truth.scale_modifier = state->scale_modifier;
    truth.dc_only = state->raster->settings().dc_only;
    truth.drop_sh_rest = true;
    truth.min_opacity = 0.002f;
    truth.visible = visible;
    std::memcpy(truth.osg_view, view_a, sizeof(view_a));
    std::memcpy(truth.osg_proj, proj_a, sizeof(proj_a));
    truth.sdk = gcam;

    if (!saveCameraTruth(truth_path, truth)) {
        std::cout << "[OSG APP] failed to write " << truth_path << "\n";
    } else {
        printCameraTruthSummary(truth, truth_path.c_str());
    }

    gsplat::Camera rebuilt;
    buildCameraFromSnapshot(V, P, state->scene_center, rebuilt);
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
        const gsplat::Status cli_st = state->raster->renderToPng(loaded.sdk, cap_w, cap_h, cli_path);
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

        case osgGA::GUIEventAdapter::KEYDOWN:

            if (ea.getKey() == 'p' || ea.getKey() == 'P') {

                state_->preview_enabled = !state_->preview_enabled;

                std::cout << "[OSG APP] SDK live preview " << (state_->preview_enabled ? "ON" : "OFF")

                          << "\n";

                return true;

            }

            if (ea.getKey() == 'r' || ea.getKey() == 'R') {
                (void)aa;
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

            if (ea.getKey() == '[') {
                state_->scale_modifier = std::max(0.005f, state_->scale_modifier * 0.8f);
                state_->force_render = true;
                std::cout << "[OSG APP] scale_modifier=" << state_->scale_modifier << "\n";
                return true;
            }
            if (ea.getKey() == ']') {
                state_->scale_modifier = std::min(0.5f, state_->scale_modifier * 1.25f);
                state_->force_render = true;
                std::cout << "[OSG APP] scale_modifier=" << state_->scale_modifier << "\n";
                return true;
            }
            if (ea.getKey() == '0') {
                state_->scale_modifier = 0.03f;
                state_->force_render = true;
                std::cout << "[OSG APP] scale_modifier reset=" << state_->scale_modifier << "\n";
                return true;
            }

            break;

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

        if (!state_ || !state_->raster) return;

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
        const char* mats_source = "none";
        if (!getDrawMatrices(renderInfo, cam, V, P, vp_w, vp_h, mats_source)) return;

        GLint gl_vp[4] = {0, 0, 0, 0};
        glGetIntegerv(GL_VIEWPORT, gl_vp);
        if (gl_vp[2] > 0 && gl_vp[3] > 0) {
            vp_w = gl_vp[2];
            vp_h = gl_vp[3];
        }

        const bool do_capture_png = state_->pending_capture;
        const bool do_capture_truth = state_->pending_truth_dump;
        if (do_capture_png || do_capture_truth) {
            std::cout << "[TRUTH] draw matrix source=" << mats_source << "\n";
            if (osg::State* st_state = renderInfo.getState()) {
                logProjNearFar("state_proj", st_state->getProjectionMatrix());
            } else {
                std::cout << "[TRUTH] state_proj unavailable\n";
            }
            logProjNearFar("camera_proj", cam->getProjectionMatrix());
            logProjNearFar("used_proj", P);
        }

        double view_a[16], proj_a[16];

        osgMatrixToArray(V, view_a);

        osgMatrixToArray(P, proj_a);

        const bool mats_changed =

            !state_->have_last_mats || !matricesEqual(view_a, state_->last_view) ||

            !matricesEqual(proj_a, state_->last_proj);



        const double now_s = osg::Timer::instance()->time_s();

        const bool throttle =

            !state_->force_render && (now_s - state_->last_render_time_s) < kMinRenderIntervalSec;



        if (state_->interacting && !state_->force_render) {

            return;

        }

        if (!mats_changed && !state_->force_render && !state_->last_rgb.empty()) {

            return;

        }

        if (throttle && !state_->last_rgb.empty()) {

            return;

        }



        int rw = vp_w;

        int rh = vp_h;

        previewRenderDims(vp_w, vp_h, state_->raster->numGaussians(), rw, rh);



        gsplat::Camera gcam;
        updateScaleForSnapshot(V, state_->scene_center, state_->raster, state_->scale_modifier);
        buildCameraFromSnapshot(V, P, state_->scene_center, gcam);



        std::vector<uint8_t> rgb;
        gsplat::Status st = state_->raster->render(gcam, rw, rh, rgb);
        int visible = (st == gsplat::Status::Ok) ? state_->raster->lastVisibleCount() : 0;

        if (st != gsplat::Status::Ok) {
            if (do_capture_png || do_capture_truth) {
                state_->pending_capture = do_capture_png;
                state_->pending_truth_dump = do_capture_truth;
                if (state_->have_last_gcam) {
                    state_->pending_capture = false;
                    state_->pending_truth_dump = false;
                    runTruthCapture(state_, V, P, vp_w, vp_h, do_capture_png, &state_->last_gcam,
                                    state_->last_w, state_->last_h);
                } else {
                    std::cout << "[OSG APP] capture skipped: preview render failed\n";
                }
            }
            return;
        }
        if (visible <= 0) {
            state_->zero_visible_streak++;
            if (do_capture_png || do_capture_truth) {
                state_->pending_capture = false;
                state_->pending_truth_dump = false;
                if (state_->have_last_gcam) {
                    runTruthCapture(state_, V, P, vp_w, vp_h, do_capture_png, &state_->last_gcam,
                                    state_->last_w, state_->last_h);
                } else {
                    // 首帧或无历史结果时也强制落盘，避免 R/T 没有任何输出文件。
                    std::cout << "[OSG APP] visible=0, capturing current frame anyway\n";
                    runTruthCapture(state_, V, P, vp_w, vp_h, do_capture_png, &gcam, rw, rh);
                }
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
            const gsplat::Camera* cap_cam =
                (visible > 0) ? &gcam : (state_->have_last_gcam ? &state_->last_gcam : nullptr);
            const int cap_w = (visible > 0) ? rw : state_->last_w;
            const int cap_h = (visible > 0) ? rh : state_->last_h;
            if (cap_cam && cap_w > 0 && cap_h > 0) {
                runTruthCapture(state_, V, P, vp_w, vp_h, do_capture_png, cap_cam, cap_w, cap_h);
            } else {
                std::cout << "[OSG APP] capture skipped: no valid preview camera\n";
            }
        }

        if (!state_->preview_enabled) return;

        if (!state_->logged_first_render) {

            state_->logged_first_render = true;

            std::cout << "[OSG APP] SDK preview " << rw << "x" << rh << " (viewport " << vp_w << "x" << vp_h
                      << ", 1:1), visible ~" << visible << " / " << state_->raster->numGaussians()
                      << "\n";

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

    }



    gsplat::Rasterizer raster;

    gsplat::RenderSettings rs;

    rs.dc_only = false;
    rs.scale_modifier = 0.02f;

    raster.setSettings(rs);

    if (raster.setGaussians(gaussians) != gsplat::Status::Ok) {

        std::cerr << "Raster upload failed\n";

        return 3;

    }



    PreviewState preview;

    preview.raster = &raster;
    preview.scale_modifier = rs.scale_modifier;

    preview.scene_center = computeSceneCenter(gaussians);

    preview.texture = new osg::Texture2D();

    preview.image = new osg::Image();

    preview.image->allocateImage(4, 4, 1, GL_RGB, GL_UNSIGNED_BYTE);

    std::memset(preview.image->data(), 0, 4 * 4 * 3);

    preview.texture->setImage(preview.image.get());



    osgViewer::Viewer viewer;

    viewer.setSceneData(buildPointNode(gaussians).get());

    auto* manip = new osgGA::TrackballManipulator();
    osg::Vec3d bound_center;
    double bound_radius = 10.0;
    sceneBounds(gaussians, bound_center, bound_radius);
    const double dist = bound_radius * 2.5;
    manip->setHomePosition(bound_center + osg::Vec3d(0, -dist, dist * 0.35), bound_center,
                           osg::Vec3d(0, 0, 1));
    viewer.setCameraManipulator(manip);

    viewer.addEventHandler(new InteractionHandler(&preview));

    viewer.setUpViewInWindow(100, 100, 1280, 720);
    viewer.home();



    viewer.addSlave(createHudCamera(preview.texture.get()), false);



    viewer.getCamera()->setPostDrawCallback(new SdkPostDrawCallback(&preview));



    std::cout << "[OSG APP] roam with mouse; SDK preview updates when you release the mouse.\n";

    std::cout << "[OSG APP] P=toggle preview, R=truth capture, T=truth dump only, [ / ] scale, 0=reset.\n";

    return viewer.run();

}


