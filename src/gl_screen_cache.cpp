#include "gl_screen_cache.h"

#include <gsplat_raster/screen_gaussian_pixel.h>

#include <osg/GL>
#include <osg/GLExtensions>
#include <osg/Matrix>
#include <osg/State>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#endif

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_SHADER_STORAGE_BARRIER_BIT
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#endif
#ifndef GLsizeiptr
typedef ptrdiff_t GLsizeiptr;
#endif

namespace app {
namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr int kLocView = 16;
constexpr int kLocProj = 20;

float clamp01(float v) {
    if (v < 0.f) return 0.f;
    if (v > 1.f) return 1.f;
    return v;
}

float sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }

const char* kVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_scale;
layout(location = 2) in vec4 a_rot;
layout(location = 3) in float a_opacity;
layout(location = 4) in vec3 a_color;
layout(location = 16) uniform mat4 u_view;
layout(location = 20) uniform mat4 u_proj;
out vec4 v_mean_opacity;
out vec4 v_scale_pad;
out vec4 v_rot;
out vec4 v_color_pad;
void main() {
    vec4 view_pos = vec4(a_pos, 1.0) * u_view;
    if (view_pos.z >= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    float view_depth = -view_pos.z;
    if (view_depth < 0.1) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec4 clip = view_pos * u_proj;
    if (clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    float iw = 1.0 / clip.w;
    float nx = clip.x * iw;
    float ny = clip.y * iw;
    if (nx < -1.3 || nx > 1.3 || ny < -1.3 || ny > 1.3) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    gl_Position = clip;
    gl_PointSize = 1.0;
    v_mean_opacity = vec4(a_pos, a_opacity);
    v_scale_pad = vec4(a_scale, 0.0);
    v_rot = a_rot;
    v_color_pad = vec4(a_color, 0.0);
}
)GLSL";

const char* kFragSrc = R"GLSL(
#version 430 core
layout(early_fragment_tests) in;
struct GaussianPixel {
    vec4 mean_opacity;
    vec4 scale_pad;
    vec4 rot;
    vec4 color_pad;
    uint frame_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};
layout(std430, binding = 0) buffer ScreenGaussians {
    float meta_view[16];
    float meta_proj[16];
    vec4 cam_pos_tan;
    uvec4 info;
    GaussianPixel cells[];
};
layout(location = 8) uniform int u_width;
layout(location = 9) uniform int u_height;
layout(location = 10) uniform int u_vp_x;
layout(location = 11) uniform int u_vp_y;
layout(location = 12) uniform int u_frame_id;
in vec4 v_mean_opacity;
in vec4 v_scale_pad;
in vec4 v_rot;
in vec4 v_color_pad;
out vec4 fragColor;
void main() {
    int x = int(gl_FragCoord.x) - u_vp_x;
    int y = int(gl_FragCoord.y) - u_vp_y;
    if (x < 0 || y < 0 || x >= u_width || y >= u_height) {
        discard;
    }
    int idx = y * u_width + x;
    cells[idx].mean_opacity = v_mean_opacity;
    cells[idx].scale_pad = v_scale_pad;
    cells[idx].rot = v_rot;
    cells[idx].color_pad = v_color_pad;
    cells[idx].frame_id = uint(u_frame_id);
    fragColor = vec4(v_color_pad.rgb, 1.0);
}
)GLSL";

#ifdef GSPLAT_CUDA_ENABLED
const char* cudaErrStr(cudaError_t err) {
    const char* msg = cudaGetErrorString(err);
    return (msg && msg[0]) ? msg : "cuda runtime unavailable";
}
#endif

unsigned compileShader(osg::GLExtensions* ext, unsigned type, const char* src) {
    if (!ext) return 0;
    const unsigned sh = ext->glCreateShader(type);
    ext->glShaderSource(sh, 1, &src, nullptr);
    ext->glCompileShader(sh);
    int ok = 0;
    ext->glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        ext->glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::cerr << "GL shader compile: " << log << "\n";
        ext->glDeleteShader(sh);
        return 0;
    }
    return sh;
}

void uploadOsgMatrix(osg::GLExtensions* ext, int location, const osg::Matrixd& m) {
    if (!ext || location < 0) return;
    const osg::Matrixf mf(m);
    ext->glUniformMatrix4fv(location, 1, GL_TRUE, mf.ptr());
}

void osgMatrixToGlslRowMajor(const osg::Matrixd& m, float out[16]) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[r * 4 + c] = static_cast<float>(m(r, c));
        }
    }
}

bool extractPerspectiveTans(const osg::Matrixd& proj, float& tan_fovx, float& tan_fovy) {
    const double m00 = proj(0, 0);
    const double m11 = proj(1, 1);
    if (std::abs(m00) < 1e-12 || std::abs(m11) < 1e-12) return false;
    tan_fovx = static_cast<float>(1.0 / std::abs(m00));
    tan_fovy = static_cast<float>(1.0 / std::abs(m11));
    return std::isfinite(tan_fovx) && std::isfinite(tan_fovy) && tan_fovx > 0.f && tan_fovy > 0.f;
}

static bool diagSsboEnabled() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("GSPLAT_DIAG_SSBO");
        cached = (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }
    return cached != 0;
}

static void mulRowMat4(const float* M, float x, float y, float z, float w, float out[4]) {
    const float in[4] = {x, y, z, w};
    for (int c = 0; c < 4; ++c) {
        float s = 0.f;
        for (int r = 0; r < 4; ++r) {
            s += in[r] * M[r * 4 + c];
        }
        out[c] = s;
    }
}

/// 与 GL/CUDA SSBO 路径一致：vec4(pos,1) * view * proj，像素取 round(ndc2Pix)。
static bool worldToPixelGlslRow(const float* view_glsl, const float* proj_glsl, float x, float y, float z,
                                int w, int h, int& px, int& py, float& fx, float& fy) {
    float t[4] = {};
    float c[4] = {};
    mulRowMat4(view_glsl, x, y, z, 1.f, t);
    mulRowMat4(proj_glsl, t[0], t[1], t[2], t[3], c);
    const float cw = c[3];
    if (cw <= 1e-7f) {
        return false;
    }
    const float iw = 1.f / cw;
    const float nx = c[0] * iw;
    const float ny = c[1] * iw;
    fx = ((nx + 1.f) * static_cast<float>(w) - 1.f) * 0.5f;
    fy = ((ny + 1.f) * static_cast<float>(h) - 1.f) * 0.5f;
    px = static_cast<int>(fx + 0.5f);
    py = static_cast<int>(fy + 0.5f);
    return px >= 0 && py >= 0 && px < w && py < h;
}

static float ndc2Pix(float v, int S) { return ((v + 1.f) * static_cast<float>(S) - 1.f) * 0.5f; }

static float percentileInPlace(std::vector<float>& v, float p) {
    if (v.empty()) {
        return 0.f;
    }
    const size_t idx = static_cast<size_t>(p * static_cast<float>(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(idx), v.end());
    return v[idx];
}

struct SsboMapOutlier {
    float l2 = 0.f;
    int cell_x = 0;
    int cell_y = 0;
    int proj_x = 0;
    int proj_y = 0;
    float scale_max = 0.f;
    float opacity = 0.f;
};

static void diagnoseSsboList(const gsplat::ScreenCacheMetaGpu& meta, int render_w, int render_h, int filled,
                             uint32_t expected_frame_id, const gsplat::GaussianPixelGpu* cells) {
    if (!cells || filled <= 0) {
        return;
    }
    int frame_ok = 0;
    int pos_opacity = 0;
    int clip_behind = 0;
    int vp_x = 0;
    int vp_y = 0;
    gsplat::unpackScreenCacheViewport(meta.info[3], vp_x, vp_y);
    const int diag_w = render_w > 0 ? render_w : 1280;
    const int diag_h = render_h > 0 ? render_h : 720;

    for (int i = 0; i < filled; ++i) {
        const gsplat::GaussianPixelGpu& g = cells[i];
        if (g.frame_id != expected_frame_id) {
            continue;
        }
        ++frame_ok;
        if (g.mean_opacity[3] > 1e-4f) {
            ++pos_opacity;
        }
        int px = 0;
        int py = 0;
        float fx = 0.f;
        float fy = 0.f;
        if (!worldToPixelGlslRow(meta.view_glsl, meta.proj_glsl, g.mean_opacity[0], g.mean_opacity[1],
                                 g.mean_opacity[2], diag_w, diag_h, px, py, fx, fy)) {
            ++clip_behind;
        }
    }

    std::cout << "[CACHE-SSBO-GRID] frame=" << expected_frame_id << " cells=" << filled
              << " frame_ok=" << frame_ok << " pos_opacity=" << pos_opacity << " clip_behind=" << clip_behind
              << " grid=" << meta.info[1] << "x" << meta.info[2] << " render=" << diag_w << "x" << diag_h
              << " vp=" << vp_x << "," << vp_y << "\n";
}

void fillSsboMetaCpu(gsplat::ScreenCacheMetaGpu& meta, const osg::Matrixd& view, const osg::Matrixd& proj,
                     int render_w, int render_h, int vp_x, int vp_y, uint32_t frame_id) {
    osgMatrixToGlslRowMajor(view, meta.view_glsl);
    osgMatrixToGlslRowMajor(proj, meta.proj_glsl);
    osg::Matrixd invV;
    if (invV.invert(view)) {
        meta.cam_pos_tan[0] = static_cast<float>(invV(3, 0));
        meta.cam_pos_tan[1] = static_cast<float>(invV(3, 1));
        meta.cam_pos_tan[2] = static_cast<float>(invV(3, 2));
    } else {
        meta.cam_pos_tan[0] = meta.cam_pos_tan[1] = meta.cam_pos_tan[2] = 0.f;
    }
    float tan_fovx = 0.f;
    float tan_fovy = 0.f;
    if (extractPerspectiveTans(proj, tan_fovx, tan_fovy)) {
        meta.cam_pos_tan[3] = tan_fovx;
    } else {
        meta.cam_pos_tan[3] = 0.57735026f;
    }
    meta.info[0] = frame_id;
    meta.info[1] = static_cast<uint32_t>(render_w);
    meta.info[2] = static_cast<uint32_t>(render_h);
    meta.info[3] = gsplat::packScreenCacheViewport(vp_x, vp_y);
}

}  // namespace

GlScreenGaussianCache::GlScreenGaussianCache() = default;

GlScreenGaussianCache::~GlScreenGaussianCache() { shutdown(); }

void GlScreenGaussianCache::destroyCudaInterop() {
#ifdef GSPLAT_CUDA_ENABLED
    if (cuda_registered_ && cuda_resource_) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_resource_));
        cuda_resource_ = nullptr;
        cuda_registered_ = false;
    }
#endif
}

void GlScreenGaussianCache::destroyGl() {
    destroyCudaInterop();
    osg::GLExtensions* ext = osg::GLExtensions::Get(gl_context_id_, false);
    if (ext) {
        if (program_ && ext->glDeleteProgram) ext->glDeleteProgram(program_);
        if (vao_ && ext->glDeleteVertexArrays) ext->glDeleteVertexArrays(1, &vao_);
        if (source_vbo_ && ext->glDeleteBuffers) ext->glDeleteBuffers(1, &source_vbo_);
        if (screen_ssbo_ && ext->glDeleteBuffers) ext->glDeleteBuffers(1, &screen_ssbo_);
    }
    program_ = vao_ = source_vbo_ = screen_ssbo_ = 0;
    program_shader_gen_ = 0;
    source_vbo_ready_ = false;
    gsplat::freeDeviceGaussianBuffers(soa_);
}

void GlScreenGaussianCache::shutdown() { destroyGl(); }

bool GlScreenGaussianCache::compileProgram(osg::GLExtensions* ext) {
    constexpr unsigned kShaderGen = 24u;
    if (program_ && program_shader_gen_ == kShaderGen) return true;
    if (program_ && ext && ext->glDeleteProgram) {
        ext->glDeleteProgram(program_);
        program_ = 0;
    }
    program_shader_gen_ = 0;
    if (!ext || !ext->glCreateShader || !ext->glCreateProgram) return false;
    const unsigned vs = compileShader(ext, GL_VERTEX_SHADER, kVertSrc);
    const unsigned fs = compileShader(ext, GL_FRAGMENT_SHADER, kFragSrc);
    if (!vs || !fs) return false;
    program_ = ext->glCreateProgram();
    ext->glAttachShader(program_, vs);
    ext->glAttachShader(program_, fs);
    if (ext->glBindAttribLocation) {
        ext->glBindAttribLocation(program_, 0, "a_pos");
        ext->glBindAttribLocation(program_, 1, "a_scale");
        ext->glBindAttribLocation(program_, 2, "a_rot");
        ext->glBindAttribLocation(program_, 3, "a_opacity");
        ext->glBindAttribLocation(program_, 4, "a_color");
    }
    if (ext->glBindFragDataLocation) {
        ext->glBindFragDataLocation(program_, 0, "fragColor");
    }
    ext->glLinkProgram(program_);
    ext->glDeleteShader(vs);
    ext->glDeleteShader(fs);
    int ok = 0;
    ext->glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        ext->glGetProgramInfoLog(program_, sizeof(log), nullptr, log);
        std::cerr << "GL program link: " << log << "\n";
        ext->glDeleteProgram(program_);
        program_ = 0;
        return false;
    }
    program_shader_gen_ = kShaderGen;
    return true;
}

bool GlScreenGaussianCache::uploadSource(const std::vector<gsplat::Gaussian>& cloud, bool dc_only) {
    pending_source_ = cloud;
    dc_only_ = dc_only;
    source_count_ = static_cast<int>(cloud.size());
    source_vbo_ready_ = false;
    if (source_vbo_) {
        if (osg::GLExtensions* ext = osg::GLExtensions::Get(gl_context_id_, false)) {
            if (ext->glDeleteBuffers) ext->glDeleteBuffers(1, &source_vbo_);
        }
        source_vbo_ = 0;
    }
    return source_count_ > 0;
}

bool GlScreenGaussianCache::ensureSourceVbo(osg::GLExtensions* ext) {
    if (source_vbo_ready_ && source_vbo_) return true;
    if (!ext || !ext->glGenBuffers || !ext->glBindBuffer || !ext->glBufferData) return false;
    if (source_count_ <= 0 || pending_source_.empty()) return false;

    struct Vertex {
        float px, py, pz;
        float sx, sy, sz;
        float qw, qx, qy, qz;
        float opacity;
        float r, g, b;
    };

    bool quat_wxyz = true;
    {
        int score_wxyz = 0;
        int score_xyzw = 0;
        const int step = std::max(1, source_count_ / 2048);
        for (int i = 0; i < source_count_; i += step) {
            const auto& g = pending_source_[static_cast<size_t>(i)];
            const float a0 = std::abs(g.rot[0]);
            const float a3 = std::abs(g.rot[3]);
            if (a0 >= a3) ++score_wxyz;
            if (a3 >= a0) ++score_xyzw;
        }
        quat_wxyz = score_wxyz >= score_xyzw;
        static bool logged_quat = false;
        if (!logged_quat) {
            logged_quat = true;
            std::cout << "[GL SSBO] quat layout: " << (quat_wxyz ? "wxyz" : "xyzw") << " (score "
                      << score_wxyz << "/" << score_xyzw << ")\n";
        }
    }

    std::vector<Vertex> verts(static_cast<size_t>(source_count_));
    for (int i = 0; i < source_count_; ++i) {
        const gsplat::Gaussian& g = pending_source_[static_cast<size_t>(i)];
        Vertex& v = verts[static_cast<size_t>(i)];
        v.px = g.x;
        v.py = g.y;
        v.pz = g.z;
        v.sx = std::exp(std::clamp(g.scale_log[0], -20.f, 0.f));
        v.sy = std::exp(std::clamp(g.scale_log[1], -20.f, 0.f));
        v.sz = std::exp(std::clamp(g.scale_log[2], -20.f, 0.f));
        v.opacity = clamp01(sigmoid(g.opacity_logit));
        float qw = 1.f, qx = 0.f, qy = 0.f, qz = 0.f;
        if (quat_wxyz) {
            qw = g.rot[0];
            qx = g.rot[1];
            qy = g.rot[2];
            qz = g.rot[3];
        } else {
            qx = g.rot[0];
            qy = g.rot[1];
            qz = g.rot[2];
            qw = g.rot[3];
        }
        const float qlen = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (qlen > 1e-8f) {
            const float inv = 1.f / qlen;
            qw *= inv;
            qx *= inv;
            qy *= inv;
            qz *= inv;
        }
        v.qw = qw;
        v.qx = qx;
        v.qy = qy;
        v.qz = qz;
        (void)dc_only_;
        v.r = clamp01(0.5f + kShC0 * g.sh[0]);
        v.g = clamp01(0.5f + kShC0 * g.sh[1]);
        v.b = clamp01(0.5f + kShC0 * g.sh[2]);
    }

    if (vao_ && ext->glDeleteVertexArrays) {
        ext->glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    ext->glGenBuffers(1, &source_vbo_);
    ext->glBindBuffer(GL_ARRAY_BUFFER, source_vbo_);
    ext->glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)), verts.data(),
                      GL_STATIC_DRAW);
    ext->glBindBuffer(GL_ARRAY_BUFFER, 0);
    source_vbo_ready_ = true;
    pending_source_.clear();
    pending_source_.shrink_to_fit();
    return true;
}

bool GlScreenGaussianCache::ensureCudaInterop() {
#ifdef GSPLAT_CUDA_ENABLED
    if (cuda_registered_ || !screen_ssbo_) return cuda_registered_;
    if (!gsplat::isCudaAvailable()) return false;
    cudaSetDevice(0);
    cudaGraphicsResource* res = nullptr;
    const cudaError_t reg_err =
        cudaGraphicsGLRegisterBuffer(&res, screen_ssbo_, cudaGraphicsRegisterFlagsReadOnly);
    if (reg_err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaErrStr(reg_err) << "\n";
        return false;
    }
    cuda_resource_ = res;
    cuda_registered_ = true;
    return true;
#else
    return false;
#endif
}

bool GlScreenGaussianCache::ensureScreenResources(osg::GLExtensions* ext, int width, int height) {
    if (!ext || width <= 0 || height <= 0) return false;
    const bool size_changed = (width_ != width || height_ != height);
    width_ = width;
    height_ = height;
    if (!compileProgram(ext)) return false;

    const size_t ssbo_bytes = sizeof(gsplat::ScreenCacheMetaGpu) +
                              static_cast<size_t>(width_) * static_cast<size_t>(height_) *
                                  sizeof(gsplat::GaussianPixelGpu);
    if (!screen_ssbo_ || size_changed) {
        destroyCudaInterop();
        if (screen_ssbo_ && ext->glDeleteBuffers) ext->glDeleteBuffers(1, &screen_ssbo_);
        ext->glGenBuffers(1, &screen_ssbo_);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, screen_ssbo_);
        ext->glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(ssbo_bytes), nullptr, GL_DYNAMIC_DRAW);
        ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    return true;
}

void GlScreenGaussianCache::bindSourceAttribs(osg::GLExtensions* ext) {
    const GLsizei stride = static_cast<GLsizei>(sizeof(float) * 14);
    if (!vao_ && ext->glGenVertexArrays && ext->glBindVertexArray) {
        ext->glGenVertexArrays(1, &vao_);
        ext->glBindVertexArray(vao_);
        ext->glBindBuffer(GL_ARRAY_BUFFER, source_vbo_);
        ext->glEnableVertexAttribArray(0);
        ext->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
        ext->glEnableVertexAttribArray(1);
        ext->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 3));
        ext->glEnableVertexAttribArray(2);
        ext->glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 6));
        ext->glEnableVertexAttribArray(3);
        ext->glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 10));
        ext->glEnableVertexAttribArray(4);
        ext->glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 11));
        ext->glBindVertexArray(0);
    }
    if (vao_ && ext->glBindVertexArray) {
        ext->glBindVertexArray(vao_);
        return;
    }
    ext->glBindBuffer(GL_ARRAY_BUFFER, source_vbo_);
    ext->glEnableVertexAttribArray(0);
    ext->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
    ext->glEnableVertexAttribArray(1);
    ext->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 3));
    ext->glEnableVertexAttribArray(2);
    ext->glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 6));
    ext->glEnableVertexAttribArray(3);
    ext->glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 10));
    ext->glEnableVertexAttribArray(4);
    ext->glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(sizeof(float) * 11));
}

gsplat::Status GlScreenGaussianCache::drawPointsAndSsbo(osg::State* state, const osg::Matrixd& view,
                                                        const osg::Matrixd& proj, int width, int height) {
    if (!state || source_count_ <= 0) return gsplat::Status::ErrorNoGaussians;

    ssbo_ready_ = false;
    gl_context_id_ = state->getContextID();
    osg::GLExtensions* ext = osg::GLExtensions::Get(gl_context_id_, true);
    if (!ext) return gsplat::Status::ErrorRenderFailed;
    if (!ensureSourceVbo(ext)) return gsplat::Status::ErrorNoGaussians;
    GLint gl_vp[4] = {0, 0, 0, 0};
    glGetIntegerv(GL_VIEWPORT, gl_vp);
    const int vp_x = gl_vp[0];
    const int vp_y = gl_vp[1];
    if (gl_vp[2] > 0 && gl_vp[3] > 0) {
        width = gl_vp[2];
        height = gl_vp[3];
    }
    if (!ensureScreenResources(ext, width, height)) return gsplat::Status::ErrorRenderFailed;

    const uint32_t frame_id = ++frame_counter_;
    gsplat::ScreenCacheMetaGpu meta{};
    fillSsboMetaCpu(meta, view, proj, width_, height_, vp_x, vp_y, frame_id);
    // 仅上传 meta；未命中格保留旧 frame_id，CUDA unpack 会将其 opacity 置 0（无需每帧清零 ~W×H 格）。
    ext->glBindBuffer(GL_SHADER_STORAGE_BUFFER, screen_ssbo_);
    ext->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(meta)), &meta);
    last_ssbo_meta_ = meta;

    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);

    ext->glUseProgram(program_);
    uploadOsgMatrix(ext, kLocView, view);
    uploadOsgMatrix(ext, kLocProj, proj);

    const int loc_w = ext->glGetUniformLocation(program_, "u_width");
    const int loc_h = ext->glGetUniformLocation(program_, "u_height");
    const int loc_vpx = ext->glGetUniformLocation(program_, "u_vp_x");
    const int loc_vpy = ext->glGetUniformLocation(program_, "u_vp_y");
    const int loc_fid = ext->glGetUniformLocation(program_, "u_frame_id");
    if (loc_w >= 0) ext->glUniform1i(loc_w, width_);
    if (loc_h >= 0) ext->glUniform1i(loc_h, height_);
    if (loc_vpx >= 0) ext->glUniform1i(loc_vpx, vp_x);
    if (loc_vpy >= 0) ext->glUniform1i(loc_vpy, vp_y);
    if (loc_fid >= 0) ext->glUniform1i(loc_fid, static_cast<int>(frame_id));

    if (ext->glBindBufferBase) {
        ext->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, screen_ssbo_);
    }

    bindSourceAttribs(ext);
    for (int a = 5; a < 16; ++a) {
        ext->glDisableVertexAttribArray(static_cast<unsigned>(a));
    }

    glDrawArrays(GL_POINTS, 0, source_count_);

    // SSBO 写入对本 context 后续操作可见；CUDA interop 前在 unpackSsboToDevice 里 glFinish。
    if (ext->glMemoryBarrier) {
        ext->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    if (vao_ && ext->glBindVertexArray) {
        ext->glBindVertexArray(0);
    } else {
        ext->glDisableVertexAttribArray(0);
        ext->glDisableVertexAttribArray(1);
        ext->glDisableVertexAttribArray(2);
        ext->glDisableVertexAttribArray(3);
        ext->glDisableVertexAttribArray(4);
        ext->glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    ext->glUseProgram(0);
    glDisable(GL_DEPTH_TEST);

    filled_count_ = width_ * height_;

    ssbo_ready_ = true;
    last_ssbo_frame_id_ = frame_id;
    return gsplat::Status::Ok;
}

gsplat::Status GlScreenGaussianCache::unpackSsboToDevice() {
    if (!ssbo_ready_ || width_ <= 0 || height_ <= 0) return gsplat::Status::ErrorRenderFailed;
#ifndef GSPLAT_CUDA_ENABLED
    return gsplat::Status::ErrorNoCuda;
#else
    if (!gsplat::isCudaAvailable()) return gsplat::Status::ErrorNoCuda;
    if (!ensureCudaInterop()) return gsplat::Status::ErrorRenderFailed;
    cudaSetDevice(0);
    cudaGraphicsResource* res = static_cast<cudaGraphicsResource*>(cuda_resource_);
    if (!res) return gsplat::Status::ErrorRenderFailed;
    glFinish();
    if (cudaGraphicsMapResources(1, &res, 0) != cudaSuccess) return gsplat::Status::ErrorRenderFailed;
    size_t mapped_bytes = 0;
    void* d_raw = nullptr;
    if (cudaGraphicsResourceGetMappedPointer(&d_raw, &mapped_bytes, res) != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &res, 0);
        return gsplat::Status::ErrorRenderFailed;
    }
    if (mapped_bytes < sizeof(gsplat::ScreenCacheMetaGpu)) {
        cudaGraphicsUnmapResources(1, &res, 0);
        return gsplat::Status::ErrorRenderFailed;
    }
    const auto* d_meta = reinterpret_cast<const gsplat::ScreenCacheMetaGpu*>(d_raw);
    const auto* d_ptr =
        reinterpret_cast<const gsplat::GaussianPixelGpu*>(reinterpret_cast<const uint8_t*>(d_raw) +
                                                          sizeof(gsplat::ScreenCacheMetaGpu));
    gsplat::ScreenCacheMetaGpu h_meta = {};
    cudaMemcpy(&h_meta, d_meta, sizeof(h_meta), cudaMemcpyDeviceToHost);
    if (h_meta.info[0] == last_ssbo_frame_id_) {
        last_ssbo_meta_ = h_meta;
        gsplat::buildCameraFromSsboMeta(h_meta, ssbo_cam_);
        has_ssbo_cam_ = true;
    } else {
        has_ssbo_cam_ = false;
    }

    if (diagSsboEnabled() && has_ssbo_cam_) {
        static int diag_frame = 0;
        if ((diag_frame++ % 30) == 0) {
            const int n = width_ * height_;
            std::vector<gsplat::GaussianPixelGpu> h_cells(static_cast<size_t>(n));
            if (cudaMemcpy(h_cells.data(), d_ptr, static_cast<size_t>(n) * sizeof(gsplat::GaussianPixelGpu),
                           cudaMemcpyDeviceToHost) == cudaSuccess) {
                diagnoseSsboList(h_meta, width_, height_, n, last_ssbo_frame_id_, h_cells.data());
            }
        }
    }

    const gsplat::Status st = gsplat::unpackScreenSsboToDevice(d_ptr, width_, height_, soa_, filled_count_,
                                                               last_ssbo_frame_id_);
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &res, 0);
    return st;
#endif
}

}  // namespace app
