#include "gl_window_preview.h"

#include <osg/GL>
#include <osg/GLExtensions>
#include <osg/State>

#include <iostream>

#ifdef GSPLAT_CUDA_ENABLED
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

namespace app {
bool copyPlanarRgbToCudaArray(cudaArray_t array, const float* d_planar_rgb, int width, int height);
}
#endif

#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif
#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_TEXTURE0
#define GL_TEXTURE0 0x84C0
#endif
#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#endif
#ifndef GL_VERTEX_ARRAY_BINDING
#define GL_VERTEX_ARRAY_BINDING 0x85B5
#endif

namespace app {
namespace {

const char* kVertSrc = R"GLSL(
#version 430 core
layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
out vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
)GLSL";

const char* kFragSrc = R"GLSL(
#version 430 core
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 fragColor;
void main() {
    fragColor = texture(u_tex, v_uv);
}
)GLSL";

unsigned int compileShader(osg::GLExtensions* ext, unsigned int type, const char* src) {
    if (!ext || !ext->glCreateShader || !ext->glShaderSource || !ext->glCompileShader ||
        !ext->glGetShaderiv || !ext->glGetShaderInfoLog) {
        return 0;
    }
    const unsigned int sh = ext->glCreateShader(type);
    const char* csrc = src;
    ext->glShaderSource(sh, 1, &csrc, nullptr);
    ext->glCompileShader(sh);
    int ok = 0;
    ext->glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = {};
        ext->glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        std::cerr << "[GPU preview] shader compile failed: " << log << "\n";
        ext->glDeleteShader(sh);
        return 0;
    }
    return sh;
}

unsigned int linkProgram(osg::GLExtensions* ext, unsigned int vs, unsigned int fs) {
    if (!ext || !ext->glCreateProgram || !ext->glAttachShader || !ext->glLinkProgram ||
        !ext->glGetProgramiv || !ext->glGetProgramInfoLog) {
        return 0;
    }
    const unsigned int prog = ext->glCreateProgram();
    ext->glAttachShader(prog, vs);
    ext->glAttachShader(prog, fs);
    ext->glLinkProgram(prog);
    int ok = 0;
    ext->glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024] = {};
        ext->glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::cerr << "[GPU preview] program link failed: " << log << "\n";
        ext->glDeleteProgram(prog);
        return 0;
    }
    return prog;
}

}  // namespace

GlWindowPreview::GlWindowPreview() = default;
GlWindowPreview::~GlWindowPreview() { shutdown(); }

void GlWindowPreview::destroyCuda() {
#ifdef GSPLAT_CUDA_ENABLED
    if (cuda_resource_) {
        cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource*>(cuda_resource_));
        cuda_resource_ = nullptr;
    }
#endif
}

void GlWindowPreview::destroyDrawProgram(osg::GLExtensions* ext) {
    if (ext && ext->glDeleteVertexArrays && vao_ != 0) {
        ext->glDeleteVertexArrays(1, &vao_);
    }
    if (ext && ext->glDeleteProgram && program_ != 0) {
        ext->glDeleteProgram(program_);
    }
    vao_ = 0;
    program_ = 0;
    program_shader_gen_ = 0;
}

void GlWindowPreview::destroyGl() {
    destroyCuda();
    destroyDrawProgram(nullptr);
    if (gl_tex_ != 0) {
        glDeleteTextures(1, &gl_tex_);
        gl_tex_ = 0;
    }
    texture_ready_ = false;
    width_ = 0;
    height_ = 0;
}

void GlWindowPreview::shutdown() { destroyGl(); }

bool GlWindowPreview::ensureDrawProgram(osg::State* state) {
    if (!state) return false;
    osg::GLExtensions* ext = state->get<osg::GLExtensions>();
    if (!ext) return false;
    if (program_ != 0 && vao_ != 0) return true;

    const unsigned int vs = compileShader(ext, GL_VERTEX_SHADER, kVertSrc);
    const unsigned int fs = compileShader(ext, GL_FRAGMENT_SHADER, kFragSrc);
    if (!vs || !fs) return false;
    program_ = linkProgram(ext, vs, fs);
    ext->glDeleteShader(vs);
    ext->glDeleteShader(fs);
    if (!program_) return false;

    if (!ext->glGenVertexArrays || !ext->glBindVertexArray || !ext->glGenBuffers || !ext->glBindBuffer ||
        !ext->glBufferData || !ext->glEnableVertexAttribArray || !ext->glVertexAttribPointer) {
        return false;
    }
    // 全屏三角形：NDC 覆盖 + UV
    const float verts[] = {
        -1.f, -1.f, 0.f, 0.f,  //
        3.f,  -1.f, 2.f, 0.f,  //
        -1.f, 3.f,  0.f, 2.f,  //
    };
    unsigned int vbo = 0;
    ext->glGenVertexArrays(1, &vao_);
    ext->glGenBuffers(1, &vbo);
    ext->glBindVertexArray(vao_);
    ext->glBindBuffer(GL_ARRAY_BUFFER, vbo);
    ext->glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    ext->glEnableVertexAttribArray(0);
    ext->glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    ext->glEnableVertexAttribArray(1);
    ext->glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                               reinterpret_cast<void*>(2 * sizeof(float)));
    ext->glBindVertexArray(0);
    ext->glDeleteBuffers(1, &vbo);
    return true;
}

bool GlWindowPreview::ensure(osg::State* state, int width, int height) {
    if (!state || width <= 0 || height <= 0) return false;
#ifndef GSPLAT_CUDA_ENABLED
    (void)width;
    (void)height;
    return false;
#else
    if (!gsplat::isCudaAvailable()) return false;
    const unsigned int ctx = state->getContextID();
    if (gl_context_id_ != 0 && gl_context_id_ != ctx) {
        shutdown();
    }
    gl_context_id_ = ctx;

    if (texture_ready_ && width_ == width && height_ == height && gl_tex_ != 0) {
        return ensureDrawProgram(state);
    }

    if (osg::GLExtensions* ext = state->get<osg::GLExtensions>()) {
        destroyDrawProgram(ext);
    } else {
        destroyDrawProgram(nullptr);
    }
    destroyCuda();
    if (gl_tex_ != 0) {
        glDeleteTextures(1, &gl_tex_);
        gl_tex_ = 0;
    }
    texture_ready_ = false;
    width_ = width;
    height_ = height;

    glGenTextures(1, &gl_tex_);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaSetDevice(0);
    cudaGraphicsResource* res = nullptr;
    const cudaError_t reg_err =
        cudaGraphicsGLRegisterImage(&res, gl_tex_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (reg_err != cudaSuccess) {
        std::cerr << "[GPU preview] cudaGraphicsGLRegisterImage failed: " << cudaGetErrorString(reg_err)
                  << "\n";
        destroyGl();
        return false;
    }
    cuda_resource_ = res;
    texture_ready_ = true;
    return ensureDrawProgram(state);
#endif
}

bool GlWindowPreview::uploadPlanarRgb(const float* d_planar_rgb, int width, int height) {
#ifndef GSPLAT_CUDA_ENABLED
    (void)d_planar_rgb;
    (void)width;
    (void)height;
    return false;
#else
    if (!texture_ready_ || !cuda_resource_ || !d_planar_rgb || width != width_ || height != height_) {
        return false;
    }
    cudaSetDevice(0);
    cudaGraphicsResource* res = static_cast<cudaGraphicsResource*>(cuda_resource_);
    if (cudaGraphicsMapResources(1, &res, 0) != cudaSuccess) {
        std::cerr << "[GPU preview] cudaGraphicsMapResources failed: " << cudaGetErrorString(cudaGetLastError())
                  << "\n";
        return false;
    }

    cudaArray_t array = nullptr;
    bool ok = false;
    if (cudaGraphicsSubResourceGetMappedArray(&array, res, 0, 0) == cudaSuccess && array) {
        ok = copyPlanarRgbToCudaArray(array, d_planar_rgb, width_, height_);
        if (!ok) {
            std::cerr << "[GPU preview] copyPlanarRgbToCudaArray failed: "
                      << cudaGetErrorString(cudaGetLastError()) << "\n";
        }
        cudaDeviceSynchronize();
    }
    cudaGraphicsUnmapResources(1, &res, 0);
    return ok;
#endif
}

bool GlWindowPreview::drawFullscreen(osg::State* state, int viewport_w, int viewport_h) {
    if (!state || !texture_ready_ || gl_tex_ == 0 || viewport_w <= 0 || viewport_h <= 0) return false;
    if (!ensureDrawProgram(state)) return false;

    osg::GLExtensions* ext = state->get<osg::GLExtensions>();
    if (!ext || !ext->glUseProgram || !ext->glBindVertexArray || !ext->glActiveTexture ||
        !ext->glGetUniformLocation) {
        return false;
    }

    GLint prev_prog = 0;
    GLint prev_vao = 0;
    GLint prev_tex = 0;
    GLint prev_vpt[4] = {};
    GLboolean depth_on = glIsEnabled(GL_DEPTH_TEST);
    glGetIntegerv(GL_VIEWPORT, prev_vpt);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex);

    glViewport(0, 0, viewport_w, viewport_h);
    glDisable(GL_DEPTH_TEST);
    ext->glUseProgram(program_);
    ext->glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, gl_tex_);
    const int loc = ext->glGetUniformLocation(program_, "u_tex");
    if (loc >= 0) {
        ext->glUniform1i(loc, 0);
    }
    ext->glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    ext->glBindVertexArray(static_cast<GLuint>(prev_vao));

    ext->glUseProgram(static_cast<GLuint>(prev_prog));
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(prev_tex));
    glViewport(prev_vpt[0], prev_vpt[1], prev_vpt[2], prev_vpt[3]);
    if (depth_on) {
        glEnable(GL_DEPTH_TEST);
    }
    return true;
}

}  // namespace app
