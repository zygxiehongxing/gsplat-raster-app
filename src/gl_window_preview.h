#pragma once

/**
 * CUDA 光栅输出 → GL 纹理（GPU 直写），PostDraw 全屏绘制到窗口，无 CPU 回读。
 */

#include <gsplat_raster/gsplat_raster.h>

namespace osg {
class State;
class GLExtensions;
}

namespace app {

class GlWindowPreview {
public:
    GlWindowPreview();
    ~GlWindowPreview();

    GlWindowPreview(const GlWindowPreview&) = delete;
    GlWindowPreview& operator=(const GlWindowPreview&) = delete;

    /// 按视口尺寸创建/调整 GL 纹理并注册 CUDA interop。
    bool ensure(osg::State* state, int width, int height);

    /// 将 SDK 光栅 planar float RGB（CHW）写入预览纹理。
    bool uploadPlanarRgb(const float* d_planar_rgb, int width, int height);

    /// 将预览纹理全屏绘制到当前默认 framebuffer（须在 PostDraw / 有效 GL 上下文中调用）。
    bool drawFullscreen(osg::State* state, int viewport_w, int viewport_h);

    int width() const { return width_; }
    int height() const { return height_; }
    bool ready() const { return texture_ready_; }

    void shutdown();

private:
    bool ensureDrawProgram(osg::State* state);
    void destroyGl();
    void destroyCuda();
    void destroyDrawProgram(osg::GLExtensions* ext);

    int width_ = 0;
    int height_ = 0;
    unsigned int gl_tex_ = 0;
    unsigned int gl_context_id_ = 0;
    unsigned int program_ = 0;
    unsigned int vao_ = 0;
    int program_shader_gen_ = 0;
    bool texture_ready_ = false;
#ifdef GSPLAT_CUDA_ENABLED
    void* cuda_resource_ = nullptr;
#endif
};

}  // namespace app
