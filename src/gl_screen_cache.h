#pragma once

/**
 * GL 视口栅格 SSBO（W×H，深度测试赢家写格）→ CUDA interop unpack → SDK EWA 光栅。
 */

#include <gsplat_raster/gsplat_raster.h>

#include <vector>

namespace osg {
class State;
class GLExtensions;
class Matrixd;
}

namespace app {

class GlScreenGaussianCache {
public:
    GlScreenGaussianCache();
    ~GlScreenGaussianCache();

    GlScreenGaussianCache(const GlScreenGaussianCache&) = delete;
    GlScreenGaussianCache& operator=(const GlScreenGaussianCache&) = delete;

    bool uploadSource(const std::vector<gsplat::Gaussian>& gaussians, bool dc_only = true);
    int sourceCount() const { return source_count_; }

    /// 主相机绘制：点云经深度测试，片元赢家写入 W×H SSBO 栅格。
    gsplat::Status drawPointsAndSsbo(osg::State* state, const osg::Matrixd& view, const osg::Matrixd& proj,
                                   int width, int height);

    /// PostDraw：glFinish 后 map SSBO 栅格并 unpack 到 CUDA SoA。
    gsplat::Status unpackSsboToDevice();

    bool ssboReady() const { return ssbo_ready_; }
    const gsplat::DeviceGaussianBuffers& deviceBuffers() const { return soa_; }
    int filledCount() const { return filled_count_; }
    bool hasSsboCamera() const { return has_ssbo_cam_; }
    const gsplat::Camera& ssboCamera() const { return ssbo_cam_; }
    int width() const { return width_; }
    int height() const { return height_; }

    void shutdown();

private:
    bool ensureSourceVbo(osg::GLExtensions* ext);
    bool ensureScreenResources(osg::GLExtensions* ext, int width, int height);
    bool ensureCudaInterop();
    bool compileProgram(osg::GLExtensions* ext);
    void bindSourceAttribs(osg::GLExtensions* ext);
    void destroyGl();
    void destroyCudaInterop();

    std::vector<gsplat::Gaussian> pending_source_;
    bool dc_only_ = true;
    bool source_vbo_ready_ = false;

    unsigned int source_vbo_ = 0;
    unsigned int screen_ssbo_ = 0;
    unsigned int program_ = 0;
    unsigned int vao_ = 0;
    unsigned int program_shader_gen_ = 0;
    int source_count_ = 0;
    int width_ = 0;
    int height_ = 0;
    int filled_count_ = 0;
    uint32_t frame_counter_ = 0;
    uint32_t last_ssbo_frame_id_ = 0;
    bool ssbo_ready_ = false;
    bool cuda_registered_ = false;
#ifdef GSPLAT_CUDA_ENABLED
    void* cuda_resource_ = nullptr;
#endif
    gsplat::DeviceGaussianBuffers soa_{};
    unsigned int gl_context_id_ = 0;
    bool has_ssbo_cam_ = false;
    gsplat::Camera ssbo_cam_{};
    gsplat::ScreenCacheMetaGpu last_ssbo_meta_{};
};

}  // namespace app
