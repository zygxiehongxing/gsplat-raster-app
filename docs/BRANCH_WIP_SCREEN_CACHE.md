# WIP 分支说明：`wip/ssbo-device-pool-pipeline`

本分支保存 **2026-06** 当前工作区快照，用于与 git 好版（App `9400f39` + SDK `c9d587e`）对比和后续回退/重构。

## 相对 `9400f39`（效果较好的基线）的主要变化

### 渲染管线（影响观感最大）

| 基线 `9400f39` | 本分支 |
|----------------|--------|
| `raster.setGaussians(全量 PLY)` | **无** `setGaussians` |
| `raster.render(cam, w, h, rgb)` 全量 EWA | GL **SSBO 屏缓存** → `unpackSsboToDevice` → `render(cam, w, h, screen_buf, rgb)` |
| OSG `buildPointNode` 点云导航 | `ScreenSsboDrawDrawable`（1px 点 + 片元写 cell） |
| `buildCameraFromSnapshot` | 优先 `gl_screen_cache.ssboCamera()` |
| 交互 `scale_modifier` 默认 0.03，上限 0.5 | 默认 0.02，上限 2.0，`0` 重置 0.015 |

### 新增文件

- `src/gl_screen_cache.cpp` / `.h` — GL VBO + SSBO + CUDA-GL interop 解包
- `scripts/*.bat` — Windows 配置/构建/运行脚本
- `cmake/copy_runtime_dlls.cmake` — 运行时 DLL 拷贝

### `render_ply_main`

已改为 **GaussianDevicePool + filterVisible + render(subset)**，与 OSG App 的 SSBO 主路径不同。

## 已知问题

1. **SSBO 主路径画质差**：每像素仅保留一个高斯，再按完整 EWA splat，易出现大块、过密、与 OSG 点预览不一致。
2. **诊断** `GSPLAT_DIAG_SSBO=1` 时几何对齐良好（`le1px≈100%`），说明矩阵问题不大，主因是渲染语义。
3. **中文 IME**：Win32 已 `ImmAssociateContext(nullptr)` + 异常键过滤，避免 OSG 卡死。

## 环境变量（SSBO 路径）

| 变量 | 默认 | 含义 |
|------|------|------|
| `GSPLAT_SCREEN_SCALE_MUL` | 0.035 | SSBO 解包时再缩小 scale |
| `GSPLAT_SCREEN_MAX_OPACITY` | 0.65 | 解包时钳位 opacity |
| `GSPLAT_DIAG_SSBO` | 0 | 打印 `[CACHE-SSBO-MAP]` 对齐审计 |

## 构建与运行

```cmd
scripts\configure_and_build.bat
scripts\run_osg_app.bat scene.ply 200000
```

## 后续计划（建议）

1. OSG App 主路径改回 **GaussianDevicePool + filterVisible + render(subset)**（与 `render_ply_main` 一致）。
2. SSBO / `gl_screen_cache` 保留为 `#ifdef` 或诊断开关，不作为默认渲染。
3. 流式八叉树：八叉树负责 LOD/IO，GPU 池负责每帧活跃 splat 子集（见对话设计）。

## 对比好版

```cmd
git diff 9400f39 -- src/osg_viewer_main.cpp
```
