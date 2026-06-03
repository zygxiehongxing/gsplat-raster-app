# gsplat-raster-app

**App 层**：负责 PLY 加载、OSG 场景展示与鼠标/键盘交互；在需要出图时，把**当前帧相机**和**高斯点云数据**交给 SDK，由 SDK 光栅化生成图片。本仓库不实现 CUDA 光栅化。

配套 SDK：`../gsplat-raster-sdk`（独立仓库，无 OSG 依赖）。

## 职责划分

| 层 | 仓库 | 做什么 |
|----|------|--------|
| App | `gsplat-raster-app` | 读 PLY → OSG 点云预览 → 交互时采集相机 + 已加载的 `Gaussian` → 调用 SDK |
| SDK | `gsplat-raster-sdk` | 接收 `Camera` + 高斯数组 → Inria CUDA 光栅化 → RGB / PNG |

```text
  [PLY] ──load──► App (OSG 可视化 + 交互)
                      │
                      │ 当前帧: Camera + Gaussians (+ 分辨率/设置)
                      ▼
                   SDK.render / renderToPng
                      │
                      ▼
                   [PNG / RGB buffer]
```

## 程序

- `gsplat_render_ply`：命令行加载 PLY，用 SDK 直接渲染 PNG（无 OSG）
- `gsplat_osg_app`（可选）：OSG 点云预览 + **SDK 实时小窗**（松开鼠标后更新）；`R` 真值对齐截图，`T` 仅导出真值，`P` 开关预览

## Dependency

- SDK project in sibling folder: `../gsplat-raster-sdk`
- Optional OSG from vcpkg (`osg`) for `gsplat_osg_app`

## Build (Windows)

```cmd
cd gsplat-raster-app
scripts\build.cmd
```

## CLI usage

```cmd
build\gsplat_render_ply scene.ply out.png 1280 720 500000
```

Optional training-camera JSON override:

```cmd
build\gsplat_render_ply scene.ply out.png 1280 720 500000 --camera camera.json
```

## OSG app usage

```cmd
build\gsplat_osg_app scene.ply 200000
REM 也支持反序: gsplat_osg_app 200000 scene.ply
```

- 鼠标漫游：OSG 点云预览（左下角小窗为 SDK 光栅化结果）
- **松开鼠标**后：用当前 OSG `view/proj` 矩阵调用 SDK 更新预览（拖动中不重复渲染，避免卡顿）
- `P`：开关 SDK 实时预览
- `R`：真值对齐截图（同帧写出 `camera_truth_N.json`、`capture_N.png`、`truth_cli_N.png`）
- `T`：只写 `camera_truth_N.json` + 进程内 `truth_cli_N.png` 复现（不写 `capture_*.png`）
- `[` / `]` / `0`：调节 `scale_modifier`

### 真值对齐（ground truth）

按 `R` 或 `T` 后，同帧会保存：

| 文件 | 含义 |
|------|------|
| `camera_truth_N.json` | OSG `view/proj` + 送入 CUDA 的 SDK `view/proj/cam_pos/tan_fov` + `mat_mode`、渲染设置、`visible` |
| `capture_N.png` | App 内 `renderToPng`（仅 `R`） |
| `truth_cli_N.png` | 从 JSON 重载相机后在 App 内再渲一次（应与 `capture` 一致） |

离线用**同一 PLY、同一 max_points** 复现：

```cmd
build\gsplat_render_ply scene.ply truth_offline.png 1280 720 1200000 --truth camera_truth_0.json
```

控制台会打印 `sdk_view` vs `rebuilt_view` 矩阵最大差；若 `capture` 与 `truth_cli` 一致但和 OSG 画面仍不同，问题在矩阵 packing 或高斯解释，而非抓帧时机。

相机矩阵走 `buildCameraFromOsg()`（Inria 约定：`view` + `proj=V×P_inria`，无多模式探测）。
