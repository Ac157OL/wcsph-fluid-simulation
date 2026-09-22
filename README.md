# WCSPH Fluid Simulation

基于 NVIDIA Warp 的三维弱可压缩 SPH（WCSPH）自由表面流体模拟。支持球体与立方体初始条件、CPU/CUDA 计算、粒子动画、PLY 点云导出、表面重建与 Blender 水体渲染。

## 快速开始

建议使用 Python 3.10 或更新版本。Blender 为单独安装的可选工具，渲染脚本面向 Blender 4.x。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 短程运行，检查模拟、帧图与点云输出
python main.py --scene sphere --steps 300 --fps 5 --diagnostics --output-dir results

# 使用随仓库附带的点云测试表面重建，无需先运行长时模拟
python reconstruct_surface.py examples/sphere-impact.ply --output-dir results/reconstruction
```

程序优先选择 CUDA，不可用时使用 CPU。短程运行只用于验证执行流程，不能展示完整的落下与铺展过程。

## 完整流程

```bash
# 20 秒物理时间，采用 Morris 运动黏度与趋近粒子对人工黏性
python main.py --scene sphere --seconds 20 --fps 5 \
  --viscosity 0.000001 --artificial-viscosity-alpha 0.1 \
  --motion-diagnostics --output-dir results

python build_animation.py --frames-dir results/frames \
  --output results/wcsph-animation.gif --fps 5

python reconstruct_surface.py results/fluid_final.ply \
  --output-dir results/reconstruction

blender -b --python blender_scene.py -- \
  results/reconstruction/fluid_surface.ply results/blender results/fluid_final.ply
```

时间步约为 `8.53e-5` 秒，20 秒模拟需要约 23 万步，CPU 上可能耗时较长。`--live` 打开实时窗口；`--diagnostics` 逐步记录密度诊断并增加同步开销；`--usd` 需要额外的 OpenUSD 支持。每次运行建议使用新的输出目录，避免混入上次运行的帧。

## 目录

```text
main.py                    模拟入口、时间推进和诊断输出
warpSph/                   粒子构造、核函数和 WCSPH 求解器
visualization.py           实时显示、PNG 与 PLY 输出
build_animation.py         PNG 帧合成 GIF
reconstruct_surface.py     点云隐式场与 Marching Cubes 重建
blender_scene.py           Blender 场景与水体渲染
examples/sphere-impact.ply  784 个模拟粒子的 ASCII 示例点云
requirements.txt           Python 依赖
```

示例点云包含球形流体撞击后的粒子位置，用于演示表面重建流程。算法使用三次样条核、Tait 状态方程、密度下限、压力与黏性力以及半隐式 Euler 积分。文献依据为 Becker 与 Teschner 的 *Weakly Compressible SPH for Free Surface Flows*（SCA 2007）。
