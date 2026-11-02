# PI-NGreyKAN 敏感性分析（独立项目）

针对参数：热输入衰减尺度 ell 和熔深损失权重 alpha。

固定四条焊缝 W01-W04，顺序保留、最后20%测试；沿用主模型共享KAN (5→20→2)、7样条节点、独立双目标灰色状态、热输入局部扰动稳定性约束 λ=2。

生成两幅图，每幅两个子图：(a)训练拟合MAE三维曲面；(b)测试MAE相对默认 (ell=0.5,alpha=0.6) 的热力图。熔深D、熔宽W单独统计。

## 安装依赖
python -m pip install numpy pandas matplotlib torch

## 运行
在这个文件夹下：

- `python run_sensitivity.py --mode quick`：**预览** 7×7×1随机种子、每配置仅8轮训练；输出结果文件后每一步都自动保存，生成含 QUICK_PREVIEW 的两幅图片。该结果不宜直接用于论文性能结论。
- `python run_sensitivity.py --mode full`：**正式实验** 7×7×3随机种子，前80%训练段再划分内部验证，最多125轮自适应选训练轮次，随后用完整训练段训练；此过程可能需要较长时间，中间结果逐条写入CSV，可以重启断点续算。
- `python run_sensitivity.py --mode plot`：仅对已有**正式实验CSV**绘图。
- `python run_sensitivity.py --mode plot --quick-plot`：只重画预览图。

## 产出
`results/sensitivity_metrics_by_target.csv`：正式实验逐随机种子、逐参数组合的熔深和熔宽训练/测试误差。
`results/sensitivity_metrics_mean.csv`：按种子平均后的49行结果。
`figures/Sensitivity_Penetration_Depth_300dpi.png/.pdf`
`figures/Sensitivity_Weld_Width_300dpi.png/.pdf`

## 指标口径
按案例样本量合并所有训练或测试点后，分别计算D与W的MAE（不是两目标平均，不是四案例等权平均），再平均3随机种子。训练拟合MAE取最终训练阶段模型（与主模型拟合逻辑对应）；测试集不参与训练。

注意：7×7网格预先固定，正式论文不得使用测试热力图再挑选最优参数并回报同一测试集为无偏估计。
