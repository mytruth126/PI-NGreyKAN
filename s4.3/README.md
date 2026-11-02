# 双目标 PI-NGreyKAN：选定模型 + 19种对比模型 + 四组绘图

本项目用于复现**2026-10-30选定的模型**，不是后续重新选出的网络：
- 结构：**5→20→2**，7个样条节点，双目标同步预测熔深 D 与熔宽 W；
- 状态：独立的自适应系数 `a` 和非线性项 `b*v*abs(v)`；
- KAN输入：4个工艺信号 + 等效电弧作用代理 `I*U/1000*exp(-0.5*(E/0.5)^2)`；
- 物理损失：**局部电流/电压 ±1% 扰动一致性**，容差 D=0.008mm、W=0.012mm、任务权重0.35/0.65，总物理权重 **2.0**；
- 基准：无物理约束版本可通过 `python code/run_simple_physics.py` 复现，8项物理约束的完整选型脚本一并保留。

> 这项物理约束是一条经验性的局部稳健软先验，非已经验证的热传导定律。它相对于无物理约束的误差增益很小，不能宣称显著提升。

## 一、数据

`raw_data/` 保存原始焊缝01—04 CSV。

`data/` 保存之前已固定的四组删点数据（**不是重新抽样**）：

|焊缝|原始行数|固定保留数|前80%训练|后20%测试|测试非6mm比例|
|---|---:|---:|---:|---:|---:|
|W01|500|212|169|43|7/43|
|W02|250|121|96|25|10/25|
|W03|250|89|71|18|6/18|
|W04|301|106|84|22|8/22|

- `data/fixed_retained_indices.csv` 固化原始行号；`code/recreate_data.py` 可用它从原始CSV严格重建这四组，**不改变样本顺序**。不是新随机抽样。
- 前80%训练；其中前64%训练、后16%作为内部验证（此处指各原始数据长度的64%和16%，对应 `int(int(0.8*n)*0.8)` 的取整实现）。
- 最后的20%真实质量标签**只用于误差评价**，不参与拟合、选参、校准、递推输入或物理先验。
- `original_index` 保留原始行号，绘图横轴是删点后测试样本编号。按熔深真实值删点会改变数据分布，不能将测试误差等同于原始连续焊缝泛化误差。
- **重要建模边界**：为了忠实复现原选定实验，主模型的灰色状态递推仍以“删点后的每行”为一个离散时间步，而非按缺失的原始时间间隔计算 `dt`。因此其结果不宜直接解释为严格的0.1秒等间隔在线预测；后续研究应增加不等时间间隔控制实验。

## 二、安装 Python

建议 Python 3.9/3.10/3.11，和 PyTorch CPU 或 CUDA 官方匹配版本。先安装 PyTorch，再安装其余依赖（可用提供的 `requirements-py39.txt`）。

```bat
python -m pip install -r requirements-py39.txt
```

如果没有 CatBoost、XGBoost、statsmodels，运行主模型和 `--step plot` 仍可行，但运行包含相应基线的步骤会需要对应依赖。

## 三、一键执行（项目根目录）

|目的|命令|
|---|---|
|只核对/重建固定数据|`python run_pipeline.py --step prepare`|
|只重新训练选定的PI-NGreyKAN|`python run_pipeline.py --step selected`|
|跑默认7个主要对比模型|`python run_pipeline.py --step baselines`|
|运行全部19种对比模型|`python run_pipeline.py --step baselines --models ALL`|
|只重新画图，不重新训练|`python run_pipeline.py --step plot`|
|主模型 + 默认7个对比 + 绘图|`python run_pipeline.py --step all`|

在Spyder中打开 `run_pipeline.py`，设置Working Directory为解压后的项目根目录，传入 `--step plot` 或 `--step selected` 等参数。Windows Powershell或CMD下模型名称含括号时建议使用双引号。例如：

```bat
python run_pipeline.py --step baselines --models LSTM-Lite GRU Holt "GM(1,N)" "Lag-GM(1,N)" SVR XGBoost
```

**默认运行不重复搜索8项物理候选。** `train_selected.py`只重新运行已经选出的 `ThermalRobust_2.0`，对每个随机种子从内部验证集确定训练轮数，再使用前80%的数据重新训练。

## 四、文件说明

```
run_pipeline.py                 # 最简统一运行入口（Python3.9 / Spyder）
README.md                       # 本说明
requirements.txt               # 通用环境依赖
requirements-py39.txt          # Python3.9安装建议
code/
  base_model.py                # 选定模型的灰色KAN核心（原实现，不改动）
  run_simple_physics.py        # 原8种物理约束筛选实验的源码（原实现）
  train_selected.py            # 选定5-20-2+ThermalRobust_2.0复现
  train_baselines.py           # 19种常见基线的训练、调参、预测
  plot_predictions.py          # 8子图总览、4张单独图、平均预测值及6指标
  recreate_data.py             # 固定原始行号重建数据
raw_data/                        # 原始W01~W04 CSV
 data/                           # 固定四组评价数据、行号、分组统计
 results/
   selected_test_predictions.csv # 最新选定主模型完整3种子预测（108测试点/任务）
   baseline_test_predictions.csv # 全部19种基线的测试逐点预测
   baseline_hyperparameters.csv  # 每种基线验证选参
   selected_training_log.csv    # 主模型验证轮数
   all_model_metrics.csv        # 20模型、2目标、MAE/RMSE/MAPE
   test_predictions.csv         # 历史8方案实验的参考数据（不被默认脚本覆写）
   validation_trials.csv        # 历史内部验证记录
   selected_physics.json        # 历史最佳物理配置
 figures/
   Four_welds_actual_vs_PI_selected_300dpi.png
   Four_welds_actual_vs_comparison_300dpi.png
   W01_actual_vs_models_300dpi.png ... W04...
```

`--step plot` 只读取 `selected_test_predictions.csv` + `baseline_test_predictions.csv`，对重复随机种子预测做平均，严格校验同一个测试点的真实值一致；不重新拟合、不调整预测值或平滑曲线。

绘制指定模型：

```bat
python code/plot_predictions.py --models PI-NGreyKAN LSTM-Lite Holt GRU "GM(1,N)" "Lag-GM(1,N)"
```

## 五、基线说明与公平性边界

`code/train_baselines.py`包含（共19种）：
- 时序神经网络：LSTM-Lite、LSTM、GRU（只使用当前及过去的工艺传感器输入）；
- 灰色：GM(1,N)、Lag-GM(1,N)、DGM(1,N)（经过正则化和数值限制的**工程适配近似实现**，不是每篇文献中所有同名模型的逐字复现）；
- 时间序列：ARIMA、Holt、Holt-Winters、SES（只利用训练段历史质量进行闭环禁用的多步外推；恒定训练序列由训练末值外推）；
- 核和线性：SVR、KRR、Ridge、PLS；
- 树模型：XGBoost、CatBoost、RandomForest、ExtraTrees；
- 浅层网络：MLP。

所有基线都统一用训练前64%/中间16%选超参，再用前80%训练，最后20%评估。随机神经网络/树集成采用2026、2027、2028三个种子；确定性模型一次运行。**每个模型的结果不得未经说明地与旧包中的历史表格混合**，因为基线实现和训练预算可能不同。

时间序列和灰色累加基线作用于删点后索引，因此其假设与原始连续0.1s采样过程不同。完整论文中应专门讨论该限制。基线输出名字与标准算法一致，但任何改动均应如实报告，不能人为削弱基线以显示主模型更优。

## 六、复现核对

本包中的选定模型重新训练与原始记录相同：
- 熔深D: **MAE 0.041299mm、RMSE 0.123645mm、MAPE 0.739411%**；
- 熔宽W: **MAE 0.134326mm、RMSE 0.194752mm、MAPE 3.350459%**。

数据长度：212、121、89、106；测试点43、25、18、22；共108个测试点/目标。

图形300dpi。运行不同依赖或平台可能造成少量随机差异，建议保留固定随机种子、保存版本信息。
