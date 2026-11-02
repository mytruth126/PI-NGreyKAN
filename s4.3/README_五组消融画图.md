# PI-NGreyKAN：五组消融实验与论文配对分析图

## 直接运行（无需重新训练）

先解压整个包，进入 `PI_NGreyKAN_完整复现项目` 文件夹：

```bash
python run_ablation_and_plot.py --step plot
```

图片保存到 `figures/ablation/Pairwise_Ablation_D_300dpi.png`（熔深）及
`figures/ablation/Pairwise_Ablation_W_300dpi.png`（熔宽），并额外提供矢量PDF。

## 从头重训5个消融模型

```bash
python run_ablation_and_plot.py --step all
```

只重训：`python run_ablation_and_plot.py --step train`。
也可调用 `python code/run_ablation.py --resume` 跳过已完成的同模型/随机种子训练。
每次不带 `--resume` 重新运行训练会覆盖 `results/ablation/` 中的结果。

需安装 `numpy, pandas, matplotlib, torch`（运行主项目的其余对比模型还需要原`requirements.txt`中的附加依赖）。

## 模型标号

|标号|显示名称|相对完整模型的唯一改动|
|---|---|---|
|M1|PI-NGreyKAN|完整模型，物理损失系数2.0|
|M2|without KAN|5→2线性映射替换5→20→2 KAN|
|M3|without Grey Development|移除线性灰色发展项 a*v|
|M4|without Adaptive Development|自适应 a_t 改为每目标固定可学习系数 a|
|M5|without Thermal Feature|将第五维标准化热输入特征 Q 恒置0|

已从论文展示和重跑默认列表中去掉`w/o nonlinear term`及`w/o physical constraint`。
**完整模型依旧保留非线性项及物理约束；没有改变主模型。**

## 数据与统计口径

- 四组数据为最新固定保留数据（W01=212, W02=121, W03=89, W04=106）。
- 各组最后20%为测试集：43、25、18、22，共108个测试点。
- 3个随机种子：2026、2027、2028。
- 所有模型和真实值来自既有训练测试输出 `results/ablation/ablation_test_predictions.csv`。
- 图上排12个点，每个点为**一个焊缝案例×一个随机种子**的MAE，X为M1，Y为相应消融模型；参考线 y=x。
- 图下排柱高：**每种子合并108个测试点算MAE，然后对3个种子求均值**；误差棒：3个随机种子MAE的标准差。柱上百分数为相对M1的误差变化，不是显著性检验。
- 上排按案例等组展示的是配对诊断数据，下排依据108点总体误差统计，与`ablation_table.csv`一致。
- 原始消融输出已筛选成这5组并统一重新标注M1-M5，筛选没有重新计算或改变任何预测结果。
- 原数据依据真实熔深标签筛选，属于筛选数据集专项评估，不宜视为未经筛选的实际在线泛化结果。

## 文件

- `code/run_ablation.py`：五组消融实验训练和验证（用最新主模型代码）。
- `code/plot_ablation_pairwise.py`：两幅2×4配对散点+柱状图（**不训练**）。
- `run_ablation_and_plot.py`：统一运行入口。
- `code/base_model.py`、`code/run_simple_physics.py`：完全一致的原始主模型实现和物理损失实现。
- `data/`：最新四组固定数据。
- `results/ablation/`：已有真实实验输出、逐点测试预测和统计汇总。
- `figures/ablation/`：输出PNG（300dpi）和PDF矢量图。
- `run_pipeline.py`及原项目的基线训练/绘图脚本仍保留，可单独复现完整20模型对比。
