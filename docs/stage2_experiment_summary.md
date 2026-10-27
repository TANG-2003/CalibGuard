# CalibGuard 第二阶段实验小结

**阶段名称：** 多任务风险感知网络微调  
**正式协议：** 6 Epoch × 3 Seeds × 4 Variants  
**状态：** 核心实验与专项分析已完成；模型权重独立备份尚需确认

## 1. 研究目标与模型

采用 DeBERTa-v3-base 作为共享编码器，建立三个预测头：Binary Head（二分类，输出 Unsafe Score）、Risk Head（22 类风险多标签识别）、Domain Head（19 类领域多标签识别）。主任务采用交叉熵；辅助任务采用带有效标签掩码的二元交叉熵。对照组为 Binary Only、Binary + Risk、Binary + Domain 和 Full Multi-task。该结构用于研究辅助监督是否有助于安全识别，而不是预设其必然有效。

## 2. 数据与模型选择协议

原始 Train/Test 维持约 7:3：Train 73,365、ID Test 31,444。Train 内部分成 Fit 66,028 与 Validation 7,337，验证集用于按 Unsafe F1 选择最佳 checkpoint。三个随机种子为 42、43、44；每次最多训练 6 Epoch，采用各自前六轮中验证集 Unsafe F1 最高的 checkpoint。所有 ID Test 使用固定二分类决策阈值 0.5；未根据测试集重新选 Epoch。

需要区分：本次 6 Epoch 正式模型在选定后进行统一测试，但更早期探索模型曾评估过同一个 ID Test，因此不能将其描述为在整个研究过程中从未查看过的“完全原封未动”测试集。

## 3. ID Test 正式消融结果

以下为三个随机种子的平均值，F1 的标准差为样本标准差。

| 模型 | Unsafe F1（Mean ± Std） | Unsafe Recall | Unsafe Precision | Safe FPR | Accuracy |
|---|---:|---:|---:|---:|---:|
| Binary Only | 0.987369 ± 0.000581 | 0.985179 | **0.989570** | **0.003132** | 0.994159 |
| Binary + Risk | 0.986259 ± 0.000444 | 0.985042 | 0.987489 | 0.003767 | 0.993639 |
| Binary + Domain | 0.987461 ± 0.000659 | 0.987009 | 0.987916 | 0.003643 | 0.994191 |
| **Full Multi-task** | **0.988343 ± 0.000380** | **0.989067** | 0.987632 | 0.003739 | **0.994594** |

Full Multi-task 相对 Binary Only 的 Unsafe F1 增加约 0.0974 个百分点，Recall 增加约 0.3888 个百分点，同时 Safe FPR 增加约 0.0607 个百分点。单独添加 Risk Head 未改善平均 Unsafe F1。当前结论仅限于所用数据、随机种子及协议；不能宣称所有辅助头分别有增益或同时改善安全与效用。

## 4. 辅助任务验证结果

Full Multi-task 在验证集上的三个种子平均结果约为：

| 任务 | Micro-F1 | Macro-F1 | Micro-AP |
|---|---:|---:|---:|
| Risk（22 类） | 0.6788 | 0.4856 | 0.7380 |
| Domain（19 类） | 0.7511 | 0.4956 | 0.8145 |

有效标注样本数：Validation Risk 1,700，Domain 1,806；ID Test 类别级分析 Risk 7,287，Domain 7,679。类别级指标以 Sigmoid 概率、阈值 0.5 计算，类别级 ID Test 分析为描述性误差分析，不用于阈值选择。

风险类别中，Cybersecurity Threats (Beyond Malware)（Support 855，平均 F1 0.7983）、Crime Content（697，0.7633）、Abusive Content（1,014，0.7473）相对较高；Confabulation（21）、Decision-Making in Critical Systems（72）、Extremism and Radicalization（49）、Intellectual Property（40）的平均 F1 均为 0。

领域类别中，Technology（1,217，0.9078）、Biology（411，0.8643）、Healthcare（749，0.8252）相对较高；Infrastructure（58）、Religion（39）、Retail（30）、Travel（33）的平均 F1 均为 0。F1 为 0 不代表模型预测概率完全无排序能力，应结合 AP 与支持度理解，不能仅凭样本数认定因果机制。

## 5. 冻结模型专项评估

下表为 Full Multi-task 三个种子平均结果：

| 数据子集 | 样本量及当前标注 | 指标 | 平均值 |
|---|---|---|---:|
| OR-Bench Hard | 1,319 条 Safe | Safe FPR | 0.008087 |
| RedBench 中 XSTest 配置子集 | 448 条 Safe | Safe FPR | 0.867560 |
| Harmful | 630 条 Unsafe | Unsafe Recall | 0.750264 |

所有专项子集的 `label_status` 均为 `provisional`。XSTest 配置子集记录的 `dataset_sources` 为 RedBench、`configs` 为 XSTest；不能未经额外来源核实即把数字表述为“官方 XSTest 独立基准成绩”。按 `prompt_hash` 检查，上述子集与 Train 无精确重复，但此检查不排除语义近重复。

专项测试表明不同子集上的预测行为差异明显，尤其 XSTest 配置子集 Unsafe 预测比例异常高、Harmful Recall 低于 ID Test。当前指标衡量的是分类器预测，不等同于下游生成模型的实际拒绝率。

## 6. 阶段结论与第三阶段衔接

1. 完整多任务联合监督在现有 ID Test 的四种配置中取得最高平均 Unsafe F1 与 Recall，但误报代价有所增加。
2. 风险和领域辅助头具备一定多标签预测能力，但类别级能力不均衡；部分类别在固定 0.5 阈值下无法正确识别正例。
3. 同分布测试分数不能代替困难合规请求、有害请求及跨分布评估。
4. 第三阶段将研究类别感知校准、长尾收缩及 ALLOW/REVIEW/REFUSE 路由；其独立校准数据来源与统计保证范围需另外确定。已用于拟合、模型选择或评估的数据不能未经协议澄清就当作全新独立校准集。

## 7. 可复现性及归档范围

正式代码包括训练公共实现、多随机种子训练脚本、6 Epoch 运行队列、辅助验证、ID Test、专项评估及类别级分析脚本。轻量实验记录保留各组 `run_config.json`、`run_summary.json`、验证日志、`id_test_metrics.json`、`specialized_metrics.json`、适用的辅助任务结果、`per_class_id_test.json` 及 `stage2_per_class_summary.csv`。

原始数据、样本 Prompt、模型权重和访问凭据不提交普通 Git 仓库。模型 checkpoint 尚需独立备份，以支持后续第三阶段运行。正式结果采用现存 JSON/CSV 为准，本文数值来自已报告的实验输出。
