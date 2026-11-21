# Progress Report 讲解：BinSight

> 对应文件：`progress/progress.tex` → 提交 `progress/progress.pdf`  
> 评分：20 分；正文 ≤ 3 页；必须有数据、基线、主模型，且各至少一组定量 + 定性结果。

---

## 一、这份 Progress Report 在证明什么

助教要看三件事是否已经落地：

1. **数据基本齐了**，清洗步骤能复现，有统计表和样例图；
2. **基线能跑**，不是随机猜，有数字和样例；
3. **主神经网络能跑**，有结构图、参数量、至少一轮训练结果。

Proposal 评语还要求：**主模型必须是你自己设计的**。所以报告里：

| 角色 | 模型 | 为什么 |
|---|---|---|
| **Primary（主模型）** | 从零训练的 Hierarchical CNN | 你的设计：材质 → 规则先验 + 修正项 |
| **Strong baseline** | 微调 ResNet-18 | 原来的主模型降级，回应评语 |
| **Classical baseline** | HOG + HSV + SVM | Proposal 里就定好的非深度基线 |

---

## 二、逐段写了什么（对照评分）

### Brief Project Description（5 分）

- 动机：Blue Bin 约 30% 污染；Waste Wizard 只接受文字。
- 输入 / 输出：一张照片 → 垃圾桶 + 材质，低置信度拒答。
- 为什么用深度学习：外观变化大；**桶 ≠ 材质函数**。
- 点明评语修改：**主模型是自设分层 CNN**，ResNet 只是强基线。
- 配了架构示意图（Figure 1）。

### Data Processing（Notable Contribution 的一部分）

当前清洗结果（`data/processed/manifest.csv`）：

| Split | Blue | Green | Garbage | Total |
|---|---:|---:|---:|---:|
| Train | 2108 | 762 | 227 | 3097 |
| Val | 451 | 164 | 47 | 662 |
| Test | 451 | 165 | 44 | 660 |
| **All** | **3010** | **1091** | **318** | **4419** |

按来源：

| Source | Blue | Green | Garbage |
|---|---:|---:|---:|
| TrashNet | 2362 | 0 | 137 |
| TACO | 648 | 13 | 181 |
| Kaggle | 0 | 1078 | 0 |

**关键风险（助教点名的 source confound）：** Green 几乎全来自 Kaggle 网页图。模型可能学“图片风格”而不是“该扔哪个桶”。报告里必须写 per-source 准确率，并计划用自拍照片最终检验。

材质分布（全体）：cardboard 474，glass 529，metal 520，organic 1081，other 161，paper 646，plastic 1008。

挑战：TACO 只成功下载约 440/1500 张原图（Flickr 链接失效）。

### Baseline Model

**SVM 结果（已完成）：**

| 指标 | 数值 |
|---|---:|
| Test bin accuracy | **0.859** |
| Test macro-F1 | **0.687** |
| Val macro-F1（选超参） | 0.756 |
| 超参 | \(C=1\), \(\gamma=10^{-3}\) |
| 按来源 acc | TrashNet 0.957 / Kaggle 0.864 / TACO 0.556 |
| Garbage recall | 仅 0.25（弱类） |

定性：`results/svm_qualitative.png`、`results/svm_confusion.png`。

### Primary Model

**分层 CNN（已完成，best = epoch 5）：**

| 指标 | 数值 |
|---|---:|
| 参数量 | **1,208,319（约 1.21M）** |
| Test bin accuracy | **0.682** |
| Test macro-F1 | **0.557** |
| Test material accuracy | 0.462 |
| Val bin accuracy | 0.675 |
| 按来源 acc | Kaggle 0.920 / TrashNet 0.698 / TACO 0.323 |

结构公式：

\[
p_{\text{prior}} = p_{\text{mat}} R,\quad
\text{bin\_logits} = \log(p_{\text{prior}}) + \text{correction}(h)
\]

其中 \(R\in\mathbb{R}^{7\times3}\) 可学习，编码多伦多材质→桶软规则；correction 学纸巾等例外。

**ResNet-18 强基线（已完成，best = epoch 7）：**

| 指标 | 数值 |
|---|---:|
| 参数量 | **11,310,410（约 11.3M）** |
| Test bin accuracy | **0.905** |
| Test macro-F1 | **0.811** |
| Test material accuracy | 0.835 |
| 按来源 acc | TrashNet 0.944 / Kaggle 0.957 / TACO 0.718 |

**怎么读这些数字：**

- 当前排序：**ResNet-18 (0.905) > SVM (0.859) > Hierarchical CNN (0.682)**。
- 这符合预期：预训练大模型最强；你的主模型从零训、参数只有 ResNet 的约 1/9、只训了 5 个 epoch，已经明显高于随机，Final 阶段用 GPU 加长训练是主要提升点。
- 三个模型在 Kaggle 来源上都很高、TACO 都明显更低 → 正好印证 style confound，Final 必须盯住自拍测试集和 Grad-CAM。

---

## 三、和 Proposal 评语的对应关系

| 评语要求 | Progress 里怎么回应 |
|---|---|
| 主模型要是你设计的 | Hierarchical CNN + 规则先验 + 修正头 |
| ResNet 降为强基线 | 表格里单独一行 “strong baseline” |
| 合并数据并报告每 split / bin / material 计数 | Table 1 + manifest |
| 注意 Green≈Kaggle 风格 | Figure + per-source accuracy |
| 尽早拍自拍测试集 | 文中写 reserved for final，尚未计入训练 |
| GitHub 链接 | **提交前必须换成真实公开 URL**（作者栏现在是 TBC） |

---

## 四、专业名词速查（中英）

| 中文 | English | 在本项目中的意思 |
|---|---|---|
| 分层头 | Hierarchical head | 先预测材质，再经规则矩阵得到桶的先验，再加残差修正 |
| 规则先验 | Rule prior | \(p_{\text{mat}} R\)，软的市政映射 |
| 残差修正 | Residual correction | 小 MLP 输出，专门学规则解释不了的例外 |
| 从零训练 | Train from scratch | 不加载 ImageNet 权重 |
| 迁移学习强基线 | Transfer-learning strong baseline | 微调 ResNet-18 |
| 来源混杂 / 风格捷径 | Source confound / shortcut | 用图片来源风格猜标签 |
| 宏平均 F1 | Macro-F1 | 三类 F1 平均，不让 Blue 主导 |
| 感知哈希去重 | pHash de-duplication | 划分前去掉近重复图 |
| 分组划分 | Grouped split | 同一 TACO 原图的裁剪不跨集合 |

---

## 五、提交前检查清单

1. [ ] 等 ResNet 训练结束，把 `progress.tex` 里 `\texttt{RESNET_*}` 占位符换成真实数字，并跑 `scripts/make_progress_figures.py`
2. [ ] 用 Tectonic / Overleaf 编译，确认**正文 ≤ 3 页**
3. [ ] 作者栏换成**真实 GitHub 或 Colab 链接**（否则会再丢链接分）
4. [ ] 提交 **`progress/progress.pdf`** 到 Quercus（不是 `.tex`，不是讲解 `.md`）
5. [ ] 开始拍自拍测试集，按材质分文件夹，**先锁住、最终评估前不用来训练**

---

## 六、本地命令备忘

```bash
python scripts/clean_data.py
python scripts/train_baseline_svm.py
python scripts/train_models.py --model hierarchical --epochs 10
python scripts/train_resnet_fast.py
python scripts/eval_checkpoint.py --ckpt results/hierarchical_best.pt
python scripts/make_progress_figures.py
```
