# Proposal 评语解读与后续修改注意事项

> 已提交的 Proposal **不再改**。本文只整理助教反馈，以及 Progress Report / Final Report 必须怎么改。
> Quercus 成绩：**4.5 / 5**（对应评分细则 **22.5 / 25**）。

---

## 一、总分怎么来的

| 项目 | 得分 | 说明 |
|---|---|---|
| Introduction | **3.5 / 4** | 扣 0.5：把“微调公开发表的 ImageNet 网络”当成了主卖点，这是常见配方，不算你自己的模型设计 |
| Architecture | **1 / 2** | 扣 1：真正干活的是预训练 ResNet-18，多任务损失和阈值有用，但网络本体不是你设计的 |
| Link to GitHub / Colab | **0 / 1** | 作者栏写了 “to be created”，没有可用的仓库链接 |
| 其余项 | 满分 | 动机、数据方案、`label_map.csv`、pHash、按原图分组、自拍测试集、HOG+HSV+SVM 基线都被表扬 |
| **合计** | **22.5 / 25 → 4.5 / 5** | |

**一句话：** 选题和数据计划很好；扣分几乎全在 “主模型是不是你自己设计的” 和 “有没有放代码链接”。

---

## 二、助教表扬了什么（继续保留）

这些后续报告里不要削弱，反而要做成可复现的结果：

1. **核心洞见**：垃圾桶类别不是材质的简单函数（用过的纸巾是纸，却进 Green Bin）。按市政规则重新标注现有数据集，本身就是有意义的数据贡献。
2. **`label_map.csv` 标签统一（label harmonisation）**
3. **划分前做 pHash 去重**
4. **TACO 裁剪按源照片分组（grouped split）**，防数据泄漏
5. **自拍手机照片只做最终测试**
6. **可完全复现的 HOG + HSV + SVM 基线**

---

## 三、扣分项怎么改（只改后续工作，不改已交 Proposal）

### 1. Introduction（再写 Progress / Final 时）

- **问题：** 引言里主要在论证 “ImageNet 预训练 + 微调有效”，听起来像跟着现成菜谱走。
- **改法：** 明确写出 **你自己的模型贡献**，例如：
  - 从零训练的紧凑 CNN（compact CNN trained from scratch）；
  - **分层头（hierarchical head）**：材质预测先给出规则先验，再加一个可学习的修正项，专门处理纸巾、咖啡杯等“材质 ≠ 桶”的例外；
  - 把微调 ResNet-18 写成 **强基线（strong baseline）**，不再写成主模型。

### 2. Architecture（必须在 Progress Report 落地）

- **问题：** 主模型 = 预训练 ResNet-18 + 换最后一层。
- **改法（助教给的具体方案，按此执行）：**
  - **主模型（Primary Model）：** 你设计并从头训练的紧凑 CNN + 分层头（材质 → 桶规则先验 + 修正项）。
  - **强深度学习基线：** 微调 ResNet-18（原来的主模型降级）。
  - **经典基线：** HOG + HSV + PCA + RBF-SVM（保留）。
  - 报告里写清：层数、参数量，并和 ResNet-18、SVM 对比。

### 3. GitHub / Colab（立刻补）

- 现在就建公开仓库或 Colab，Progress Report 作者栏必须放真实链接。
- 仓库里至少放：`README`、数据清洗脚本、训练脚本、`label_map.csv`、报告用图（不要上传整库原始图）。

---

## 四、Progress Report 前必须完成的清单（来自评语末段）

| 优先级 | 任务 | 为什么 |
|---|---|---|
| P0 | **定下哪个模型是“你的”** | 否则 Architecture 还会继续丢分 |
| P0 | **建好合并后的数据集**，报告每个 split 下每个 bin / material 的数量 | 评分要求数据工作可复现 |
| P0 | **创建 GitHub / Colab 并写进作者栏** | 否则继续丢 1 分 |
| P1 | **尽早开始拍自拍测试集**，拍完先锁住、最终评估前不要碰 | 真实场景测试是最好的检验 |
| P1 | **按数据来源（source）报准确率** + Grad-CAM | Green Bin 大多来自 Kaggle 网页图，模型可能学“图片风格”而不是学“该扔哪个桶” |
| P2 | 把微调 ResNet-18 和 SVM 都当基线，和自设 CNN 同表对比 | 回应 “make ResNet your strong baseline” |

---

## 五、后续写作时固定口径（避免再被扣分）

**可以说：**

- “Primary model is a from-scratch hierarchical CNN that encodes Toronto material→bin rules and learns a residual correction for exceptions.”
- “A fine-tuned ResNet-18 is reported as a strong transfer-learning baseline, not as the main contribution.”
- “Classical HOG+HSV+SVM is the non-deep baseline.”

**不要再说：**

- “We use ResNet-18 as our model.”（会再次触发 Architecture 扣分）
- “Deep learning works because ImageNet fine-tuning is strong.”（要改成：深度模型能学外观变化；**我们的贡献**是分层规则头 + 市政标签数据）

---

## 六、和 Final Report 的衔接

Final 时必须展示：

1. 自设 CNN（主） vs ResNet-18（强基线） vs SVM（经典基线）的定量表；
2. 参数量 / 层数对比；
3. 按 source 的准确率，证明没有只靠 Kaggle 风格作弊；
4. 自拍测试集结果（从未参与训练/调参）；
5. Grad-CAM：例外样本（纸巾等）上，模型是否看物体而不是背景；
6. 拒答阈值 \(\tau\) 的 coverage–accuracy 曲线。
