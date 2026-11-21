# Proposal 讲解：BinSight —— 用深度学习做多伦多垃圾分类

> 对应文件：`proposal/proposal.tex`（源文件）和 `proposal/proposal.pdf`（编译结果）。
> 本文分三部分：**① 项目一句话概括与选题理由**；**② 逐段解释 proposal 写了什么、对应哪条评分标准**；**③ 用到的专业基础知识**（中英双语）。

---

## 目录

1. [项目一句话概括](#一项目一句话概括)
2. [为什么选这个题目](#二为什么选这个题目)
3. [Proposal 逐段解读](#三proposal-逐段解读)
4. [专业基础知识详解](#四专业基础知识详解)
5. [模型代码草图（帮助理解）](#五模型代码草图帮助理解)
6. [提交前必须核对的事项](#六提交前必须核对的事项)
7. [助教可能会问的问题](#七助教可能会问的问题)

---

## 一、项目一句话概括

**输入一张家庭垃圾的照片，模型输出它应该扔进多伦多的哪个垃圾桶：Blue Bin（可回收）、Green Bin（厨余有机）或 Garbage（其他垃圾），同时识别它的材质；如果模型不确定，就回答“不确定，请查 Waste Wizard”。**

---

## 二、为什么选这个题目

| 考虑因素 | 这个题目的情况 |
|---|---|
| **真实痛点** | 多伦多市政府 2023 年报告：Blue Bin 里约 30%（按重量）是不该放进去的东西（contamination，污染物），宣传多年都没有改善 |
| **深度学习适合** | 输入是图像，同一类垃圾的外观差异极大（压扁、脏污、光照不同），手工特征效果差（TrashNet 的 SVM 只有 63%） |
| **满足“数据工作”要求** | 合并 3 个公开数据集 + 自己拍照；需要裁剪、重新标注、去重、清洗，工作量很实在 |
| **满足“和同学做法不同”** | 往届有 3 个垃圾分类项目，但它们都是按“材质”分类。本项目的区别是：①按**多伦多市政规则**分到垃圾桶；②**多任务**同时预测垃圾桶和材质；③不确定时**拒答**；④在**自己拍的真实照片**上测试 |
| **Colab 可完成** | ResNet-18 只有约 1170 万个参数，几千张图在 Colab 免费 GPU 上每个 epoch 只需一两分钟 |
| **难度高于 Lab** | Lab 3 是单任务的手势分类；本项目多了多数据源合并、标签体系设计、多任务输出头、拒答机制和真实场景测试 |
| **与你的 Lab 经验衔接** | 迁移学习（Lab 3 Part 4 的 AlexNet 特征）、按组划分防泄漏（Lab 3 按人划分）、训练曲线与调参（Lab 2/3）都能直接复用 |

> 如果你更想做别的题目，可以保留 proposal 的结构，替换内容。

---

## 三、Proposal 逐段解读

### 3.1 标题与作者块

- 标题：*BinSight: Deep Learning for Toronto Household Waste Sorting*，给项目起一个好记的名字，方便在报告里引用；
- 作者块包含 GitHub 链接，对应评分标准里的 **Link to Github or Colab Notebook（1 分）**，你需要把 `YOUR-USERNAME` 换成自己的仓库地址。

### 3.2 Introduction（引言，4 分）

**写了什么：**

1. **痛点 + 数据证据**：多伦多居民经常扔错桶，Blue Bin 污染率约 30% [市政府 2023 年报告]；现有的 Waste Wizard 需要用户知道物品名称并手动搜索；
2. **明确目标（Goal）**：一张照片 → (i) 垃圾桶 (ii) 材质（7 类），不确定时拒答；
3. **为什么用深度学习（Why deep learning）**：
   - 垃圾外观变化很大，手工特征 SIFT + SVM 只能到 63%；
   - ImageNet 预训练的 CNN 可以直接从像素学到纹理、光泽和形状，几千张图就能微调（fine-tune）到 95%（RecycleNet）；
   - 垃圾桶类别**不是材质的简单函数**（用过的纸巾是“纸”，却要进 Green Bin），所以需要模型学习物体层面的特征，因此用**多任务 CNN**。

**对应满分标准：** 目标清楚、说明了为什么有用（有数据）、令人信服地论证了为什么要用深度学习。

### 3.3 Illustration（示意图，4 分）

用 TikZ 画的流程图，从左到右分别是：

```
4 个数据源 → 数据处理（裁剪/重新标注/去重/分组划分/增强）→ ResNet-18 → 共享全连接层
          → 两个输出头（垃圾桶 3 类 + 材质 7 类）→ 置信度判断 → 输出垃圾桶或“不确定”
```

- 虚线框和虚线箭头表示“自己拍的照片只用于测试”；
- 图下方写出了损失函数 \(\mathcal{L} = \mathrm{CE}_{\text{bin}} + \lambda\,\mathrm{CE}_{\text{material}}\)；
- **对应满分标准：** 一眼就能看懂核心思路和结构，字少而清晰。

### 3.4 Background & Related Work（相关工作，4 分）

列了 **6 个**相关工作（要求至少 5 个），每个都写了“做了什么 + 结果 / 局限”：

| 相关工作 | 做了什么 | 对本项目的意义 |
|---|---|---|
| **TrashNet**（Thung & Yang, 2016） | 2527 张白底单物体照片，6 类；SIFT + SVM 准确率 63%，优于他们的小 CNN | 数据来源之一；说明手工特征有上限 |
| **RecycleNet**（Bircanoğlu et al., 2018） | 在 TrashNet 上比较多种 CNN；ImageNet 预训练的 DenseNet121 微调后达到 95% | 证明迁移学习有效，但只在白底图上验证过 |
| **TACO**（Proença & Simões, 2020） | 1500 张真实场景照片，4784 个垃圾实例，60 个类别，COCO 格式，Mask R-CNN 基线 | 数据来源之一；说明真实场景很难 |
| **Majchrowska et al.**（2022） | 合并 10 多个垃圾数据集，统一成 7 类；EfficientDet-D2 + EfficientNet-B2，分类准确率约 75% | 说明真实场景下准确率大幅下降，跨数据集统一标签至关重要 |
| **Kaggle Waste Classification Data**（Sekar, 2019） | 25,077 张网络图片，只分有机物和可回收物两类 | 有机垃圾图片来源 |
| **Toronto Waste Wizard** | 市政府的文字查询工具 | 现有产品，本项目的“视觉版”补充 |

最后一句点明了**本项目的差异**：按市政规则分类、多任务联合学习、不确定时拒答、在新采集的家庭照片上测试。

> 所有数字都已核实过原始论文或官方页面。

### 3.5 Data Processing（数据处理，4 分）

这部分是评分重点（“要看到收集、改造、清洗数据的努力”），写得最详细：

**四个数据源：**

| 来源 | 数量 | 用途 |
|---|---|---|
| TrashNet | 2527 张 | 训练 / 验证 / 测试 |
| TACO 物体裁剪 | 预计约 3000 张 | 训练 / 验证 / 测试 |
| Kaggle 有机物（清洗后） | 预计约 800 张 | 训练 / 验证 / 测试 |
| 自己拍的照片 | 约 400 张（每个材质类至少 50 张） | **只用于最终测试** |

**六个处理步骤：**

- **(a) TACO 裁剪**：TACO 是检测数据集，一张图里有多个垃圾。按标注的边界框（bounding box）向外扩 10% 裁剪出单个物体；短边小于 64 像素的丢掉（主要是烟头这类太小的物体）；去掉 “Unlabeled litter”（未标注垃圾）类别；记录每个裁剪来自哪张原图（`image_id`）；
- **(b) 标签统一（Label Harmonization）**：建一张映射表 `label_map.csv`，把 TACO 的 60 个类别和 TrashNet 的 6 个类别分别映射成（材质，垃圾桶）对，依据是 Waste Wizard 的查询结果。例如：易拉罐 → (metal, Blue)、食物残渣 → (organic, Green)、烟头 → (other, Garbage)。含义不明确的类别直接丢弃；TrashNet 里 137 张 “trash” 图逐张人工重新标注；
- **(c) 有机物清洗**：从 Kaggle 有机类中随机抽 1500 张，人工删掉手绘图、拼图、带文字的图和多物体场景，目标保留约 800 张；
- **(d) 去重（De-duplication）**：用感知哈希（pHash）找出近似重复的图片（汉明距离 ≤ 4），在划分数据集**之前**删掉；
- **(e) 划分（Split）**：70/15/15，按垃圾桶类别分层；**同一张 TACO 原图裁出来的所有物体必须在同一个集合里**（分组划分，防止泄漏）；自己拍的照片完全不参与训练和调参；
- **(f) 变换（Transforms）**：先补边成正方形，再缩放到 224×224，用 ImageNet 的均值和方差归一化；训练时做数据增强；损失函数里用类别权重处理不平衡。

**对应满分标准：** 数据来源清楚并有引用；每一步都具体到阈值和数量，同学可以照着复现。

### 3.6 Architecture（网络结构，2 分）

- **骨干网络（Backbone）**：ImageNet 预训练的 ResNet-18，去掉原来的 1000 类输出层；
- **共享全连接层**：512 → 256，ReLU，Dropout 0.3；
- **两个输出头（heads）**：垃圾桶头 3 类、材质头 7 类；
- **损失函数**：\(\mathcal{L}=\mathrm{CE}_{\text{bin}}+\lambda\,\mathrm{CE}_{\text{material}}\)，\(\lambda=0.5\)（之后调）；
- **两阶段训练**：
  - 阶段 1：冻结骨干网络，只训练头部（Adam，学习率 \(10^{-3}\)，5 个 epoch）；
  - 阶段 2：解冻 ResNet 的 `layer3` 和 `layer4`（学习率 \(10^{-4}\)，余弦衰减，最多 30 个 epoch，batch size 64），用验证集的垃圾桶 Macro-F1 做早停；
- **拒答机制**：垃圾桶头的最大概率低于阈值 \(\tau\) 时输出“不确定”。\(\tau\) 在验证集上选，使被接受的预测准确率 ≥ 95%；
- **可解释性**：用 Grad-CAM 检查模型是在看物体还是在看背景；
- **消融实验**：单任务 vs 多任务、冻结 vs 微调；
- **成功标准**：测试集垃圾桶准确率 ≥ 85%，自拍照片上 ≥ 75%，且比基线高至少 15 个百分点。

**对应满分标准：** 写清了网络类型和关键组件，细节足够让同学接手。

### 3.7 Baseline Model（基线模型，2 分）

- **基线 1：多数类预测**，永远预测样本最多的那个桶，作为下界；
- **基线 2：HOG + 颜色直方图 + SVM**：
  1. 缩放到 128×128；
  2. 灰度图上提取 HOG 特征（9 个方向，16×16 像素的 cell，2×2 个 cell 组成一个 block），得到 1764 维；
  3. HSV 颜色直方图 8×8×8 = 512 维；
  4. 拼接 → 标准化 → PCA 降到 256 维 → RBF 核 SVM；
  5. 在验证集上网格搜索 \(C\in\{0.1,1,10,100\}\)、\(\gamma\in\{10^{-4},10^{-3},10^{-2}\}\)；
  6. 工具：scikit-image + scikit-learn；
- 和 TrashNet 的 SIFT + SVM 思路一致，可以对照。

**对应满分标准：** 选择合理，参数细节足够同学复现。

### 3.8 Ethical Considerations（伦理考量，2 分）

分三个层面（满分要求“数据采集的伦理 + 模型使用的影响”两方面都要谈）：

1. **数据**：TACO 图片来自 Flickr，可能拍到人或车牌，所以只用物体裁剪，含人的裁剪丢掉；遵守许可证（TACO 是 CC BY 4.0）；自拍照片不含个人信息（排除带地址的信件，删除 EXIF 元数据）；
2. **偏见与局限**：训练图片主要来自美国、欧洲和网络，加拿大的包装可能代表性不足；垃圾分类规则因城市而异、也会随时间变化，所以模型在多伦多以外无效，映射表需要做版本管理；
3. **错误的后果**：错判成 Blue Bin 会加重污染；锂电池等危险品可能在垃圾车里起火，不在模型的处理范围内。因此设计了拒答机制，引导用户去查 Waste Wizard，并声明模型不应被用来监控或处罚居民。

### 3.9 References（参考文献，1 分）

`references.bib` 里有 14 条，用 `plain` 风格自动生成，格式统一；参考文献在 `\newpage` 之后，不占正文页数。

---

## 四、专业基础知识详解

### 4.1 任务类型

| 概念 | 解释 |
|---|---|
| **图像分类（Image Classification）** | 输入一张图，输出一个类别。本项目属于这一类 |
| **目标检测（Object Detection）** | 输入一张图，输出多个边界框（bounding box）及其类别。TACO 原本是检测数据集，我们把框裁出来，转化成分类任务 |
| **实例分割（Instance Segmentation）** | 在检测基础上再给出每个物体的像素级掩码（mask）。TACO 论文用的 Mask R-CNN 就是这类模型 |
| **COCO 格式（COCO Format）** | 一种常用的标注 JSON 格式，包含 `images`、`annotations`（含 `bbox: [x, y, w, h]`、`category_id`、`image_id`）和 `categories` |

### 4.2 卷积神经网络与 ResNet

- **卷积神经网络（Convolutional Neural Network, CNN）**：用卷积核（kernel / filter）在图像上滑动提取局部特征。浅层学边缘和颜色，深层学纹理、部件和物体（Lab 3 讲过）；
- **ResNet（残差网络，Residual Network）**：He et al. 2016 提出。核心是**残差连接 / 跳跃连接（Residual / Skip Connection）**：
  \[
  \mathbf{y} = F(\mathbf{x}) + \mathbf{x}
  \]

  网络只需要学习“残差” \(F(\mathbf{x})\)，梯度可以通过 \(+\mathbf{x}\) 这条捷径直接回传，因此能训练很深的网络，缓解**梯度消失（Vanishing Gradient）**；
- **ResNet-18 结构**：`conv1` → `layer1`～`layer4`（每个 layer 含 2 个残差块，通道数依次为 64/128/256/512）→ **全局平均池化（Global Average Pooling, GAP）** → 512 维向量 → `fc`（原本输出 1000 类）。约 1170 万参数；
- **为什么选 ResNet-18 而不是更大的模型**：数据只有几千张，大模型容易**过拟合（Overfitting）**；ResNet-18 在 Colab 上训练快，迭代实验方便。

### 4.3 迁移学习（Transfer Learning）

- **思想**：在大数据集（ImageNet，约 120 万张图、1000 类）上预训练的网络已经学会了通用视觉特征，只要在小数据集上稍加调整就能用；
- **特征提取（Feature Extraction）**：冻结（freeze）骨干网络，只训练新加的头部（Lab 3 Part 4 用 AlexNet 就是这种方式）；
- **微调（Fine-tuning）**：解冻部分或全部骨干网络，用**更小的学习率**继续训练；
- **本项目的两阶段策略**：先冻结训练头部（新头部是随机初始化的，如果一开始就解冻，大梯度会破坏预训练权重），再解冻 `layer3`、`layer4` 微调（深层特征更“任务相关”，浅层的边缘和纹理特征通用，可以保持不动）；
- **ImageNet 归一化**：输入按通道减去均值 `[0.485, 0.456, 0.406]`、除以标准差 `[0.229, 0.224, 0.225]`，与预训练时保持一致。

### 4.4 全连接层、激活函数与正则化

- **全连接层（Fully-connected / Linear Layer）**：\(\mathbf{y} = W\mathbf{x} + \mathbf{b}\)；
- **ReLU（Rectified Linear Unit，线性整流单元）**：\(\max(0, x)\)，引入非线性；
- **Dropout（随机失活）**：训练时以概率 \(p\)（本项目 \(p=0.3\)）随机把神经元置零，防止过拟合；测试时关闭；
- **正则化（Regularization）**：所有降低过拟合的手段的统称，包括 Dropout、数据增强、权重衰减（weight decay）、早停，以及多任务学习。

### 4.5 Softmax 与交叉熵损失

- **Softmax**：把输出的 logits 转成概率：
  \[
  p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
  \]
- **交叉熵损失（Cross-Entropy Loss, CE）**：\(\mathrm{CE} = -\log p_{y}\)，即正确类别概率的负对数。PyTorch 的 `nn.CrossEntropyLoss` 内部已经包含 softmax，所以模型最后一层**不要**再加 softmax；
- **加权交叉熵（Weighted Cross-Entropy）**：
  \[
  \mathrm{CE}_w = -\, w_{y} \log p_{y},\quad w_k \propto \frac{1}{n_k}
  \]

  \(n_k\) 是第 \(k\) 类的样本数。样本少的类别权重大，用于处理**类别不平衡（Class Imbalance）**。PyTorch 中写作 `nn.CrossEntropyLoss(weight=...)`。

### 4.6 多任务学习（Multi-task Learning）

- **定义**：一个网络同时学习多个相关任务，共享底层表示（shared representation），各自有独立的输出头（task-specific heads）；
- **本项目**：共享 ResNet-18 + 共享全连接层，然后分出“垃圾桶头”（主任务）和“材质头”（辅助任务，auxiliary task）；
- **总损失**：\(\mathcal{L}=\mathrm{CE}_{\text{bin}}+\lambda\,\mathrm{CE}_{\text{material}}\)，\(\lambda\) 控制辅助任务的权重；
- **好处**（Caruana 1997）：辅助任务提供额外的监督信号，相当于一种正则化，让共享特征更通用。比如材质信息（“这是金属”）能帮助判断垃圾桶；
- **为什么这里需要两个头**：垃圾桶类别不能完全由材质推出（纸巾是纸却进 Green Bin），所以垃圾桶不能只靠“材质 → 查表”得到，需要单独学习；材质头又为模型提供了可解释的中间信息。

### 4.7 优化器与学习率调度

- **Adam**：自适应学习率的优化器，结合了动量（momentum）和 RMSProp，是默认的稳妥选择（Lab 2/3 用过）；
- **学习率（Learning Rate, lr）**：每一步更新的步长。微调时要用小学习率（\(10^{-4}\)），避免破坏预训练权重；
- **余弦退火 / 余弦衰减（Cosine Annealing / Cosine Decay）**：学习率按余弦曲线从初始值平滑降到接近 0，训练后期更稳定：
  \[
  \eta_t = \tfrac{1}{2}\eta_0\left(1+\cos\tfrac{\pi t}{T}\right)
  \]

  PyTorch 中是 `torch.optim.lr_scheduler.CosineAnnealingLR`；
- **Epoch / Batch / Iteration**：一个 epoch 指完整遍历一次训练集；batch size 64 指每次更新用 64 张图；
- **早停（Early Stopping）**：验证集指标连续若干个 epoch 不再提升就停止训练，并取验证集最好的检查点（checkpoint）。

### 4.8 数据相关概念

| 概念 | 解释 |
|---|---|
| **训练 / 验证 / 测试集（Train / Validation / Test Set）** | 训练集用来学参数；验证集用来调超参数和选模型；测试集只在最后评估一次 |
| **分层划分（Stratified Split）** | 每个子集里各类别的比例和整体一致，避免某个子集缺少某一类 |
| **分组划分（Grouped Split）** | 同组样本（同一张原图的多个裁剪）只能进入同一个子集。sklearn 有 `GroupShuffleSplit` 和 `StratifiedGroupKFold` |
| **数据泄漏（Data Leakage）** | 测试信息“漏”进训练过程，导致测试结果虚高。比如同一个瓶子的两张几乎一样的照片分别进了训练集和测试集 |
| **感知哈希（Perceptual Hash, pHash）** | 把图片压缩成 64 位指纹，视觉上相似的图片指纹也相似。两个指纹不同的位数叫**汉明距离（Hamming Distance）**，≤ 4 视为近似重复。Python 库：`imagehash` |
| **标签统一（Label Harmonization）** | 不同数据集的类别定义不同，需要映射到同一套标签体系 |
| **数据增强（Data Augmentation）** | 对训练图做随机变换，生成“新”样本，提升泛化能力。本项目使用：`RandomResizedCrop(224, scale=(0.6, 1.0))`、`RandomHorizontalFlip()`、`RandomRotation(15)`、`ColorJitter(0.3, 0.3, 0.3)` |
| **补边成正方形（Padding to Square）** | 直接缩放会把细长的物体（瓶子）压扁变形；先补边成正方形再缩放可以保持长宽比（aspect ratio） |
| **领域偏移 / 分布偏移（Domain Shift / Distribution Shift）** | 训练数据和真实使用场景的数据分布不同（白底照片 vs 家里随手拍），模型性能会下降。自拍测试集就是用来度量这一点的 |
| **捷径学习（Shortcut Learning）** | 模型学到“错误但有效”的线索。比如 TrashNet 全是白底，模型可能靠背景判断。用 Grad-CAM 可以检查 |
| **EXIF 元数据（EXIF Metadata）** | 手机照片里嵌入的拍摄时间、GPS 位置等信息，涉及隐私，需要删除 |

### 4.9 评估指标（Evaluation Metrics）

设某一类的真正例为 TP（True Positive），假正例为 FP（False Positive），假负例为 FN（False Negative）：

| 指标 | 公式 / 含义 |
|---|---|
| **准确率（Accuracy）** | 预测正确的比例。类别不平衡时会有误导性 |
| **精确率（Precision）** | \(\frac{TP}{TP+FP}\)：预测为该类的样本中，真正属于该类的比例 |
| **召回率（Recall）** | \(\frac{TP}{TP+FN}\)：真正属于该类的样本中，被找出来的比例 |
| **F1 分数（F1 Score）** | \(\frac{2PR}{P+R}\)：精确率和召回率的调和平均 |
| **宏平均 F1（Macro-F1）** | 先算每一类的 F1 再取平均，每类权重相同，不会被样本多的类主导，所以适合不平衡数据 |
| **混淆矩阵（Confusion Matrix）** | 行是真实类别，列是预测类别，能直观看出哪些类之间容易混淆 |
| **覆盖率（Coverage）** | 拒答机制下，模型给出答案（没有拒答）的样本比例。覆盖率和准确率之间存在权衡 |

### 4.10 选择性分类 / 拒答（Selective Classification / Abstention）

- **思想**：允许模型说“我不知道”。把 softmax 最大概率当作置信度（confidence），低于阈值 \(\tau\) 时拒绝回答（Geifman & El-Yaniv 2017）；
- **怎么选 \(\tau\)**：在验证集上从高到低扫描阈值，找到使“被接受样本的准确率 ≥ 95%”的最小 \(\tau\)，然后在测试集上报告覆盖率和准确率；
- **注意**：神经网络的 softmax 概率往往**过度自信（overconfident）**，即校准（calibration）不佳。如果后期发现效果不好，可以加**温度缩放（Temperature Scaling）**来校准。

### 4.11 Grad-CAM（梯度加权类激活映射）

- **全称**：Gradient-weighted Class Activation Mapping；
- **原理**：用目标类别分数对最后一个卷积层特征图的梯度做加权，生成热力图，显示模型“在看图片的哪里”；
- **本项目用途**：检查模型是看物体本身，还是在“作弊”看背景（白底 → 来自 TrashNet → 可回收）。可以用 `pytorch-grad-cam` 库实现。

### 4.12 基线模型相关知识

- **HOG（方向梯度直方图，Histogram of Oriented Gradients）**：
  1. 计算每个像素的梯度方向和大小；
  2. 把图分成 cell（16×16 像素），每个 cell 统计 9 个方向的梯度直方图；
  3. 相邻 2×2 个 cell 组成一个 block 做归一化，block 以 1 个 cell 为步长滑动；
  4. 维度计算：128÷16 = 8，每边 8 个 cell，所以每边有 7 个 block；总维度 \(7\times7\times(2\times2)\times9 = 1764\)；
  5. HOG 主要描述**形状和边缘**；
- **HSV 颜色直方图（HSV Color Histogram）**：HSV 是色相（Hue）、饱和度（Saturation）、明度（Value）三个通道，每个通道分 8 个区间，共 \(8^3=512\) 维，描述**颜色分布**（比如有机物多为绿色和棕色）；
- **标准化（Standardization）**：每一维减均值、除以标准差，让不同量纲的特征可比，这对 SVM 很重要；
- **主成分分析（Principal Component Analysis, PCA）**：线性降维，保留方差最大的方向，从 2276 维降到 256 维，可以加速训练并去噪；
- **支持向量机（Support Vector Machine, SVM）**：寻找最大间隔（maximum margin）的分类超平面；
  - **RBF 核（Radial Basis Function Kernel，高斯核）**：\(K(\mathbf{x},\mathbf{x}')=\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)\)，把数据隐式映射到高维空间，从而处理非线性边界；
  - **超参数 \(C\)**：错分惩罚，越大越不容忍错误，也越容易过拟合；
  - **超参数 \(\gamma\)**：核宽度的倒数，越大决策边界越“弯曲”；
- **网格搜索（Grid Search）**：遍历所有超参数组合，在验证集上选最好的一组；
- **SIFT（尺度不变特征变换，Scale-Invariant Feature Transform）**：TrashNet 论文使用的局部特征，和 HOG 一样属于手工特征（hand-crafted features）。

### 4.13 伦理相关概念

| 概念 | 在本项目中的体现 |
|---|---|
| **隐私（Privacy）** | Flickr 图片中的人和车牌、信件上的地址、照片的 EXIF |
| **许可证（License）** | TACO 是 CC BY 4.0（可自由使用，但必须署名，attribution） |
| **数据偏见（Data Bias）** | 训练数据的地域分布和真实使用场景不一致 |
| **适用范围（Scope）** | 规则因城市而异、会随时间变化；危险品不在处理范围内 |
| **误用（Misuse）** | 不能用于监控或处罚居民 |
| **以人为本的设计（Human-in-the-loop）** | 不确定时把决定交还给用户 |

---

## 五、模型代码草图（帮助理解）

下面是 proposal 里描述的模型的 PyTorch 写法，后面写代码时可以作为起点：

```python
import torch
import torch.nn as nn
from torchvision import models


class BinSight(nn.Module):
    def __init__(self, num_bins=3, num_materials=7, dropout=0.3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()          # 去掉原来的 1000 类输出层，输出 512 维特征
        self.backbone = backbone
        self.shared = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout)
        )
        self.bin_head = nn.Linear(256, num_bins)
        self.material_head = nn.Linear(256, num_materials)

    def forward(self, x):
        h = self.shared(self.backbone(x))
        return self.bin_head(h), self.material_head(h)   # 返回 logits，不加 softmax


model = BinSight()

# 阶段 1：冻结骨干网络
for p in model.backbone.parameters():
    p.requires_grad = False

# 阶段 2：解冻 layer3、layer4
for name, p in model.backbone.named_parameters():
    if name.startswith(("layer3", "layer4")):
        p.requires_grad = True

# 多任务损失
bin_weights = torch.tensor([1.0, 1.0, 1.0])       # 换成按训练集统计的逆频率权重
ce_bin = nn.CrossEntropyLoss(weight=bin_weights)
ce_mat = nn.CrossEntropyLoss()
lam = 0.5

def loss_fn(bin_logits, mat_logits, bin_y, mat_y):
    return ce_bin(bin_logits, bin_y) + lam * ce_mat(mat_logits, mat_y)

# 推理时的拒答
@torch.no_grad()
def predict(x, tau):
    model.eval()
    bin_logits, _ = model(x)
    probs = bin_logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
    pred[conf < tau] = -1                          # -1 表示“不确定”
    return pred
```

---

## 六、提交前必须核对的事项

1. **个人信息**：在 `proposal.tex` 的 `\author{}` 里填名字、学号、邮箱；
2. **GitHub 链接**：建好仓库后替换 `YOUR-USERNAME`，仓库要设为公开（或私有并分享给助教）；
3. **预估数字**：TACO 约 3000 张裁剪、Kaggle 清洗后约 800 张、自拍约 400 张都是**预计值**。proposal 里已经用 “expected”、“target”、“about” 表述，后续以实际数字为准，在 Progress Report 里更新；
4. **多伦多规则**：标签映射表要以你实际查询 Waste Wizard 时的结果为准（规则会变），映射表放进仓库；
5. **在 Overleaf 上重新编译**：上传 `proposal.tex`、`references.bib`、`aps360.sty`，编译器选 pdfLaTeX，确认正文是 2 页；
6. **通读一遍英文**：确认每句话你都理解，答辩或面试时能讲清楚。

---

## 七、助教可能会问的问题

| 问题 | 回答思路 |
|---|---|
| 垃圾分类往届已经做过了，你有什么不同？ | 按市政规则分桶而不是按材质分类；多数据源合并和标签统一；多任务；拒答机制；在真实自拍数据上测试 |
| 为什么不直接用材质预测再查表得到垃圾桶？ | 垃圾桶不是材质的函数（纸巾、油污纸盒等），而且查表无法利用图像中的其他线索。可以把“材质 + 查表”作为一个消融对比 |
| 为什么用 ResNet-18 而不是更新的模型？ | 数据量小，模型小不容易过拟合，训练快、方便做多轮实验。时间允许的话可以加 EfficientNet / MobileNetV3 对比 |
| 自拍照片只有 400 张，结论可靠吗？ | 它只用于测试、不参与训练，能真实反映泛化能力；报告时可以加上置信区间（confidence interval） |
| 如果 TACO 裁剪后样本太少怎么办？ | 降低最小尺寸阈值、合并相近类别，或者多拍一些照片补充（加入训练的部分要和测试集严格分开） |
| 模型靠背景作弊怎么办？ | 用 Grad-CAM 检查；在增强中加入随机背景替换；在自拍测试集上验证 |
