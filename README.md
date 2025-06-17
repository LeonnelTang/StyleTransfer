# 基于卷积神经网络的图像风格迁移

## 一、理论介绍：基于卷积神经网络的图像风格迁移

图像风格迁移（Neural Style Transfer, NST）是一项将一张图像的“内容”与另一张图像的“风格”相融合的技术，其核心目标是生成一张图像，使其在语义上与内容图像相似，在视觉风格上与风格图像一致。

### 1、基本思想

Gatys 等人在 2015 年首次提出使用卷积神经网络（CNN）进行图像风格迁移的方法。他们利用了深层 CNN 在图像表达方面的强大能力，尤其是不同网络层对图像内容和风格的分离表示能力：

- **内容表示（Content Representation）**：中层 CNN 特征图能较好保留图像的语义结构。
- **风格表示（Style Representation）**：通过统计特征图之间的相关性（Gram矩阵）来表示图像风格。

最终目标是生成一张图像，使其内容特征与内容图像一致、风格特征与风格图像一致。

------

### 2、损失函数设计

1. **内容损失（Content Loss）**：

内容图像 $C$ 和生成图像 $G$ 在特定层的特征图记为 $F^l_C$ 和 $F^l_G$，则内容损失定义为：$\mathcal{L}_{\text{content}}(C, G, l) = \frac{1}{2} \|F^l_C - F^l_G\|^2$

2. **风格损失（Style Loss）**：

风格图像 $S$ 和生成图像 $G$ 在层 $l$ 的 Gram 矩阵分别为 $G^l_S$ 和 $G^l_G$，Gram 矩阵计算如下：$G^l_{ij} = \sum_k F^l_{ik} F^l_{jk}$

风格损失则为：$\mathcal{L}_{\text{style}}(S, G) = \sum_l w_l \cdot \|G^l_S - G^l_G\|^2$

其中 $w_l$ 是不同层次的权重。

**总损失函数（Total Loss）**：$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{content}} + \beta \cdot \mathcal{L}_{\text{style}}$，其中：

- $\alpha$ 是内容损失的权重；
- $\beta$ 是风格损失的权重；
- 通常取 $\beta \gg \alpha$。

------

### 3、模型结构

本实验采用 PyTorch 提供的预训练 VGG19 网络作为特征提取器，具体做法如下：

1. 截取 VGG19 的前若干层构造特征提取模型；
2. 插入自定义的内容损失层和风格损失层；
3. 冻结 VGG 网络参数，仅对输入图像进行梯度更新；
4. 使用 LBFGS 优化器迭代优化输入图像，使损失函数最小化。

这种方法不需要训练网络，而是通过优化输入图像本身，从而达到风格迁移的目的。


## 二、实验结果分析

### 1、实验设置

本实验选取一张内容图、两张风格图进行风格迁移：

- **内容图像（Content Image）**：输入为 `chicago.jpg`，一张芝加哥城市风景图；
- **风格图像（Style Image）**：选择 `StarryNight.jpg` 和 `mosaic.jpg`，分别是《星夜》（代表梵高风格的艺术作品）和《日出·印象》（代表莫奈印象派风格作品）；

| 参数                          | 值                                        |
| ----------------------------- | ----------------------------------------- |
| 图像尺寸                      | 内容图按短边512缩放，风格图尺寸适应内容图 |
| 优化器                        | L-BFGS                                    |
| 内容权重 α （content_weight） | 1                                         |
| 风格权重 β （style_weight）   | 1e6                                       |
| 迭代次数（num_steps）         | 500                                       |

### 2、迁移效果

| 图像类型 | 风格1                                                        | 风格2                                                        |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 内容图像 | <img src="img_data\chicago.jpg" alt="chicago" width = 400 /> | <img src="img_data\chicago.jpg" alt="chicago" width = 400/> |
| 风格图像 | <img src="img_data\StarryNight.jpg" alt="StarryNight" width = 400 /> | <img src="img_data\mosaic.jpg" alt="mosaic" width = 350 /> |
| 输出图像 | <img src="output\notRT_chicago(StarryNight).png" alt="notRT_chicago(StarryNight)" width = 400 /> | <img src="output\notRT_chicago(mosaic).png" alt="notRT_chicago(mosaic)" width = 400 /> |

------

### 3、性能分析

#### （1）loss变化

以梵高《星夜》的风格迁移过程为例，迁移过程中每 50 步记录一次loss变化：

```
Transferring content image chicago to the style of StarryNight...
run 50: Style Loss : 191.882370 Content Loss: 37.860607
run 100: Style Loss : 52.532497 Content Loss: 36.277004
run 150: Style Loss : 13.773139 Content Loss: 34.227951
run 200: Style Loss : 6.743725 Content Loss: 31.228165
run 250: Style Loss : 4.546108 Content Loss: 28.683067
run 300: Style Loss : 3.352822 Content Loss: 26.722250
run 350: Style Loss : 2.514446 Content Loss: 25.450394
run 400: Style Loss : 1.965243 Content Loss: 24.532776
run 450: Style Loss : 1.556310 Content Loss: 23.818813
run 500: Style Loss : 1.340904 Content Loss: 23.339428
Style Transfer completed in 47.96s

```

<img src="output\loss_trend.png" alt="loss_trend" style="zoom: 67%;" />

- **风格损失**从初始的约 192 快速下降到最终的约 1.34，下降速度较快，在前 200 步内已经大幅收敛，表明模型在早期就能够有效捕捉风格特征。**内容损失**从约 38 稳定下降到约 23，收敛趋势较慢但更加平稳，说明保持内容结构较为稳定。

#### （2）时间消耗

在设备：`NVIDIA CUDA GPU`（4090）上运行，运行时间： **47.96 秒**（含前向、反向传播与优化），符合预期。
