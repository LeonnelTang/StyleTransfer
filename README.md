# 实验报告

## 一、算法理论介绍：基于卷积神经网络的图像风格迁移

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

## 二、代码介绍：非实时图像风格迁移的pytorch实现

### 1、图像加载与预处理

#### **图像加载**

```python
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])
```

```python
def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)
```

- 图像被统一调整为固定尺寸
- 使用 `PIL` 加载图像，转为张量并添加 batch 维度。
- 所有图像都转入计算设备

#### **初始化输入的内容图和风格图**

```python
class InputImg():
    def __init__(self, name):
        self.name = name
        self.path = '../img_data/' + self.name + '.jpg'
        self.img = image_loader(self.path)
```

------

### 2、内容损失与风格损失

#### 内容损失（ContentLoss）：

```python
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.loss = None
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input
```

- 比较输入图像与目标图像在特定 CNN 层的特征差异。
- 使用 MSE Loss 衡量相似度。

#### 风格损失（StyleLoss）与 Gram 矩阵：

```python
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.loss = None
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input
```

```python
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)
```

- Gram 矩阵统计特征通道之间的相关性，用于风格表示。
- 风格损失通过比对生成图像与目标风格图像的 Gram 矩阵计算。

------

### 3、构建风格迁移模型

```python
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=None,
                               style_layers=None):
    
    if style_layers is None:
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    if content_layers is None:
        content_layers = ['conv_4']
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)
    content_losses = []
    style_losses = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # 裁剪掉后续不必要的层
    global j
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j + 1]

    return model, style_losses, content_losses

```

- 以 VGG19 为骨架，逐层复制模型，并在指定层插入内容损失和风格损失模块；
- 每插入一个卷积层后判断是否需要加入损失模块；
- 最后截断网络，仅保留包含损失模块之前的部分。

该过程同时返回模型本体、风格损失列表与内容损失列表，供后续优化使用。

------

### 4、图像优化流程

```python
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content, style, initial_img,
                       num_steps=500, style_weight=1e6, content_weight=1):
    print(f"Transferring content image {content.name} to the style of {style.name}...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style.img, content.img)
    initial_img.requires_grad_(True)
    optimizer = optim.LBFGS([initial_img])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                initial_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(initial_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run[0]}: Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
                # plt.figure(); current_title = f"run = {run[0]}"
                # imshow(initial_img, title=current_title)
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        initial_img.clamp_(0, 1)
    return initial_img
```

- 使用 LBFGS 优化器直接优化输入图像的像素；
- 每次迭代中：
  - 先 `clamp` 限制图像像素在 `[0,1]` 范围；
  - 前向传播，计算内容损失与风格损失；
  - 反向传播并更新图像；
  - 每 100 步打印一次损失信息。

------

### 5、主函数

```python
if __name__ == "__main__":
    style = InputImg('mosaic')
    content = InputImg('chicago')

    style_weight = 1e6
    content_weight = 1
    num_steps = 500

    # 适配尺寸
    style.img = torch.nn.functional.interpolate(
        style.img, size=content.img.shape[-2:], mode='bilinear', align_corners=False)
    assert style.img.size() == content.img.size()

    initial_img = content.img.clone()
    plt.figure();imshow(style.img, title='Style Image')
    plt.figure();imshow(content.img, title='Content Image')

    # 导入预训练 VGG 并添加归一化模块
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    start = time.time()
    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content, style, initial_img,
        num_steps=num_steps, style_weight=style_weight, content_weight=content_weight
    )
    end = time.time()
    print(f"Style Transfer completed in {end - start:.2f}s")

    plt.figure();
    imshow(output, title='Output Image')

    # 保存
    output_pil = unloader(output.cpu().clone().squeeze(0))
    output_name = f'notRT_{content}({style}).png'
    output_path = f'../output/{output_name}'
    output_pil.save(output_path)
    print(f"Saved {output_name}")
    plt.ioff();
    plt.show()
```

- 加载风格图与内容图，并调整尺寸匹配；
- 使用 VGG19 提取特征，并执行迁移；
- 显示并保存迁移结果。

## 三、实验结果分析

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
| 内容图像 | <img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\img_data\chicago.jpg" alt="chicago" style="zoom:25%;" /> | <img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\img_data\chicago.jpg" alt="chicago" style="zoom:25%;" /> |
| 风格图像 | <img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\img_data\StarryNight.jpg" alt="StarryNight" style="zoom:15%;" /> | <img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\img_data\mosaic.jpg" alt="mosaic" style="zoom: 33%;" /> |
| 输出图像 | <img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\output\notRT_chicago(StarryNight).png" alt="notRT_chicago(StarryNight)" style="zoom:25%;" /> | <img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\output\notRT_chicago(mosaic).png" alt="notRT_chicago(mosaic)" style="zoom:25%;" /> |

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

<img src="D:\Learning Materials\魏宪智能计算系统\第二次作业\hw2_SrcCode\output\loss_trend.png" alt="loss_trend" style="zoom: 67%;" />

- **风格损失**从初始的约 192 快速下降到最终的约 1.34，下降速度较快，在前 200 步内已经大幅收敛，表明模型在早期就能够有效捕捉风格特征。**内容损失**从约 38 稳定下降到约 23，收敛趋势较慢但更加平稳，说明保持内容结构较为稳定。

#### （2）时间消耗

在设备：`NVIDIA CUDA GPU`（4090）上运行，运行时间： **47.96 秒**（含前向、反向传播与优化），符合预期。

## 四、总结与个人见解

​	本次图像风格迁移实验基于 Gatys 等人提出的经典神经风格迁移方法，通过优化输入图像以同时逼近目标内容特征与风格特征，实现了高质量的图像风格融合。迁移结果较好地保持了内容图像的主体结构，同时风格图像的颜色、笔触、纹理特征得以生动体现，符合审美预期。

​	通过本实验，我深入理解了神经网络中不同层对图像“内容”与“风格”的编码能力，掌握了如何构建自定义损失函数并插入预训练模型中，感受到深度学习在图像创意生成方面的强大潜力。此外，调试图像处理流程（如归一化、图像尺寸匹配、梯度更新）也提升了我对 PyTorch 框架的使用熟练度，为后续研究视觉迁移学习和生成模型打下了良好基础。
