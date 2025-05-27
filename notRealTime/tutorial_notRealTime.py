#!/usr/bin/env python
# neural_style_tutorial.py
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import copy

# 1. 选择运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# 2. 图像加载与预处理
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 3. 图像显示工具
unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)


# 4. 内容损失模块
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# 5. 风格损失及 Gram 矩阵
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std  = std.view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# 7. 构建包含风格与内容损失的模型
content_layers_default = ['conv_4']
style_layers_default   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)
    content_losses = []
    style_losses   = []
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1; name = f"conv_{i}"
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
            style_loss     = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # 裁剪掉后续不必要的层
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j+1]

    return model, style_losses, content_losses

# 10. 运行风格迁移
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       num_steps=500, style_weight=1e6, content_weight=1):
    print("Building the style transfer model...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)
    input_img.requires_grad_(True)
    optimizer = optim.LBFGS([input_img])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score   = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            style_score  *= style_weight
            content_score*= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 100 == 0:
                print(f"run {run[0]}: Style Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}")
                # plt.figure(); current_title = f"run = {run[0]}"
                # imshow(input_img, title=current_title)
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

if __name__ == "__main__":

    style = 'VanGogh'
    content = 'hyy'
    style_weight = 1e6
    content_weight = 1
    num_steps = 500

    style_path = '../img_data/'+style+'.jpg'
    content_path = '../img_data/'+content+'.jpg'
    style_img = image_loader(style_path)
    content_img = image_loader(content_path)
    # 适配尺寸
    style_img = torch.nn.functional.interpolate(
        style_img, size=content_img.shape[-2:], mode='bilinear', align_corners=False)
    assert style_img.size() == content_img.size()

    input_img = content_img.clone()
    plt.figure(); imshow(style_img, title='Style Image')
    plt.figure(); imshow(content_img, title='Content Image')

    # 6. 导入预训练 VGG 并添加归一化模块
    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    start=time.time()
    output = run_style_transfer(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        content_img, style_img, input_img,
        num_steps=num_steps, style_weight=style_weight, content_weight=content_weight
    )
    end=time.time()
    print(f"Style Transfer completed in {end-start:.2f}s")

    plt.figure(); imshow(output, title='Output Image')

    # 保存
    output_pil = unloader(output.cpu().clone().squeeze(0))

    output_name = f'notRT_{content}({style}).png'
    output_path = f'../output/{output_name}'
    output_pil.save(output_path)
    print(f"Saved {output_name}")
    plt.ioff(); plt.show()