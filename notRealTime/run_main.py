import time
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
import copy
from utils import *
from StyleTransfer_model import get_style_model_and_losses

# 运行风格迁移
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




if __name__ == "__main__":
    style = InputImg('StarryNight')
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
    output_name = f'notRT_{content.name}({style.name}).png'
    output_path = f'../output/{output_name}'
    output_pil.save(output_path)
    print(f"Saved {output_name}")
    plt.ioff();
    plt.show()