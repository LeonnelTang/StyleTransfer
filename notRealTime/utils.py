import torchvision.transforms as transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

# 使用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# 图像加载/预处理
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# 图像显示
unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# gram矩阵
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# 初始化输入的内容图和风格图
class InputImg():
    def __init__(self, name):
        self.name = name
        self.path = '../img_data/' + self.name + '.jpg'
        self.img = image_loader(self.path)