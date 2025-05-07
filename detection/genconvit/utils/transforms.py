from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from utils.augment import Aug
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class GenConViTTransforms():
    def __init__(self, img_size=224, aug=False):
        if aug:
            self.transforms = Compose([
                Aug(),
                Resize((img_size, img_size), antialias=True),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
        else:
            self.transforms = Compose([
                Resize((img_size, img_size), antialias=True),
                ToTensor(),
                Normalize(mean=mean, std=std)
            ])
    def __call__(self, img):
        return self.transforms(img)