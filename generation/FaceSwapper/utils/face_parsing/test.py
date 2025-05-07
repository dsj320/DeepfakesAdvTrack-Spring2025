import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import BiSeNet

def evaluate(
    img_dir='xxx/data/test/test',
    parsing_save_dir='xxx/data/test/test_parsing_images',
    ckpt_path='xxx/utils/face-parsing/79999_iter.pth'
):
    os.makedirs(parsing_save_dir, exist_ok=True)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    net.load_state_dict(torch.load(ckpt_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        for image_name in os.listdir(img_dir):
            if not (image_name.endswith('.jpg') or image_name.endswith('.png')):
                continue

            img_path = osp.join(img_dir, image_name)
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize((256, 256), Image.BILINEAR)

            img_tensor = to_tensor(img_resized).unsqueeze(0).cuda()
            out = net(img_tensor)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)

            # ✅ 直接保存 parsing，不变
            save_path = osp.join(parsing_save_dir, image_name)
            cv2.imwrite(save_path, parsing)

            print(f'Processed {image_name}, unique labels: {np.unique(parsing)}')

if __name__ == "__main__":
    evaluate()