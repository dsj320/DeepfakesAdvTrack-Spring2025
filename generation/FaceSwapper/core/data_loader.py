"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
from munch import Munch
from PIL import Image
import random
import glob
import copy
import torch
from torch.utils import data
from torchvision import transforms

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames
class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None
    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.samples)

class TrainFaceDataSet(data.Dataset):
    def __init__(self, data_root, transform=None, transform_seg=None,name=''):
        self.data_root = data_root
        self.lm_image_path = self.data_root + f'/{name}_lm_images/'
        self.mask_image_path = self.data_root + f'/{name}_mask_images/'
        self.data_root=self.data_root+'/'+name
        #import pdb;pdb.set_trace()
        print(self.data_root)
        print(self.lm_image_path)
        print(self.mask_image_path)
        # 获取所有图片名（只取文件名，不带路径）
        self.samples = [Path(x).name for x in listdir(self.data_root)]
        self.samples.sort()
        self.transform = transform
        self.transform_seg = transform_seg
        print(f"数据集长度：{len(self.samples)}")

    def __getitem__(self, idx):
        # 固定源脸
        src_name = self.samples[idx]
        
        # 随机选择目标脸（确保与源脸不同）
        ref_idx = idx
        while ref_idx == idx:
            ref_idx = random.randint(0, len(self.samples) - 1)
        ref_name = self.samples[ref_idx]

        # 拼接路径
        source_image_path = f"{self.data_root}/{src_name}"
        source_lm_image_path = f"{self.lm_image_path}{src_name}"
        source_mask_image_path = f"{self.mask_image_path}{src_name}"

        reference_image_path = f"{self.data_root}/{ref_name}"
        reference_lm_image_path = f"{self.lm_image_path}{ref_name}"
        reference_mask_image_path = f"{self.mask_image_path}{ref_name}"

        # 检查文件是否存在（可选，调试用）
        for p in [source_image_path, source_lm_image_path, source_mask_image_path,
                  reference_image_path, reference_lm_image_path, reference_mask_image_path]:
            if not Path(p).exists():
                print(f"Warning: {p} not found!")

        # 读取图片
        source_image = Image.open(source_image_path).convert('RGB')
        source_lm_image = Image.open(source_lm_image_path).convert('RGB')
        source_mask_image = Image.open(source_mask_image_path).convert('L')

        reference_image = Image.open(reference_image_path).convert('RGB')
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('L')

        # 变换
        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
        if self.transform_seg is not None:
            source_mask_image = self.transform_seg(source_mask_image)
            reference_mask_image = self.transform_seg(reference_mask_image)

        outputs = dict(
            src=source_image, ref=reference_image,
            src_lm=source_lm_image, ref_lm=reference_lm_image,
            src_mask=1 - source_mask_image, ref_mask=1 - reference_mask_image,
            src_name=src_name, ref_name=ref_name
        )
        return outputs

    def __len__(self):
        return len(self.samples)

class TestFaceDataSet(data.Dataset):
    def __init__(self, data_root, test_img_list, transform=None, transform_seg=None):
        self.data_root = data_root.rstrip('/')
        self.lm_image_path = self.data_root + '_lm_images/'
        self.mask_image_path = self.data_root + '_mask_images/'
        self.biseg_parsing_path = self.data_root + '_parsing_images/'

        self.source_dataset = []
        self.reference_dataset = []
        with open(test_img_list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                self.source_dataset.append(parts[0].strip())
                self.reference_dataset.append(parts[1].strip())

        self.transform = transform
        self.transform_seg = transform_seg

    def __getitem__(self, item):
        # 拼接路径
        source_image_path = f"{self.data_root}/{self.source_dataset[item]}"
        source_lm_image_path = f"{self.lm_image_path}{self.source_dataset[item]}"
        source_mask_image_path = f"{self.mask_image_path}{self.source_dataset[item]}"
        source_parsing_image_path = f"{self.biseg_parsing_path}{self.source_dataset[item]}"

        reference_image_path = f"{self.data_root}/{self.reference_dataset[item]}"
        reference_lm_image_path = f"{self.lm_image_path}{self.reference_dataset[item]}"
        reference_mask_image_path = f"{self.mask_image_path}{self.reference_dataset[item]}"
        reference_parsing_image_path = f"{self.biseg_parsing_path}{self.reference_dataset[item]}"

        # 检查文件是否存在
        for p in [source_image_path, source_lm_image_path, source_mask_image_path, source_parsing_image_path,
                  reference_image_path, reference_lm_image_path, reference_mask_image_path, reference_parsing_image_path]:
            if not Path(p).exists():
                print(f"Warning: {p} not found!")

        # 读取图片
        source_image = Image.open(source_image_path).convert('RGB')
        source_lm_image = Image.open(source_lm_image_path).convert('RGB')
        source_mask_image = Image.open(source_mask_image_path).convert('L')
        source_parsing_image = Image.open(source_parsing_image_path).convert('L')

        reference_image = Image.open(reference_image_path).convert('RGB')
        reference_lm_image = Image.open(reference_lm_image_path).convert('RGB')
        reference_mask_image = Image.open(reference_mask_image_path).convert('L')
        reference_parsing = Image.open(reference_parsing_image_path).convert('L')

        if self.transform is not None:
            source_image = self.transform(source_image)
            source_lm_image = self.transform(source_lm_image)
            source_mask_image = self.transform_seg(source_mask_image)
            source_parsing = self.transform_seg(source_parsing_image)
            reference_image = self.transform(reference_image)
            reference_lm_image = self.transform(reference_lm_image)
            reference_mask_image = self.transform_seg(reference_mask_image)
            reference_parsing = self.transform_seg(reference_parsing)

        outputs = dict(
            src=source_image, ref=reference_image,
            src_lm=source_lm_image, ref_lm=reference_lm_image,
            src_mask=1 - source_mask_image, ref_mask=1 - reference_mask_image,
            src_parsing=source_parsing, ref_parsing=reference_parsing,
            src_name=self.source_dataset[item], ref_name=self.reference_dataset[item]
        )
        return outputs

    def __len__(self):
        return len(self.source_dataset)

def get_train_loader(root, img_size=256,
                     batch_size=8, num_workers=4,name=''):
    print('Preparing dataLoader to fetch images during the training phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])
    train_dataset = TrainFaceDataSet(root, transform, transform_seg,name)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
    return train_loader

def get_test_loader(root, test_img_list, img_size=256,
                     batch_size=8, num_workers=4):
    print('Preparing dataLoader to fetch images during the testing phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])
    transform_seg = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
    ])
   
    test_dataset = TestFaceDataSet(root, test_img_list, transform, transform_seg)

    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, drop_last=False)
    return test_loader

class InputFetcher:
    def __init__(self, loader, mode=''):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    def _fetch_inputs(self):
        try:
            inputs_data = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            inputs_data= next(self.iter)
        return inputs_data
    def __next__(self):
        t_inputs = self._fetch_inputs()
        inputs = Munch(src=t_inputs['src'], tar=t_inputs['ref'], src_lm=t_inputs['src_lm'],
                       tar_lm=t_inputs['ref_lm'], src_mask=t_inputs['src_mask'], tar_mask=t_inputs['ref_mask'])
        if self.mode=='train':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()})
        elif self.mode=='test':
            inputs = Munch({k: t.to(self.device) for k, t in inputs.items()}, src_parsing=t_inputs['src_parsing'].to(self.device),
                           tar_parsing=t_inputs['ref_parsing'].to(self.device), src_name=t_inputs['src_name'],tar_name=t_inputs['ref_name'])
        return inputs

