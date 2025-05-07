import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm

class Trainer():
    def __init__(self, 
                 model: nn.Module, 
                 dataloader: DataLoader, 
                 lr: float=1e-4, 
                 epochs: int=10,
                 num_warmup_steps: int=100,
                 device: torch.device=torch.device('cpu')):
        
        self.model = model
        self.dataloader = dataloader
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=len(self.dataloader)*epochs)
        self.epochs = epochs
        self.cur_epoch = 1

        self.device = device
        self.model.to(device)
    
    def loss(self, pred, label):
        return torch.nn.functional.cross_entropy(pred, label)
    
    def train(self):
        self.model.train()
        progress = tqdm(self.dataloader, desc=f'[Epoch {self.cur_epoch}/{self.epochs}]')

        for data in progress:
            self.optimizer.zero_grad()
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            pred = self.model(img)
            loss = self.loss(pred, label)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            progress.set_postfix({'loss': loss.item()})
        
        progress.close()
        self.cur_epoch += 1
            