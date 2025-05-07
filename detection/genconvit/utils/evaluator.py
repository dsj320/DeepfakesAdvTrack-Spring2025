import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

class Evaluator():
    def __init__(self, model: torch.nn.Module, 
                 dataloader: DataLoader, 
                 device: torch.device=torch.device('cpu')):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def evaluate_auc(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        progress = tqdm(self.dataloader, desc='Evaluating')

        for data in progress:
            img, label = data
            img = img.to(self.device)
            label = label.to(self.device)
            #import pdb; pdb.set_trace()
            # logits = self.model(img)
            # probs = torch.softmax(logits, dim=1)[:, 1]  # Assume binary classification, use prob of class 1
            probs=self.model(img)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        return auc
