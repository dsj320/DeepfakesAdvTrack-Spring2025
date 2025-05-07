import torch
import torch.nn as nn
from .genconvit_ed import GenConViTED
from .genconvit_vae import GenConViTVAE
import torch.nn.functional as F

class GenConViT(nn.Module):

    def __init__(self, ed_path, vae_path):
        super(GenConViT, self).__init__()

        self.model_ed = GenConViTED()
        self.model_vae = GenConViTVAE()
        #import pdb; pdb.set_trace()

        if ed_path is not None and vae_path is not None:
            self.load_pretrained_weights(ed_path, vae_path)
            print("Pretrained weights loaded successfully.")
    def swap_labels(self):
        self.model_ed.fc2.weight.data = self.model_ed.fc2.weight.data[[1, 0], :]
        self.model_ed.fc2.bias.data = self.model_ed.fc2.bias.data[[1, 0]]
        self.model_vae.fc2.weight.data = self.model_vae.fc2.weight.data[[1, 0], :]
        self.model_vae.fc2.bias.data = self.model_vae.fc2.bias.data[[1, 0]]

    def load_pretrained_weights(self, ed_path, vae_path):
        checkpoint_ed = torch.load(ed_path, map_location=torch.device('cpu'))
        checkpoint_vae = torch.load(vae_path, map_location=torch.device('cpu'))

        if 'state_dict' in checkpoint_ed:
            self.model_ed.load_state_dict(checkpoint_ed['state_dict'])
        else:
            self.model_ed.load_state_dict(checkpoint_ed)

        if 'state_dict' in checkpoint_vae:
            self.model_vae.load_state_dict(checkpoint_vae['state_dict'])
        else:
            self.model_vae.load_state_dict(checkpoint_vae)

        #0 for real, 1 for fake
        self.swap_labels()

        self.model_ed.eval()
        self.model_vae.eval()

    # def forward(self, x):

    #     x1 = self.model_ed(x)
    #     x2,_ = self.model_vae(x)
    #     x = (x1 + x2) / 2
    #     return x
   
    
    def forward(self, x):
        x1 = self.model_ed(x)
        x2,_ = self.model_vae(x)
        x = (x1 + x2) / 2

        if self.training:
            return x
        else:
            x = F.softmax(x, dim=-1)
            return x[..., 1]
