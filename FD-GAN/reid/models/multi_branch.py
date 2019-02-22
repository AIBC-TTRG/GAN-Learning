from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class SiameseNet(nn.Module):
    def __init__(self, base_model, embed_model):
        print('SiameseNet___init__')
        super(SiameseNet, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model

    def forward(self, x1, x2):
        print('SiameseNet__forward')
        x1, x2 = self.base_model(x1), self.base_model(x2)
        if self.embed_model is None:
            return x1, x2
        return x1, x2, self.embed_model(x1, x2)