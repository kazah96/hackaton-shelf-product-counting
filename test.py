
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50
import torchvision.transforms.functional as F

from triple_model import TripletNet
from datasets import TripletsDataset
from utils import get_rand_array

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()

class ContrastiveLoss(nn.Module):
  def __init__(self, m=2.0):
    super(ContrastiveLoss, self).__init__()  # pre 3.3 syntax
    self.m = m  # margin or radius

  def forward(self, y1, y2, d=0):
    # d = 0 means y1 and y2 are supposed to be same
    # d = 1 means y1 and y2 are supposed to be different
    
    euc_dist = nn.functional.pairwise_distance(y1, y2)

    if d == 0:
      return mean(pow(euc_dist, 2))  # distance squared
    else:  # d == 1
      delta = self.m - euc_dist  # sort of reverse distance
      delta = T.clamp(delta, min=0.0, max=None)
      return T.mean(T.pow(delta, 2))  # mean over all rows
    