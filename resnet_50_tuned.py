import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms.functional as F

class TripletNet(nn.Module):
    def __init__(self, model_filename: str, load=True):
        super(TripletNet, self).__init__()
        self.model_filename = model_filename

        model = resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 200)

        self.embedding_net = model

        if load:
            self.load_from_file()


    def save_to_file(self):
        torch.save(self.embedding_net.state_dict(),
                   'models/' + self.model_filename)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    def load_from_file(self):
        self.embedding_net.load_state_dict(
            torch.load('models/' + self.model_filename))
