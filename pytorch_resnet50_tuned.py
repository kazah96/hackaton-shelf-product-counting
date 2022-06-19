from ast import Sub
import json
import os
from pathlib import Path
import random
from matplotlib import pyplot, transforms
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import torchvision.transforms.functional as F
MODEL_FILE_NAME = 'hackaton_model_1.pth'
DATASET_FILE_NAME = 'dataset_training_actual.json'

# Схема:
# 1. Взять готовую натренированную сеть
# 2. Заморозить все слои кроме последнего слоя
# 3. Обучить на выборке датасета


# model = resnet50(pretrained=True)

# for child in model.children:
#     child.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = pyplot.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = F.to_pil_image(img)
        axs[0, i].imshow(img)
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    pyplot.waitforbuttonpress()


class ByJsonDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.dataset_json_file = Path(
            Path.cwd()) / DATASET_FILE_NAME

        with open(self.dataset_json_file, mode='r') as file:
            self.samples = json.loads(file.read())

    def __getitem__(self, index):
        return [self.get_torch_image(file) for file in self.samples[index]]

    def __len__(self):
        return len(self.samples)

    def get_torch_image(self, name):
        input_image = Image.open(name).convert('RGB')

        preprocess = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225])
        ])

        return preprocess(input_image)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


model = resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1000)

model.to(device)
loss_function = nn.TripletMarginLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

triple_net = TripletNet(model)

model.load_state_dict(torch.load('models/' + MODEL_FILE_NAME))
triple_net.to(device)


def train(data_loader):
    model.train()

    size = len(data_loader)

    for batch, d in enumerate(data_loader):
        anchor_image_batch, positive_image_batch, negative_image_batch = d

        # Showin images
        # show([anchor_image_batch[0], positive_image_batch[0], negative_image_batch[0]])

        anchor_image_batch = anchor_image_batch.to(device)
        positive_image_batch = positive_image_batch.to(device)
        negative_image_batch = negative_image_batch.to(device)

        pred = triple_net.forward(
            anchor_image_batch, positive_image_batch, negative_image_batch)
        loss = loss_function(*pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"BATCH {batch} of {size}")


def get_rand_array(count, max_number):
    return [random.randint(0, max_number-1) for i in range(count)]


def run():
    epochs = 20
    BATCH_SIZE = 32
    MAX_ITERATIONS = BATCH_SIZE * 1000

    dataset = ByJsonDataset()

    dataset_size = len(dataset)

    for t in range(epochs):
        train_subset = Subset(dataset, get_rand_array(
            MAX_ITERATIONS, dataset_size))

        train_dataloader = DataLoader(
            train_subset, batch_size=BATCH_SIZE,  pin_memory=True, shuffle=True, num_workers=4)

        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader)
        torch.save(model.state_dict(), 'models/' + MODEL_FILE_NAME)

    print("Done!")


if __name__ == "__main__":
    run()
