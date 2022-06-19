import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50
import torchvision.transforms.functional as F

from triple_model import TripletNet
from datasets import ByJsonDataset
from utils import get_rand_array

MODEL_FILE_NAME = 'hackaton_model_2.pth'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class Trainer():
    def __init__(self, epochs=20, batch_size=20, iterations=100000) -> None:
        self.triple_net = TripletNet(MODEL_FILE_NAME, load=False)
        self.triple_net.to(device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = iterations  

        self.loss_function = nn.TripletMarginWithDistanceLoss()
        # self.loss_function = nn.TripletMarginLoss()
        self.optimizer = torch.optim.SGD(
            self.triple_net.embedding_net.fc.parameters(), lr=0.001, momentum=0.9)

    def train(self, data_loader):
        self.triple_net.train()

        size = len(data_loader)

        for batch, d in enumerate(data_loader):
            anchor_image_batch, positive_image_batch, negative_image_batch = d

            # Showin images
            # show([anchor_image_batch[0], positive_image_batch[0], negative_image_batch[0]])

            anchor_image_batch = anchor_image_batch.to(device)
            positive_image_batch = positive_image_batch.to(device)
            negative_image_batch = negative_image_batch.to(device)

            pred = self.triple_net.forward(
                anchor_image_batch, positive_image_batch, negative_image_batch)

            loss = self.loss_function(*pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"BATCH {batch} of {size}")

    def run(self):
        MAX_ITERATIONS = self.batch_size * self.iterations
        dataset = ByJsonDataset()
        dataset_size = len(dataset)

        for t in range(self.epochs):
            train_subset = Subset(dataset, get_rand_array(
                MAX_ITERATIONS, dataset_size))

            train_dataloader = DataLoader(
                train_subset, batch_size=self.batch_size,  pin_memory=True, shuffle=True, num_workers=4)

            print(f"Epoch {t+1}\n-------------------------------")
            self.train(train_dataloader)
            self.triple_net.save_to_file()

        print("Done!")


if __name__ == "__main__":
    trainer = Trainer(iterations=4000)
    trainer.run()
