from ast import Sub
import json
import csv
import math
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

device = "cpu"
print(f"Using {device} device")


def get_image(name):
    return Image.open(name).convert('RGB')


preprocess = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class TripletNet(nn.Module):
    def __init__(self, model_filename: str):
        super(TripletNet, self).__init__()
        self.model_filename = model_filename
        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1000)
        model.to(device)

        self.embedding_net = model

        self.to(device)

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    def load_from_file(self):
        self.embedding_net.load_state_dict(torch.load(self.model_filename))


def get_rand_array(count, max_number):
    return [random.randint(0, max_number) for i in range(count)]


class Tester():
    def __init__(self, dataset_pathname, model_filename, test_indexes=None) -> None:
        self.net = TripletNet(model_filename)
        self.net.eval()
        self.test_indexes = test_indexes

        self.dataset_path = Path(Path.cwd()) / dataset_pathname
        self.shelves_path = self.dataset_path / 'shelves'

        with open(self.dataset_path / 'shelves_info.json', mode='r') as file:
            inp = file.read()
            self.shelves_info = json.loads(inp)

    def get_cosine_for_two_images(self, img1, img2):
        vec_original, vec_positive, vec_negative = self.net.forward(
            img1, img2, img2)

        similarity = torch.nn.CosineSimilarity(dim=1)

        g = similarity(vec_original, vec_positive)

        return g.item()

    def shelf_product_count(self, shelf_name):
        return len(self.shelves_info[shelf_name]['bboxes'])

    def iterate_over_shelf(self, shelf_name: str):
        shelf_image = get_image(self.shelves_path / shelf_name)

        bboxes = self.shelves_info[shelf_name]['bboxes']

        for box in bboxes:
            left = box[0]
            right = box[2]
            upper = box[1]
            lower = box[3]
            cropped_part = shelf_image.crop((left, upper, right, lower))
            yield cropped_part

    def get_over_test_dataset(self, visualise=False):
        THRESHOLD = 0.80
        queries_path = self.dataset_path / 'queries'
        requests_file_path = self.dataset_path / 'requests.csv'
        output_result_file_path = self.dataset_path / 'output.csv'

        data = []

        with open(requests_file_path, mode='r') as requests_file:
            reader = csv.reader(requests_file)
            for row in reader:
                data.append([row[0], row[1], 0])

        if self.test_indexes:
            data = [data[i] for i in self.test_indexes]

        for data_index, row in enumerate(data[:1]):
            shelf, query, count = row

            ref_image = get_image(queries_path / query)

            side = math.ceil(
                math.sqrt(self.shelf_product_count(shelf)))
            fig, m_axs = pyplot.subplots(side, side, figsize=(16, 16))

            if visualise:
                m_axs[0, 0].imshow(ref_image)
                m_axs[0, 0].axis('off')
                m_axs[0, 0].set_title("REF IMAGE")

            ref_image = preprocess(
                ref_image).unsqueeze(0)

            count = 0

            for image_idx, image_from_shelf in enumerate(self.iterate_over_shelf(shelf)):
                # pyplot.imshow(image_from_shelf)
                # pyplot.waitforbuttonpress()

                cosine_distance = self.get_cosine_for_two_images(
                    ref_image, preprocess(image_from_shelf).unsqueeze(0))
                is_similar = cosine_distance >= THRESHOLD
                if is_similar:
                    count += 1

                if visualise:

                    x = (image_idx+1) % side
                    y = (image_idx+1) // side

                    m_axs[x, y].imshow(image_from_shelf)
                    m_axs[x, y].axis('off')
                    color = 'green' if is_similar else 'red'
                    m_axs[x, y].set_title("{:.4f}".format(
                        cosine_distance), color=color)

            data[data_index][2] = count

            if visualise:
                pyplot.waitforbuttonpress()

        with open(output_result_file_path, mode='w') as result_file:
            writer = csv.writer(result_file)

            for record in data:
                writer.writerow(record)


def run():
    MODEL_FILE_NAME = 'hackaton_model_1.pth'

    tester = Tester("datasets/PublicTestSet",
                    MODEL_FILE_NAME, test_indexes=[0])
    tester.get_over_test_dataset(visualise=True)
    # tester.get_over_test_dataset(visualise=False)


if __name__ == "__main__":
    run()
