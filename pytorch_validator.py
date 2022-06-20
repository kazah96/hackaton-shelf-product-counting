import json
import csv
import math
from pathlib import Path
from matplotlib import pyplot, transforms
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

from triple_model import TripletNet

device = "cuda"
print(f"Using {device} device")


def get_image(name):
    return Image.open(name).convert('RGB')


preprocess = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])


class Tester():
    def __init__(self, dataset_pathname, model_filename, submission_file_name='output.csv', test_indexes=None) -> None:
        self.submission_file_name = submission_file_name
        self.test_indexes = test_indexes
        self.dataset_path = Path(Path.cwd()) / dataset_pathname
        self.shelves_path = self.dataset_path / 'shelves'
        self.model_filename = model_filename

        self.net = TripletNet(model_filename, load=True)
        self.net.to(device)
        self.net.eval()

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
        THRESHOLD = 0.83
        queries_path = self.dataset_path / 'queries'
        requests_file_path = self.dataset_path / 'requests.csv'
        output_result_file_path = self.dataset_path / self.submission_file_name

        data = []

        with open(requests_file_path, mode='r') as requests_file:
            reader = csv.reader(requests_file)
            for row in reader:
                data.append([row[0], row[1], 0])

        if self.test_indexes:
            data = [data[i] for i in self.test_indexes]

        for data_index, row in enumerate(data):
            shelf, query, count = row
            print(f"shelf: {shelf}, query: {query}")

            ref_image = get_image(queries_path / query)

            side = math.ceil(
                math.sqrt(self.shelf_product_count(shelf) + 1))

            if visualise:
                fig, m_axs = pyplot.subplots(side, side, figsize=(16, 16))
                m_axs[0, 0].imshow(ref_image)
                m_axs[0, 0].axis('off')
                m_axs[0, 0].set_title("REF IMAGE")

            ref_image = preprocess(
                ref_image).unsqueeze(0).to(device)

            for image_idx, image_from_shelf in enumerate(self.iterate_over_shelf(shelf)):
                processed_image = preprocess(
                    image_from_shelf).unsqueeze(0).to(device)
                cosine_distance = self.get_cosine_for_two_images(
                    ref_image, processed_image)
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
                pyplot.savefig(
                    f"results/result_{shelf}_{query}_{self.model_filename}.jpg")
                pyplot.waitforbuttonpress()

        with open(output_result_file_path, mode='a', newline="") as result_file:
            writer = csv.writer(result_file)

            for record in data:
                writer.writerow(record)


def run(test_set_path, output_file, model_file_name, visualize=True, test_indexes=None):
    print(f"Running over ${test_set_path}")
    print(f"Model using ${model_file_name}")

    tester = Tester(test_set_path,
                    model_file_name, submission_file_name=output_file, test_indexes=test_indexes)
    tester.get_over_test_dataset(
        visualise=visualize)
