import json
from pathlib import Path
from matplotlib import  transforms
from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image

DATASET_FILE_NAME = 'dataset_training_actual.json'

class TripletsDataset(Dataset):
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
