import os
import random
from PIL import Image
from matplotlib import pyplot
from torchvision import transforms

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def get_rand_array(count, max_number):
    return [random.randint(0, max_number-1) for i in range(count)]


def show_query_shelf(shelf, query):
    fig, m_axs = pyplot.subplots(2, 2, figsize=(16, 16))
    img1 = Image.open('datasets/PrivateTestSet/shelves/' + shelf)
    img2 = Image.open('datasets/PrivateTestSet/queries/' + query)

    m_axs[0, 0].imshow(img1)
    m_axs[0, 1].imshow(img2)

    pyplot.waitforbuttonpress()


def get_image(name):
    return Image.open(name).convert('RGB')


preprocess = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225])
])
