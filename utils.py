import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA_PATH = 'data'
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test')

CLASS_NAME_TO_ID = {
    'trafficlight': 0,
    'stop': 1,
    'speedlimit': 2,
    'crosswalk': 3,
}


def get_dataset(path, size):
    images = []
    labels = []

    for class_name, class_id in CLASS_NAME_TO_ID.items():
        class_path = os.path.join(path, class_name)

        for image_path in os.listdir(class_path):
            image_path = os.path.join(class_path, image_path)

            with Image.open(image_path) as image:
                image = image.resize((size, size))
                image = np.array(image)[..., :3]  # only use RGB
                images.append(image)
                labels.append(class_id)

    return (np.array(images), np.array(labels))


def get_train_test_dataset(size):
    train_dataset = get_dataset(TRAIN_DATA_PATH, size)
    test_dataset = get_dataset(TEST_DATA_PATH, size)

    return train_dataset, test_dataset


def show_result(result, images):
    for class_name, class_id in CLASS_NAME_TO_ID.items():
        image_idx = np.where(result == class_id)[0]
        print(f'{class_name}: {image_idx}')

        fig, axes = plt.subplots(
            1,
            len(image_idx),
            figsize=(8, 8),
            sharey=True,
        )

        fig.supylabel(f'{class_name}')

        for j, idx in enumerate(image_idx):
            image = images[idx]
            axes[j].set_yticks([])
            axes[j].get_xaxis().set_visible(False)
            # axes[j].get_yaxis().set_visible(False)
            axes[j].imshow(image)

        plt.show()


def compute_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
