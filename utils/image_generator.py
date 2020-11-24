from random import sample, shuffle

import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd

def imshow(img):
    f, ax = plt.subplots(1, 1)
    ax.imshow(img)
    plt.show()

class ImageGenerator:
    def __init__(self, annotations_path, batch_size, input_size=(512, 512), is_torch=True):
        self.annotations = pd.read_csv(annotations_path)
        self.batch_size = batch_size
        self.val_batches = self.create_val_batches()
        self.input_size = input_size
        self.is_torch = is_torch

    def create_train_batches(self):
        num_samples = len(self.annotations)
        self.num_batches = num_samples // self.batch_size
        remainder = num_samples % self.batch_size
        if remainder:
            self.num_batches += 1
        difference = self.batch_size - remainder
        indices = list(range(num_samples))

        duplicates = sample(indices, difference)
        indices = indices + duplicates
        shuffle(indices)
        batched = [indices[i:i+self.batch_size] for i in range(self.num_batches)]
        return batched

    def create_val_batches(self):
        num_samples = len(self.annotations)
        self.num_batches = num_samples // self.batch_size
        remainder = num_samples % self.batch_size
        batches = [(i * self.batch_size, (i + 1) * self.batch_size) for i in range(self.num_batches)]
        if remainder:
            s = self.num_batches * self.batch_size
            e = s + remainder
            batches.append((s, e))
        return batches

    def resize(self, img):
        height, width = img.shape
        pad_diff = height - width

        if pad_diff < 0:
            dst_img = np.zeros((width, width), dtype=np.uint8)
            dst_img[-pad_diff//2 : pad_diff //2, :] = img
        else:
            dst_img = np.zeros((height, height), dtype=np.uint8)
            dst_img[:, pad_diff // 2 : -pad_diff // 2] = img

        dst_img = cv2.resize(dst_img, self.input_size)
        return dst_img

    def read_raw(self, path, is_mask=False):
        img = io.imread(path, plugin='simpleitk').squeeze()
        img = self.resize(img)

        if is_mask:
            # fixme, debug this properly make sure its happening properly
            mask = np.zeros((3, *self.input_size), dtype=np.uint8)
            mask[0, img == 1] = 255
            mask[1, img == 2] = 255
            mask[2, img == 3] = 255
            img = mask

        elif self.is_torch:
            img = np.expand_dims(img, 0)

        return img
    def imread_batch(self, batch):
        batch_list = batch.values.tolist()
        images = []
        masks = []
        for img_path, mask_path in batch_list:
            img = self.read_raw(img_path)
            mask = self.read_raw(mask_path, is_mask=True)
            images.append(img)
            masks.append(mask)
        images = np.array(images)
        masks = np.array(masks)
        return images, masks

    def train_generator(self):
        while True:
            print('beginning batch')
            batches = self.create_train_batches()
            for batch_indices in batches:
                batch = self.annotations.loc[batch_indices]
                images, masks = self.imread_batch(batch)
                yield images, masks

    def val_generator(self):
        for batch_indices in self.val_batches:
            batch = self.annotations.loc[batch_indices]
            images, masks = self.imread_batch(batch)
            yield images, masks

    def __len__(self):
        return self.num_batches

if __name__ == '__main__':
    from pathlib import Path

    data_path = Path(__file__).joinpath('..', '..', '..', 'data').resolve()
    img_gen = ImageGenerator(data_path.joinpath('train_annotations.csv').resolve(), 8)
    gen = img_gen.train_generator()
    for i in gen:
        None