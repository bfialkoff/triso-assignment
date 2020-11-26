from random import sample, shuffle

from torch import from_numpy
import cv2
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd

def imshow_img_pair(img, mask):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()

class ImageGenerator:
    def __init__(self, annotations_path, batch_size, input_shape=(512, 512), num_classes=3):
        self.annotations = pd.read_csv(annotations_path)
        self.batch_size = batch_size
        self.val_batches = self.create_val_batches()
        self.input_shape = input_shape
        self.num_classes = num_classes

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
        batches = [list(range(s, e)) for s, e in batches]
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

        dst_img = cv2.resize(dst_img, self.input_shape)
        return dst_img

    def read_raw(self, path, is_mask=False):
        img = io.imread(path, plugin='simpleitk').squeeze()
        img = self.resize(img)

        if is_mask:
            mask = np.zeros((*self.input_shape, self.num_classes), dtype=np.uint8)
            for i in range(self.num_classes):
                mask[img == (i + 1), i] = 255
            img = mask
        else:
            img = np.concatenate(3 * [np.expand_dims(img, 2)], axis=2)
        return img

    def imread_batch(self, batch):
        batch_list = batch.values.tolist()
        images = []
        masks = []
        for img_path, mask_path in batch_list:
            img = self.read_raw(img_path)
            mask = self.read_raw(mask_path, is_mask=True)
            images.append(img / 255)
            masks.append(mask / 255)
        images = np.array(images)
        masks = np.array(masks)

        images = from_numpy(images).permute(0, 3, 1, 2).float()
        masks = from_numpy(masks).permute(0, 3, 1, 2).float()
        #imshow_img_tensor_pair(images[0].permute(1, 2, 0), masks[0].permute(1, 2, 0))
        return images, masks

    def train_generator(self):
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

    data_path = Path(__file__).joinpath('..', '..', 'data').resolve()
    img_gen = ImageGenerator(data_path.joinpath('train_annotations.csv').resolve(), 8)
    gen = img_gen.train_generator()
    for imgs, masks in gen:
        for i, m in zip(imgs, masks):
            i = i.detach().permute(1, 2, 0).numpy()
            m = m.detach().permute(1, 2, 0).numpy()
            imshow_img_pair(i, m)