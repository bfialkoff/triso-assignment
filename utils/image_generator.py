from random import sample, shuffle

from torch import from_numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.img_ops import ImgOps

def imshow_img_pair(img, mask):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()


class ImageGenerator(ImgOps):

    def __init__(self, annotations_path, batch_size, input_shape=(512, 512), num_classes=3):
        self.annotations = pd.read_csv(annotations_path)
        self.batch_size = batch_size
        self.val_batches, self.num_val_batches = self.create_val_batches()
        _, self.num_train_batches = self.create_train_batches()
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_train_batches(self):
        num_samples = len(self.annotations)
        num_batches = num_samples // self.batch_size
        remainder = num_samples % self.batch_size
        if remainder:
            num_batches += 1
        difference = self.batch_size - remainder
        indices = list(range(num_samples))

        duplicates = sample(indices, difference)
        indices = indices + duplicates
        shuffle(indices)
        batched = [indices[i:i+self.batch_size] for i in range(num_batches)]
        return batched, num_batches

    def create_val_batches(self):
        num_samples = len(self.annotations)
        num_batches = num_samples // self.batch_size
        remainder = num_samples % self.batch_size
        batches = [(i * self.batch_size, (i + 1) * self.batch_size) for i in range(num_batches)]
        if remainder:
            s = num_batches * self.batch_size
            e = s + remainder
            batches.append((s, e))
        batches = [list(range(s, e)) for s, e in batches]
        return batches, num_batches

    def imread_batch(self, batch):
        batch_list = batch.values.tolist()
        images = []
        masks = []
        for img_path, mask_path in batch_list:
            img = self.read_image(img_path, self.input_shape)
            mask = self.read_mask(mask_path, self.input_shape, self.num_classes)
            images.append(img / 255)
            masks.append(mask / 255)
        images = np.array(images)
        masks = np.array(masks)

        images = from_numpy(images).permute(0, 3, 1, 2).float()
        masks = from_numpy(masks).permute(0, 3, 1, 2).float()
        return images, masks

    def train_generator(self):
        batches, num_batches = self.create_train_batches()
        for batch_indices in batches:
            batch = self.annotations.loc[batch_indices]
            images, masks = self.imread_batch(batch)
            yield images, masks

    def val_generator(self):
        for batch_indices in self.val_batches:
            batch = self.annotations.loc[batch_indices]
            images, masks = self.imread_batch(batch)
            yield images, masks

    def inference_generator(self):
        # fixme this should also do paths
        for batch_indices in self.val_batches:
            batch_list = self.annotations.loc[batch_indices, 'image_path'].values.reshape(-1).tolist()
            images = []
            for img_path in batch_list:
                img = self.read_image(img_path, self.input_shape)
                images.append(img / 255)
            images = np.array(images)
            images = from_numpy(images).permute(0, 3, 1, 2).float()
            yield batch_list, images

    def test_generator(self):
        # fixme this should also do paths
        for batch_indices in self.val_batches:
            batch = self.annotations.loc[batch_indices]
            images, masks = self.imread_batch(batch)
            yield batch.values.tolist(), images

    def __len__(self):
        return self.num_train_batches

if __name__ == '__main__':
    from pathlib import Path

    data_path = Path(__file__).joinpath('..', '..', 'data').resolve()
    img_gen = ImageGenerator(data_path.joinpath('train_annotations.csv').resolve(), 8)
    gen = img_gen.train_generator()
    for imgs, masks in gen:
        continue
        for i, m in zip(imgs, masks):
            i = i.detach().permute(1, 2, 0).numpy()
            m = m.detach().permute(1, 2, 0).numpy()
            imshow_img_pair(i, m)