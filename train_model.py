from pathlib import Path
from datetime import datetime

from segmentation_models_pytorch import Unet
import torch
import torch.nn.functional as F
from torch import from_numpy
import numpy as np
from triso.utils.image_generator import ImageGenerator
from triso.utils.trainer import Trainer

import matplotlib.pyplot as plt

def imshow(img):
    f, ax = plt.subplots(1, 1)
    ax.imshow(img)
    plt.show()

class Unet1C(Unet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_conv = torch.nn.Conv2d(1, 3, (7, 7))

    def forward(self, x):
        x = self.extra_conv(x)
        x = F.pad(x, (0, 6, 0, 6))
        x = super(Unet1C, self).forward(x)
        return x

def make_weights_dir(experiment_dir):
    weights_path = experiment_dir.joinpath('weights')
    if not weights_path.exists():
        weights_path.mkdir(parents=True)
    return weights_path

if __name__ == '__main__':
    initial_date = ''
    initial_epoch = -1
    num_epochs = 10000

    date_id = initial_date if initial_date else datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__).joinpath('..', 'triso_weights', date_id).resolve()
    summary_path = experiment_dir.joinpath('summaries', 'summary.json')
    weights_dir = make_weights_dir(experiment_dir)
    initial_weights = weights_dir.joinpath(f'epoch_{initial_epoch}.pth') if (initial_epoch > 0) else None

    model = Unet1C("resnet18", encoder_weights="imagenet", classes=3, activation=None)
    model(from_numpy(np.random.random((8, 1, 512, 512))).float())

    data_path = Path(__file__).joinpath('..', '..', 'data').resolve()
    train_gen_obj = ImageGenerator(data_path.joinpath('train_annotations.csv').resolve(), 8)
    val_gen_obj = ImageGenerator(data_path.joinpath('val_annotations.csv').resolve(), 8)
    trainer = Trainer(model, train_gen_obj, val_gen_obj, experiment_dir)
    trainer.start()
