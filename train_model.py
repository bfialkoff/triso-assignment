from pathlib import Path
from datetime import datetime

from segmentation_models_pytorch import Unet
import torch
import torch.nn.functional as F
from torch import from_numpy
import numpy as np
import matplotlib.pyplot as plt

from utils.image_generator import ImageGenerator
from utils.trainer import Trainer



def imshow(img):
    f, ax = plt.subplots(1, 1)
    ax.imshow(img)
    plt.show()

mkdir = lambda p: p.mkdir(parents=True) if not p.exists() else None


if __name__ == '__main__':
    initial_date = ''
    initial_epoch = -1
    num_epochs = 10000

    date_id = initial_date if initial_date else datetime.now().strftime('%Y%m%d%H%M')

    experiment_dir = Path(__file__).joinpath('..', 'triso_weights', date_id).resolve()
    mkdir(experiment_dir)

    model = Unet("resnet18", encoder_weights="imagenet", classes=3, activation=None)

    data_path = Path(__file__).joinpath('..', 'data').resolve()
    train_gen_obj = ImageGenerator(data_path.joinpath('train_annotations.csv').resolve(), 8, input_size=(128, 128))
    val_gen_obj = ImageGenerator(data_path.joinpath('val_annotations.csv').resolve(), 8, input_size=(128, 128))
    trainer = Trainer(model, train_gen_obj, val_gen_obj, experiment_dir)
    trainer.start()
