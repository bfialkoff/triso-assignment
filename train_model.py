from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from segmentation_models_pytorch import Unet

import torch
tensor_type = 'torch.cuda.FloatTensor' if torch.cuda.device_count() else 'torch.FloatTensor'
torch.set_default_tensor_type(tensor_type) # 'torch.cuda.FloatTensor'

from utils.image_generator import ImageGenerator
from utils.trainer import Trainer



def imshow(img):
    f, ax = plt.subplots(1, 1)
    ax.imshow(img)
    plt.show()

mkdir = lambda p: p.mkdir(parents=True) if not p.exists() else None


if __name__ == '__main__':
    initial_date = ''
    num_epochs = 10000
    date_id = initial_date if initial_date else datetime.now().strftime('%Y%m%d%H%M')


    backbone = 'resnet18'
    model = Unet(backbone,
                 encoder_weights='imagenet',
                 classes=3)

    weights_path = Path(__file__).joinpath('..', 'triso_weights', f'{date_id}_{backbone}_model.pth').resolve()
    init_weights = Path(__file__).joinpath('..', 'triso_weights', f'202011262249_{backbone}_model.pth').resolve()
    # init_weights = None

    mkdir(weights_path.parent)
    input_size = 256
    batch_size = 16
    data_path = Path(__file__).joinpath('..', 'data').resolve()
    train_gen_obj = ImageGenerator(data_path.joinpath('train_annotations.csv').resolve(), batch_size, input_shape=(input_size, input_size))
    val_gen_obj = ImageGenerator(data_path.joinpath('val_annotations.csv').resolve(), batch_size, input_shape=(input_size, input_size))
    trainer = Trainer(model, train_gen_obj, val_gen_obj, weights_path, num_epochs=num_epochs, initial_weights=init_weights)
    trainer.start()
