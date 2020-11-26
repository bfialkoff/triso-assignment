from pathlib import Path


from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2
import numpy as np

from utils.image_generator import ImageGenerator

def imshow_img_pair(img, mask, t=None):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    if t is not None:
        plt.suptitle(t)
    plt.show()

mkdir = lambda p: p.mkdir(parents=True) if not p.exists() else None
sigmoid = lambda x: 1/(1 + np.exp(-x))

class Predictor:
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, generator_object, weights, phase='val'):
        self.weights = weights
        self.generator_object = generator_object
        self.batch_size = generator_object.batch_size
        self.device = torch.device('gpu' if torch.cuda.device_count() else 'cpu') # fixme needs to actually be cuda:0 or somethin
        self.net = model
        self.restore_state()
        self.net = self.net.to(self.device)
        self.net.train(False)
        self.dst_path = Path(__file__).joinpath('..', '..', 'data', 'prediction', phase).resolve()


    def restore_state(self):
        state = torch.load(self.weights)
        self.net.load_state_dict(state['state_dict'])

    def forward(self, images):
        images = images.to(self.device)
        outputs = self.net(images)
        return outputs

    def write(self, input_tensor, output_tensor, src_path):
        src_path  = Path(src_path)
        patient = src_path.parent.name
        name = src_path.name
        dst_path = self.dst_path.joinpath(patient).resolve()

        gt_src_path = src_path.parent.joinpath(name.replace('.mhd', '_gt.mhd'))

        gt_dst_path = dst_path.joinpath(gt_src_path.name.replace('.mhd', '.jpg'))
        input_path = dst_path.joinpath(name.replace('.mhd', '.jpg'))
        output_path = dst_path.joinpath(name.replace('.mhd', '_pred.jpg'))

        input_image = np.moveaxis(input_tensor.detach().numpy(), 0, -1)
        output_image = sigmoid(np.moveaxis(output_tensor.detach().numpy(), 0, -1))

        input_image = (255 * input_image).astype(np.uint8)
        output_image = (255 * output_image).astype(np.uint8)

        mkdir(input_path.parent)
        cv2.imwrite(str(input_path), input_image)
        cv2.imwrite(str(output_path), output_image)
        if gt_src_path.exists():
            gt_mask = self.generator_object.read_mask(str(gt_src_path))
            success = cv2.imwrite(str(gt_dst_path), gt_mask)
            if not success:
                imshow_img_pair(gt_mask, gt_mask)

    def predict_and_save(self, dataloader):

        num_batches = len(self.generator_object)

        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
            paths, images = batch
            images = images.to(self.device)
            outputs = self.net(images)
            outputs = outputs.detach().cpu()
            for p, i, o in zip(paths, images, outputs):
                self.write(i, o, p)


if __name__ == '__main__':
    backbone = 'resnext50_32x4d'
    init_weights = Path(__file__).joinpath('..', '..', 'triso_weights', f'202011252258_{backbone}_model.pth').resolve()
    # init_weights = None
    # epoch 45 has 0.8 val iou
    model = Unet('resnext50_32x4d',
                 encoder_weights='imagenet',
                 classes=3,
                 activation=None)

    input_size = 256
    data_path = Path(__file__).joinpath('..','..', 'data').resolve()

    val_gen_obj = ImageGenerator(data_path.joinpath('val_annotations.csv').resolve(), 8, input_shape=(input_size, input_size))
    test_gen_obj = ImageGenerator(data_path.joinpath('test_annotations.csv').resolve(), 8,
                                   input_shape=(input_size, input_size))
    predictor = Predictor(model, val_gen_obj, init_weights, phase='val')
    predictor.predict_and_save(predictor.generator_object.inference_generator())
    # want 85 % ish