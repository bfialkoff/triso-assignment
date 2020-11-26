import time

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from segmentation_models_pytorch.utils.losses import DiceLoss
from utils.meter import Meter

def imshow_img_pair(img, mask):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()

epoch_log = lambda epoch_loss, dice, iou: print('Loss: %0.4f | IoU: %0.4f | dice: %0.4f |' % (epoch_loss, iou, dice))

class Criterion:
    def __init__(self, *losses):
        self.losses = losses

    def __call__(self, p, t):
        l = sum(loss(p, t) for loss in self.losses)
        return l

class Trainer:
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, train_generator_object, val_generator_object, weights_path, num_epochs=20, initial_weights=None):
        self.num_workers = 6
        self.initial_weights = initial_weights
        self.train_generator_object = train_generator_object
        self.val_generator_object = val_generator_object
        self.weights_path = weights_path
        self.batch_size = {'train': train_generator_object.batch_size,
                           'val': val_generator_object.batch_size}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = num_epochs
        self.initial_epoch = 0
        self.best_loss = float('inf')
        self.best_iou = 0
        self.phases = ['train', 'val']
        self.device = torch.device('cuda:0' if torch.cuda.device_count() else 'cpu')
        self.net = model
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(activation='softmax2d')
        self.criterion = Criterion(self.bce_loss, self.dice_loss)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)
        if self.initial_weights:
            self.restore_state()
        self.net = self.net.to(self.device)
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def restore_state(self):
        state = torch.load(self.initial_weights)
        self.net.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.best_loss = state['best_loss']
        self.initial_epoch = state['epoch']

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase, dataloader, num_batches):
        meter = Meter(phase, epoch)
        start = time.strftime('%H:%M:%S')
        print(f'Starting epoch: {epoch} | phase: {phase} | ⏰: {start}')
        self.net.train(phase == 'train')
        running_loss = 0.0
        self.optimizer.zero_grad()
        for i, batch in tqdm(enumerate(dataloader), total=num_batches):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == 'train':
                loss.backward()
                if (i + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        epoch_loss = (running_loss * self.accumulation_steps) / num_batches

        dice, iou = meter.get_metrics()
        epoch_log(epoch_loss, dice, iou)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, dice, iou

    def start(self):
        for epoch in range(self.initial_epoch, self.num_epochs):
            print(f'Epoch {epoch}')
            self.iterate(epoch, 'train', self.train_generator_object.train_generator(), len(self.train_generator_object))

            with torch.no_grad():
                print('beginning eval')
                val_loss, val_dice, val_iou = self.iterate(epoch, 'val', self.val_generator_object.val_generator(), len(self.val_generator_object))
                self.scheduler.step(val_loss)
            if val_iou < self.best_iou:
                state = {
                    'epoch': epoch,
                    'loss': val_loss,
                    'best_iou': val_iou,
                    'dice': val_dice,
                    'state_dict': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                print('******** New optimal found, saving state ********')
                torch.save(state, self.weights_path)

