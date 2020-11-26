import time

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .meter import Meter

def imshow_img_pair(img, mask):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.show()


def epoch_log(epoch_loss, dice, iou):
    '''logging the metrics at the end of an epoch'''
    print('Loss: %0.4f | IoU: %0.4f | dice: %0.4f |' % (epoch_loss, iou, dice))

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
        self.phases = ['train', 'val']
        self.device = torch.device('gpu' if torch.cuda.device_count() else 'cpu') # fixme needs to actually be cuda:0 or somethin
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
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
        return epoch_loss

    def start(self):
        for epoch in range(self.initial_epoch, self.num_epochs):
            print(f'Epoch {epoch}')
            self.iterate(epoch, 'train', self.train_generator_object.train_generator(), len(self.train_generator_object))
            state = { # fixme move this to only happen if we save the state else no reason to do it
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            with torch.no_grad():
                print('beginning eval')
                val_loss = self.iterate(epoch, 'val', self.val_generator_object.val_generator(), len(self.val_generator_object))
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('******** New optimal found, saving state ********')
                state['best_loss'] = self.best_loss = val_loss
                torch.save(state, self.weights_path)

