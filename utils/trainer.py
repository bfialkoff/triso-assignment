import time

from tqdm import tqdm
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .meter import Meter

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print('Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f' % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

class Trainer:
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, train_generator_object, val_generator_object, experiment_dir):
        self.num_workers = 6
        self.train_generator_object = train_generator_object
        self.val_generator_object = val_generator_object
        self.experiment_dir = experiment_dir
        self.batch_size = {'train': 4, 'val': 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor') # 'torch.cuda.FloatTensor'
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, verbose=True)
        self.net = self.net.to(self.device)
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

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
        batch_size = self.batch_size[phase]
        self.net.train(phase == 'train')
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, batch in tqdm(enumerate(dataloader), total=num_batches):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == 'train':
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / num_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}')
            self.iterate(epoch, 'train', self.train_generator_object.train_generator(), len(self.train_generator_object))
            state = {
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
                torch.save(state, self.experiment_dir.parent.joinpath(self.experiment_dir.name + '_model.pth').resolve())
            print()
