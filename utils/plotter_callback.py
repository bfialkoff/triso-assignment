"""
this callback
"""
import os
import json
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np


class PlotterBase:

    def __init__(self, train_generator, val_generator, summary_path, loss, sample_size=1, model=None):
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.summary_path = summary_path
        self.loss = loss
        self.sample_size = sample_size

    @classmethod
    def load_summary(cls, summary_path):
        with open(summary_path, 'r') as f:
            s = json.load(f)
        return s

    def save_summary(self, summary):
        with open(self.summary_path, 'w') as f:
            json.dump(summary, f, indent=4, sort_keys=True)

    def write_summary(self, key, update):
        summary = self.load_summary(self.summary_path)
        summary.update({f'{key:02d}': update})
        self.save_summary(summary)

    def get_metrics(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        return loss

    def on_epoch_end(self, epoch, model):
        train_labels, train_predictions = self.run(model, self.train_generator)
        val_labels, val_predictions = self.run(model, self.val_generator)
        train_loss = self.get_metrics(train_labels, train_predictions)
        val_loss = self.get_metrics(val_labels, val_predictions)

        # fixme metrics
        update = {
                  'train_loss': train_loss.astype(float),
                  'val_loss': val_loss.astype(float),
                  }
        print('train_loss', update['train_loss'], 'val_loss', update['val_loss'])
        self.write_summary(epoch, update)

    def on_train_begin(self, model, logs=None):
        """
        if summary_path_dir doesnt exist create dir call write_sumamry
        """
        self.model_summary = self.summary_path.parents[0].joinpath('model.txt').resolve()
        if model is not None and False: # fixme this doesnt work for torch models
            with open(self.model_summary, 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))


    # fixme this needs to calculate loss and metrics during run
    def run(self, model, gen_obj):
        y_true = np.zeros(gen_obj.steps * gen_obj.batch_size * self.sample_size) # here
        y_pred = np.zeros(gen_obj.steps * gen_obj.batch_size * self.sample_size) # here
        for i, (data, labels) in enumerate(gen_obj.val_generator()):
            start_index = i * gen_obj.batch_size
            end_index = start_index + len(data)
            pred = model.predict_on_batch(data)
            y_true[start_index:end_index] = labels
            y_pred[start_index:end_index] = pred.reshape(-1) # here
        return y_true, y_pred


if __name__ == '__main__':
    PlotterBase.get_gridspec()
    pass