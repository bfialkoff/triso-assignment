import torch
import numpy as np

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou

def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.reshape(batch_size, -1)
        truth = truth.reshape(batch_size, -1)
        #truth = torch.from_numpy(truth).view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p*t).sum(-1)/((p+t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

    return dice

def predict(X, threshold):
    '''X is sigmoid output of the model'''
    X_p = np.copy(X)
    preds = (X_p > threshold).astype('uint8')
    return preds

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.iou_scores = []
        self.pp_base_dice_scores = []
        self.pp_iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1]) # fixme iou is computed on 1 class ?!
        self.iou_scores.append(iou)

    def update_with_post_process(self, targets, outputs, post_process_function):
        probs = torch.sigmoid(outputs)
        probs = probs.detach().permute(0, 2, 3, 1).numpy()
        probs = (255 * probs).astype(np.uint8)
        probs = post_process_function(probs) / 255
        probs = torch.from_numpy(probs).permute(0, 3, 1, 2).float()

        dice = metric(probs, targets, self.base_threshold)
        self.pp_base_dice_scores.extend(dice.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1]) # fixme iou is computed on 1 class ?!
        self.pp_iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        pp_iou = np.nanmean(self.pp_base_dice_scores)
        pp_dice = np.nanmean(self.pp_iou_scores)
        return dice, iou, pp_dice, pp_iou
