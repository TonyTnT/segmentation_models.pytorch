import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import numpy as np
import pandas as pd
from collections import Counter
import time, os, platform, sys, re
import torch.backends.cudnn as cudnn


def metric(probability, truth, threshold=0.5, reduction='none'):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert (probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def predict(self, X, threshold):
        '''X is sigmoid output of the model'''
        X_p = np.copy(X)
        preds = (X_p > threshold).astype('uint8')
        return preds

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = self.predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (
        epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou


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


def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


class Seg_Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, dataloaders, model, criterion, out_dir, lr=5e-4, batch_size=8, epochs=20, use_sam=False):
        self.batch_size = {"train": batch_size, "val": 4 * batch_size}
        if self.batch_size["train"] < 24:
            self.accumulation_steps = 24 // self.batch_size["train"]
        else:
            self.accumulation_steps = 1
        # print(self.accumulation_steps)
        self.out_dir = out_dir
        self.lr = lr
        self.num_epochs = epochs
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            # torch.set_default_tensor_type("torch.cuda.FloatTensor")
            cudnn.benchmark = True
        # else:
        #     torch.set_default_tensor_type("torch.FloatTensor")
        self.net = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4, amsgrad=True)
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.2)

        self.dataloaders = dataloaders
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.use_sam = use_sam

    def forward(self, images, targets, usmasks=None):
        if self.cuda:
            images = images.cuda()
            masks = targets.cuda()
            if self.use_sam:
                usmasks = usmasks.cuda()
        if self.use_sam:
            outputs = self.net(images, usmasks)
        else:
            outputs = self.net(images)
        # masks=masks.unsqueeze(1)
        loss = self.criterion(outputs, masks)
        # print(loss)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        # tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        if self.use_sam:
            for itr, (images, targets, usmasks) in enumerate(dataloader):
                loss, outputs = self.forward(images, targets, usmasks)
                loss = loss / self.accumulation_steps
                if phase == "train":
                    loss.backward()
                    if (itr + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                running_loss += loss.item()
                outputs = outputs.detach().cpu()
                meter.update(targets, outputs)
        else:
            for itr, (images, targets) in enumerate(dataloader):

                loss, outputs = self.forward(images, targets)
                loss = loss / self.accumulation_steps
                if phase == "train":
                    loss.backward()
                    if (itr + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                running_loss += loss.item()
                outputs = outputs.detach().cpu()
                meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):

        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                # self.scheduler.step(val_loss)
                restart = self.scheduler.step()
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, self.out_dir + '/model_lowest_loss.pth')
                with open(self.out_dir + '/train_log.txt', 'a') as acc_file:
                    acc_file.write('New optimal found (loss %s), state saved.\n' % val_loss)

            with open(self.out_dir + '/train_log.txt', 'a') as acc_file:
                acc_file.write('Epoch: %2d, Loss: %.8f, Dice: %.8f, IoU: %.8f.\n ' % (epoch,
                                                                                      self.losses['val'][-1],
                                                                                      self.dice_scores['val'][-1],
                                                                                      self.iou_scores['val'][-1]))

            if restart:
                with open(self.out_dir + '/train_log.txt', 'a') as acc_file:
                    acc_file.write('At Epoch: %2d, Optim Restart. Got Loss: %.8f, Dice: %.8f, IoU: %.8f.\n ' % (epoch,
                                                                                                                self.losses[
                                                                                                                    'val'][
                                                                                                                    -1],
                                                                                                                self.dice_scores[
                                                                                                                    'val'][
                                                                                                                    -1],
                                                                                                                self.iou_scores[
                                                                                                                    'val'][
                                                                                                                    -1]))
            print()
