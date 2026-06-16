import numpy as np
import torch

from sklearn.metrics import f1_score


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, enable_checkpoint=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.enable_checkpoint = enable_checkpoint

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.enable_checkpoint:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def cal_accuracy(y_pred, y_true):
    # 原始的是计算Accuracy，这里计算f1 weight score
    # return np.mean(y_pred == y_true)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return f1
