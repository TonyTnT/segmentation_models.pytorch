import pandas as pd
import numpy as np
import glob
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import cv2, os, time, platform
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensor
from albumentations import (
    OneOf, Resize, Normalize, Compose, Transpose,
    HorizontalFlip, VerticalFlip, Flip, Cutout, RandomCrop, RandomRotate90, ShiftScaleRotate,
    RandomContrast, RandomBrightness, RandomBrightnessContrast,
)
import argparse
from utils import Seg_Trainer
import warnings

warnings.filterwarnings('ignore')

# image transform
def seg_transforms(phase, resize=(512, 512), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """ Get segmentation albumentation tansforms
    Args:
        phase: train or valid
        resize: input image shape into model

    Returns:
        albu compose transforms
    Raises:
        IOError: An error occurred accessing ablumentation object.
    """
    assert (phase in ['train', 'valid', 'test'])
    transforms_list = []
    if phase == 'train':
        transforms_list.extend([
            # Rotate
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(rotate_limit=20, border_mode=0, p=0.2),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            Resize(resize[0] + 64, resize[1] + 64, interpolation=Image.BILINEAR),
            Normalize(mean=mean, std=std, p=1),
            RandomCrop(resize[0], resize[1]),
            ToTensor(),
        ])
    else:
        transforms_list.extend([
            Resize(resize[0], resize[1], interpolation=Image.BILINEAR),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ])
    transforms = Compose(transforms_list)
    return transforms


# custom dataset

class Defect_Dataset(Dataset):
    """Get torch dataset
    Args:
        dataframe: dataframe contains image, label & mask
        resize: input image shape into model

    Returns:
        torch dataset
    Raises:
        IOError: An error occurred accessing torch dataset.
    """

    def __init__(self, jpg_path, seg_path, sam_path=None, transform=None):

        self.use_sam = True if sam_path else False
        self.jpg_path = jpg_path
        self.seg_path = seg_path
        self.sam_path = sam_path
        self.filenames = os.listdir(self.jpg_path)
        self.seg_transforms = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        imgPath = self.filenames[idx]

        image = cv2.imread(os.path.join(self.jpg_path, imgPath))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = rle_to_mask(rle, image.shape[0], image.shape[1])
        mask = cv2.imread(os.path.join(self.seg_path, imgPath.replace('.jpg', '.png')), 0)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        if self.use_sam:
            sam = cv2.imread(os.path.join(self.sam_path, imgPath.replace('.jpg', '.png')), 0)
            mask_sam = np.dstack((mask, sam))
            if self.seg_transforms is not None:
                aug = self.seg_transforms(image=image, mask=mask_sam)
                # mask_sam = aug["mask"].squeeze()
                # print(aug["image"].shape, aug["mask"][:,:, :, 0].shape, aug["mask"][:,:, :, 1].shape)
                return aug["image"], aug["mask"][:, :, :, 0], aug["mask"][:, :, :, 1]
            else:
                return image, mask, sam
        else:
            if self.seg_transforms is not None:
                aug = self.seg_transforms(image=image, mask=mask)
                # print(aug["image"].shape, aug["mask"].shape)
                return aug["image"], aug["mask"]
            else:
                return image, mask


model = smp.Unet('resnet18', encoder_weights='imagenet')
model = smp.Unet('resnet18', encoder_weights='imagenet')

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ###################################################################################################################
    # 部分训练超参数设定
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="batch size of input")
    parser.add_argument("--epochs", type=int, default=2, help="total_epoch")
    parser.add_argument("--img_size", type=int, default=512, help="size of image height")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of the optimizer")
    opt = parser.parse_args()
    ###################################################################################################################

    ###################################################################################################################
    # 输入输出路径设定
    jpg_path = '/media/mdisk/chenx2/US_dalian_train/JPEGImages'
    seg_path = '/media/mdisk/chenx2/US_dalian_train/Segmentation'
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "model"))
    out_dir = os.path.join(out_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time())))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    log_file = os.path.join(out_dir, 'train_log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as acc_file:
            pass
    with open(log_file, 'a') as acc_file:
        acc_file.write('The details of this training is: {}.\n'.format(opt))
    ###################################################################################################################

    ###################################################################################################################
    # 部分训练参数设定
    if platform.system() == 'Linux':
        work_num = 10
    else:
        work_num = 0

    ratio_trainval = 0.10
    ###################################################################################################################

    ###################################################################################################################
    # 训练集、验证集的划分
    datasets = Defect_Dataset(jpg_path, seg_path, seg_path,
                              seg_transforms(phase='train', resize=(opt.img_size, opt.img_size)))

    trainsize = int(len(datasets) * ratio_trainval)
    valsize = len(datasets) - trainsize
    train_set, val_set = torch.utils.data.random_split(datasets, [trainsize, valsize])

    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch_size,
        num_workers=work_num,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=opt.batch_size,
        num_workers=work_num,
        pin_memory=True,
        shuffle=True,
    )
    dataloaders = {"train": train_loader, "val": val_loader}
    ###################################################################################################################

    ###################################################################################################################
    # 初始化 模型、loss函数
    # model = smp.Unet('se_resnext101_32x4d', encoder_weights='imagenet', classes=1, activation=None)
    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=1, activation=None)

    criterion = nn.BCEWithLogitsLoss()
    ###################################################################################################################
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
        criterion = criterion
    elif torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion
    else:
        criterion = criterion
    ###################################################################################################################
    # 训练
    model_trainer = Seg_Trainer(dataloaders, model, criterion, out_dir, opt.lr, opt.batch_size, opt.epochs,
                                use_sam=True)
    model_trainer.start()
    ###################################################################################################################
    torch.cuda.empty_cache()
