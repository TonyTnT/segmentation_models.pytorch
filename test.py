import pandas as pd
import numpy as np
import glob
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import cv2, os
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


# Thanks to the authors of: https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
def mask_to_rle(mask):
    '''
    Convert a mask into RLE
    
    Parameters: 
    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background

    Returns: 
    sring: run length encoding 
    '''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


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

    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.seg_transforms = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgPath = self.df[idx]

        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.seg_transforms is not None:
            aug = self.seg_transforms(image=image)
            return imgPath, aug["image"]
        else:
            return imgPath, image


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    outputpath = '/media/mdisk/chenx2/stomach/result'
    img_list = os.listdir('/media/mdisk/chenx2/stomach/test')  # 需要自己根据需求更改路径
    model = smp.Unet('resnext101_32x8d', encoder_weights='imagenet', classes=1, activation=None)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    elif torch.cuda.is_available():
        model = model.cuda()

    pth_dir = './model/20200907_111645/model_lowest_loss.pth'  # 需要自己根据需求更改路径
    params = torch.load(pth_dir)
    model.load_state_dict(params['state_dict'])
    model.eval()

    test_transform = seg_transforms(phase='test', resize=(512, 512))
    for i in tqdm(range(len(img_list))):

        imgPath = '/media/mdisk/chenx2/stomach/test/%s' % img_list[i]  # 需要自己根据需求更改路径

        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size=(image.shape[1], image.shape[0])
        test_retransform = seg_transforms(phase='test', resize=ori_size)

        image = test_transform(image=image)['image']
        image = torch.unsqueeze(image, 0)
        mask = torch.sigmoid(torch.squeeze(model(image))).detach().cpu().numpy()
        mask[mask > 0.4] = 255  # 阈值可自行更改
        mask[mask <= 0.4] = 0
        if mask_to_rle(mask) == '':  # 防止输出空，导致结果无法评分
            mask[250:255, 250:255] = 255
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.imshow(mask, cmap="gray")  # 以灰度图显示图片
        # ax.set_title("hei,i'am the title")  # 给图片加titile
        # plt.show()  # 显示刚才所画的所有操作

        image = torch.squeeze(image).permute(2, 1, 0).detach().cpu().numpy()
        image = cv2.resize(image*255,ori_size)
        mask = cv2.resize(mask,ori_size)
        # print('mask', mask.shape,'image',image.shape)

        cv2.imwrite(os.path.join(outputpath, img_list[i].replace('.jpg', '_mask.jpg')), mask)
        # cv2.imwrite(os.path.join(outputpath, img_list[i]), image)

        # break
    print('finish')
