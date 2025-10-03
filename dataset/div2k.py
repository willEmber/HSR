import random

import math
import os
import pdb

import mmcv
import numpy as np
from PIL import Image
from mmengine import list_from_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from scipy.ndimage import zoom

import dataset.transform as Xform
import torch


class DIV2K(Dataset):

    def __init__(self, data_list=None, training=False, cfg=None):
        super(DIV2K, self).__init__()
        self.cfg = cfg
        # Load list entries first without prefix, then resolve to absolute paths.
        # This avoids issues when the list file contains absolute paths or blank lines.
        raw_list = list_from_file(data_list)
        resolved: list[str] = []
        for entry in raw_list:
            if entry is None:
                continue
            p = str(entry).strip()
            if not p:
                # Skip blank lines
                continue
            # If relative path, join with data_root; otherwise keep absolute
            if not os.path.isabs(p):
                p = os.path.join(cfg.data_root, p)
            # Filter out directories or non-existent paths early to prevent IsADirectoryError
            if os.path.isdir(p):
                # Silently skip directories; list files should only contain image files
                continue
            if not os.path.isfile(p):
                # Skip invalid entries
                continue
            resolved.append(p)

        if len(resolved) == 0:
            raise FileNotFoundError(
                f"No valid image files resolved from list '{data_list}'. "
                f"Please check DATA.data_root='{cfg.data_root}' and list file contents.")

        self.imgs = resolved
        assert self.cfg.patch_size % self.cfg.base_resolution == 0, "Patch size must base resolution"
        self.training = training

        self.transforms_ = transforms.Compose([transforms.Resize(cfg.patch_size, InterpolationMode.BICUBIC),
                       # transforms.RandomCrop(256),
                       # transforms.RandomHorizontalFlip(),
                       transforms.ToTensor()]
                       )

    def __len__(self):
        if not self.cfg.debug:
            return len(self.imgs) * self.cfg.loop
        else:
            return 512

    def __getitem__(self, index_long):
        """
        :param index_long:
        :return: RGB, np.float32
        """
        index = index_long % len(self.imgs)
        index_sec = random.randint(0, len(self.imgs) - 1)
        img_gt = mmcv.imread(self.imgs[index]).astype(np.float32) / 255.
        img_sec = mmcv.imread(self.imgs[index_sec]).astype(np.float32) / 255.
        img_sec_2 = mmcv.imread(self.imgs[random.randint(0, len(self.imgs) - 1)]).astype(np.float32) / 255.
        img_sec_3 = mmcv.imread(self.imgs[random.randint(0, len(self.imgs) - 1)]).astype(np.float32) / 255.

        if self.cfg.benchmark:
            # img_gt_pil = Image.fromarray((img_gt * 255).astype(np.uint8))
            # img_gt = self.transforms_(img_gt_pil)

            img_gt = mmcv.imresize(img_gt, (self.cfg.patch_size, self.cfg.patch_size))
            img_sec = mmcv.imresize(img_sec, (self.cfg.patch_size, self.cfg.patch_size))
            img_sec_2 = mmcv.imresize(img_sec_2, (self.cfg.patch_size, self.cfg.patch_size))
            img_sec_3 = mmcv.imresize(img_sec_3, (self.cfg.patch_size, self.cfg.patch_size))


        if self.training:
            img_gt = Xform.random_crop(img_gt, self.cfg.patch_size)
            img_gt = Xform.augment(img_gt, hflip=self.cfg.hflip, rotation=self.cfg.rotation)
            img_sec = Xform.random_crop(img_sec, self.cfg.patch_size)
            img_sec = Xform.augment(img_sec, hflip=self.cfg.hflip, rotation=self.cfg.rotation)
            img_sec_2 = Xform.random_crop(img_sec_2, self.cfg.patch_size)
            img_sec_2 = Xform.augment(img_sec_2, hflip=self.cfg.hflip, rotation=self.cfg.rotation)
            ############################# if sec3
            img_sec_3 = Xform.random_crop(img_sec_3, self.cfg.patch_size)
            img_sec_3 = Xform.augment(img_sec_3, hflip=self.cfg.hflip, rotation=self.cfg.rotation)


        img_gt = mmcv.bgr2rgb(img_gt)
        img_gt = torch.from_numpy(img_gt.transpose((2, 0, 1)))
        img_sec = mmcv.bgr2rgb(img_sec)
        img_sec = torch.from_numpy(img_sec.transpose((2, 0, 1)))
        img_sec_2 = mmcv.bgr2rgb(img_sec_2)
        img_sec_2 = torch.from_numpy(img_sec_2.transpose((2, 0, 1)))
        img_sec_3 = mmcv.bgr2rgb(img_sec_3)
        img_sec_3 = torch.from_numpy(img_sec_3.transpose((2, 0, 1)))

        return {'img_gt': img_gt, 'img_sec': img_sec, 'img_sec_2': img_sec_2, 'img_sec_3': img_sec_3}


        # return img_gt
        # return {'img_gt': img_gt, 'img_sec': img_sec, 'img_sec_2': img_sec_2}


