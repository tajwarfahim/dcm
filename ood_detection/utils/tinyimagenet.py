# Citation:
# 1. https://github.com/luoyan407/congruency/blob/master/train_timgnet.py

# general imports
import torch
import os
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0, len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames, classes


def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            # if datasets.folder.is_image_file(fname):
            if datasets.folder.has_file_allowed_extension(
                fname, [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]
            ):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class TImgNetVal(torch.utils.data.Dataset):
    """Dataset wrapping images and ground truths."""

    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, y) where y is the label of the image.
        """
        img = None
        with open(os.path.join(self.img_path, self.imgs[index]), "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        y = self.classidx[index]
        return img, y

    def __len__(self):
        return len(self.imgs)


class TinyImageNet(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        transform=None,
        target_transform=None,
        return_label=True,
        verbose=False,
    ):
        assert split in ["train", "val", "test"]
        self.split = split

        self.num_classes = 200
        self.transform = transform
        self.target_transform = target_transform

        self.return_label = return_label
        self.verbose = verbose
        self.root_dir = root_dir

        self.load_dataset()

    def get_class_to_idx(self):
        return self.class_to_idx.copy()

    def load_dataset(self):
        train_dir = os.path.join(self.root_dir, "train")
        self.dataset = datasets.ImageFolder(
            root=train_dir,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        # fix class to idx copy
        self.class_to_idx = self.dataset.class_to_idx

        if self.split in ["val", "test"]:
            val_dir = os.path.join(self.root_dir, "val", "images")
            gt_path = os.path.join(self.root_dir, "val", "val_annotations.txt")
            self.dataset = TImgNetVal(
                img_path=val_dir,
                gt_path=gt_path,
                class_to_idx=self.class_to_idx.copy(),
                transform=self.transform,
            )

        elif self.split != "train":
            raise ValueError(f"Given split type: {self.split} is not supported.")

        self.indices = [i for i in range(len(self.dataset))]
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        new_index = self.indices[index]
        img, label = self.dataset[new_index]

        if not self.return_label:
            return img

        return img, label
