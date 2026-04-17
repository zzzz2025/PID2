import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
import random


def random_crop(image1, image2):
    min_ratio = 0.5
    max_ratio = 1
    w, h = image1.size
    ratio = random.random()
    scale = min_ratio + ratio * (max_ratio - min_ratio)
    new_h = int(h * scale)
    new_w = int(w * scale)
    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)
    image1 = image1.crop((x, y, x + new_w, y + new_h))
    image2 = image2.crop((x, y, x + new_w, y + new_h))
    return image1, image2


class my_transform_crop():
    def __init__(self, crop_p=0.5):
        self.crop_p = crop_p

    def crop_enhance(self, image1, image2):
        if random.random() <= self.crop_p:
            image1, image2 = random_crop(image1, image2)
        return image1, image2


class my_transform_flip():
    def __init__(self, flip_p=0.5):
        self.flip_p = flip_p

    def flip_enhance(self, image1, image2):
        if random.random() <= self.flip_p:
            image1 = np.flip(image1, axis=1)
            image2 = np.flip(image2, axis=1)
        return image1, image2


class _PairedDatasetBase(Dataset):
    """Shared base for paired IR + RGB datasets. Subclasses define split (train/val)."""
    SPLIT = None  # override in subclass: 'train' or 'val'

    def __init__(self,
                 data_root,
                 split_ratio=0.8,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 random_crop=True,
                 seed=42,
                 ):
        self.data_root = data_root
        self.ir_data_root = os.path.join(data_root, 'lwir')
        self.vi_data_root = os.path.join(data_root, 'visible')
        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip_enhance = my_transform_flip(flip_p=flip_p)
        self.crop_enhance = my_transform_crop(crop_p=flip_p if random_crop else 0.0)

        # Discover all IR files
        ir_files = sorted([
            f for f in os.listdir(self.ir_data_root)
            if f.endswith(('jpg', 'png'))
        ])
        n_total = len(ir_files)
        n_train = int(n_total * split_ratio)

        rng = np.random.RandomState(seed)
        indices = np.arange(n_total)
        rng.shuffle(indices)
        train_indices = set(indices[:n_train])
        val_indices = set(indices[n_train:])

        if self.SPLIT == 'train':
            self._file_indices = [i for i in range(n_total) if i in train_indices]
        else:
            self._file_indices = [i for i in range(n_total) if i in val_indices]

        self._all_files = ir_files
        self._length = len(self._file_indices)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        real_idx = self._file_indices[idx]
        ir_name = self._all_files[real_idx]
        ir_path = os.path.join(self.ir_data_root, ir_name)
        vi_path = os.path.join(self.vi_data_root, ir_name)

        image_ir = Image.open(ir_path)
        if not image_ir.mode == "RGB":
            image_ir = image_ir.convert("RGB")
        image_vi = Image.open(vi_path)
        if not image_vi.mode == "RGB":
            image_vi = image_vi.convert("RGB")

        # Center-crop to square
        img_ir = np.array(image_ir).astype(np.uint8)
        crop = min(img_ir.shape[0], img_ir.shape[1])
        h, w = img_ir.shape[0], img_ir.shape[1]
        img_ir = img_ir[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image_ir = Image.fromarray(img_ir)

        img_vi = np.array(image_vi).astype(np.uint8)
        crop = min(img_vi.shape[0], img_vi.shape[1])
        h, w = img_vi.shape[0], img_vi.shape[1]
        img_vi = img_vi[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image_vi = Image.fromarray(img_vi)

        # Resize
        if self.size is not None:
            image_ir, image_vi = self.crop_enhance.crop_enhance(image_ir, image_vi)
            image_ir = image_ir.resize((self.size, self.size), resample=self.interpolation)
            image_vi = image_vi.resize((self.size, self.size), resample=self.interpolation)

        img_ir = np.array(image_ir).astype(np.uint8)
        img_vi = np.array(image_vi).astype(np.uint8)
        img_ir, img_vi = self.flip_enhance.flip_enhance(img_ir, img_vi)

        return {
            "image": (img_ir / 127.5 - 1.0).astype(np.float32),
            "conditional": (img_vi / 127.5 - 1.0).astype(np.float32),
            "rgb_image": (img_vi / 127.5 - 1.0).astype(np.float32),
        }


class TrainSplitDataset(_PairedDatasetBase):
    SPLIT = 'train'


class ValSplitDataset(_PairedDatasetBase):
    SPLIT = 'val'
