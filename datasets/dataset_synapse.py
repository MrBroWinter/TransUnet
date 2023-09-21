# import os
# import random
# import h5py
# import numpy as np
# import torch
# from scipy import ndimage
# from scipy.ndimage.interpolation import zoom
# from torch.utils.data import Dataset


# def random_rot_flip(image, label):
#     k = np.random.randint(0, 4)
#     image = np.rot90(image, k)
#     label = np.rot90(label, k)
#     axis = np.random.randint(0, 2)
#     image = np.flip(image, axis=axis).copy()
#     label = np.flip(label, axis=axis).copy()
#     return image, label


# def random_rotate(image, label):
#     angle = np.random.randint(-20, 20)
#     image = ndimage.rotate(image, angle, order=0, reshape=False)
#     label = ndimage.rotate(label, angle, order=0, reshape=False)
#     return image, label


# class RandomGenerator(object):
#     def __init__(self, output_size):
#         self.output_size = output_size

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']

#         if random.random() > 0.5:
#             image, label = random_rot_flip(image, label)
#         elif random.random() > 0.5:
#             image, label = random_rotate(image, label)
#         x, y = image.shape
#         if x != self.output_size[0] or y != self.output_size[1]:
#             image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#         image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#         label = torch.from_numpy(label.astype(np.float32))
#         sample = {'image': image, 'label': label.long()}
#         return sample


# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None):
#         self.transform = transform  # using transform in torch!
#         self.split = split
#         self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
#         self.data_dir = base_dir

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         if self.split == "train":
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name+'.npz')
#             data = np.load(data_path)
#             image, label = data['image'], data['label']
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#             data = h5py.File(filepath)
#             image, label = data['image'][:], data['label'][:]

#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         return sample

import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import SimpleITK as sitk

def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    img = reader.Execute()
    vol = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = spacing[::-1]
    return vol, img, spacing

def random_rotate(image, label):
    
    angle = random.uniform(-15, 15)
    image = ndimage.rotate(image.copy(), angle, axes=(1, 2), reshape=False, cval=0)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_flip(image, label):
    if random.random() > 0.5:
        image = image[:, ::-1, :]
        label = label[::-1, :]
    else:
        image = image[:, :, ::-1]
        label = label[:, ::-1]
    return image, label

def gamma(ct):
    mn = ct.mean()
    sd = ct.std()
    gamma = np.random.uniform(0.75, 1.5)
    minm = ct.min()
    rnge = ct.max() - minm
    ct_res = np.power((ct.copy() - minm) / float(rnge + 1e-7), gamma) * rnge + minm

    ct_res = ct_res - ct_res.mean()
    ct_res = ct_res / (ct_res.std() + 1e-8) * sd
    ct_res = ct_res + mn
    return ct_res

def linear_shift(ct):
    mn = ct.min()
    coefficient = random.uniform(0.8, 1.2)
    ct_res = (ct.copy() - mn) * coefficient + mn
    return ct_res


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() < 0.5:
            image, label = random_rotate(image, label)
        if random.random() < 0.3:
            image, label = random_flip(image, label)
        if random.random() < 0.3:
            image = gamma(image)
        if random.random() < 0.3:
            image = linear_shift(image)
        c, x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32))# .unsqueeze(0)
        label = torch.FloatTensor(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = os.path.join(self.data_dir, vol_name + '.npz')
            data = np.load(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample