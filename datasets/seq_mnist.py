from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from backbone.MNISTMLP_our import MNISTMLP_our
import torch.nn.functional as F
from utils.conf import base_path
from PIL import Image
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders_minist, get_previous_gan_loader
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize_one
import numpy as np
import torchvision.transforms.functional as transofrms_f


class MyMNIST(MNIST):
    """
    Overrides the MNIST dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.ToTensor()
        super(MyMNIST, self).__init__(root, train,
                                      transform, target_transform, download)

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        original_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target, original_img


class SequentialMNIST(ContinualDataset):

    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def get_data_loaders_gan(self, gan_batch_size):
        train_dataset = MyMNIST(base_path() + 'MNIST', train=True,
                                  download=True, transform=None)

        return get_previous_gan_loader(train_dataset, gan_batch_size, self)

    def get_data_loaders(self):
        transform = self.TRANSFORM
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        transform, self.NAME)
        else:
            test_dataset = MNIST(base_path() + 'MNIST',
                                train=False, download=True, transform=transform)

        train, test = store_masked_loaders_minist(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = self.TRANSFORM
        # transform = transforms.ToTensor()
        train_dataset = MyMNIST(base_path() + 'MNIST',
                                train=True, download=True, transform=transform)
        train_mask = np.logical_and(np.array(train_dataset.targets) >= self.i -
            self.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < self.i)

        train_dataset.data = train_dataset.data[train_mask]
        train_dataset.targets = np.array(train_dataset.targets)[train_mask]

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size, shuffle=True)
        return train_loader

    @staticmethod
    def get_backbone():
        return MNISTMLP_our(28 * 28, SequentialMNIST.N_TASKS
                        * SequentialMNIST.N_CLASSES_PER_TASK * 50)

    @staticmethod
    def get_transform_mnist():
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return transform

    @staticmethod
    def get_rotation_transform():
        transform = transforms.Compose([transforms.ToPILImage(), Rotation(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        return transform

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = None
        transform = transforms.Normalize((0.5,), (0.5,))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = None
        transform = DeNormalize_one(0.5, 0.5)
        # return lambda x: x
        return transform

    @staticmethod
    def get_test_transform():
        test_transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        return test_transform


class Rotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg_min: int = -10, deg_max: int = 10) -> None:  # 设为0,结果就不会再随机了
        """
        Initializes the rotation with a random angle.
        :param deg_min: lower extreme of the possible random angle
        :param deg_max: upper extreme of the possible random angle
        """
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.degrees = np.random.uniform(self.deg_min, self.deg_max)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Applies the rotation.
        :param x: image to be rotated
        :return: rotated image
        """
        return transofrms_f.rotate(x, self.degrees)
