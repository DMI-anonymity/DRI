from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from typing import Tuple
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
import torchvision.transforms.functional as transofrms_f


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders_minist(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    batch_size = setting.args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK

    return train_loader, test_loader


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    batch_size = setting.args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_previous_gan_loader(train_dataset: datasets, batch_size: int, setting: ContinualDataset):
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
                                setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
                                < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    train_x1 = []
    train_x2 = []
    train_x1_label = []
    for label_id in set(train_dataset.targets):
        idx = (train_dataset.targets == label_id)
        data = train_dataset.data[idx]
        labels = train_dataset.targets[idx]
        x2_data = list(data)
        np.random.shuffle(x2_data)
        train_x1.extend(data)
        train_x2.extend(x2_data)
        train_x1_label.extend(labels)

    in_channels = train_dataset.data.shape[-1]
    img_size = train_dataset.data.shape[2]

    gan_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            setting.get_normalization_transform(),
        ]
    )

    train_dataset = DaganDataset(train_x1, train_x1_label, train_x2, gan_transform, setting.get_denormalization_transform())

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def get_buffer_loaders_gan(buf_x, buf_y, batch_size, setting):
    train_x1 = []
    train_x2 = []
    train_x1_label = []
    for label_id in set(buf_y):
        idx = (buf_y == label_id)
        data = buf_x[idx]
        labels = buf_y[idx]
        x2_data = list(data)
        np.random.shuffle(x2_data)
        train_x1.extend(data)
        train_x2.extend(x2_data)
        train_x1_label.extend(labels)

    in_channels = buf_x.shape[-1]
    img_size = buf_x.shape[2]

    gan_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            setting.get_normalization_transform(),
        ]
    )

    train_dataset = DaganDataset(train_x1, train_x1_label, train_x2, gan_transform, setting.get_denormalization_transform())

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


class DaganDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1_examples, x1_labels, x2_examples, transform, denormalization):
        assert len(x1_examples) == len(x2_examples)
        self.x1_examples = x1_examples
        self.x1_labels = x1_labels
        self.x2_examples = x2_examples
        self.transform = transform
        self.denormalization = denormalization

    def __len__(self):
        return len(self.x1_examples)

    def __getitem__(self, idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.transform(self.x1_examples[idx]), self.transform(self.x2_examples[idx])


class AugmentRotation(object):
    """
    Defines a fixed rotation for a numpy array.
    """

    def __init__(self, deg_min: int = 90, deg_max: int = 90) -> None:
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
