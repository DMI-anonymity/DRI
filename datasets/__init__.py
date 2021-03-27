from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
}

GCL_NAMES = {
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
