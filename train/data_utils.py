import csv

import torch
from torch.utils.data.sampler import Sampler



def get_folds(images, num_folds, balanced=False):
    if not 2 <= num_folds <= len(images):
        raise ValueError('num_folds must be between 2 and the number of images')

    images_folds = []
    if balanced:
        # Keep every image: 50/4 becomes 13, 13, 12, 12.
        base_size, remainder = divmod(len(images), num_folds)
        start = 0
        for i in range(num_folds):
            fold_size = base_size + (1 if i < remainder else 0)
            images_folds.append(images[start:start + fold_size])
            start += fold_size
    else:
        # Compatibility mode for resuming checkpoints made by the baseline.
        fold_size = len(images) // num_folds
        for i in range(num_folds):
            images_folds.append(images[i * fold_size:(i + 1) * fold_size])

    folds = []
    for i in range(len(images_folds)):
        current_fold = {
            'validation': images_folds[i],
            'training': sum([sl for j, sl in enumerate(images_folds) if j != i], [])
        }
        folds.append(current_fold)

    return folds


class SubsetSequentialSampler(Sampler):
    """ Samples elements sequentially from a given list of indices, always in the same order.

    Args:
        indices (list): a list of indices
    """
    def __init__(self, indices, data_source=None):
        # PyTorch 2.2+ removed the unused data_source argument from Sampler.
        # Calling the base initializer without it also works on older releases.
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class SubsetRandomSampler(Sampler):
    """ Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (list): a list of indices
    """

    def __init__(self, indices, data_source=None):
        # PyTorch 2.2+ removed the unused data_source argument from Sampler.
        # Calling the base initializer without it also works on older releases.
        super().__init__()
        self.indices = indices

    def __iter__(self):
        return (self.indices[i.item()] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def save_to_csv(data, filepath):
    """ Writes a given data into a .csv.

    """
    with open(filepath, 'a') as file:
        writer = csv.writer(file)
        writer.writerows(data)
