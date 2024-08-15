import torch

from typing import Callable
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from .datautils import DataCard

def set_device(config):
    # Set device
    device = config['WHICH_DEVICE']
    if device != 'cpu':    
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)    
    print('Device:\t', device)
    return device 


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def construct_parser():
    parser = ArgumentParser()

    parser.add_argument(
        '--dataset',
        metavar='--ds',
        type=str,
        help='Name of the dataset'
    )
    
    return parser

def acc_rate(x, y):
    a = x.cpu()
    b = y.cpu()
    acc = balanced_accuracy_score(a, b)
    return acc

# def rmse_error(x, y):
#     mse = F.mse_loss(y.flatten(), x.flatten()).sqrt()
#     return mse

def mse_error(x, y):
    a = x.cpu()
    b = y.cpu()
    mse = mean_squared_error(a, b)
    return mse


def collate_func(examples, datacard: DataCard) -> dict:

    # Init batch
    batch = {var : [] for var in datacard.all_vars}

    # Assemble batch
    for example in examples:
        for var in datacard.all_vars:
            batch[var].append(example[var])

    # Convert to tensors
    for var in datacard.cat_vars:
        batch[var] = torch.tensor(batch[var], dtype = torch.int64)

    for var in datacard.num_vars:
        batch[var] = torch.tensor(batch[var], dtype = torch.float32).reshape(-1, 1)

    for var in datacard.img_vars:
        batch[var] = torch.stack(batch[var])

    return batch


def make_loaders(datasets: dict, collator: Callable, batch_size: int = 32, num_workers: int = 0) -> dict:

    loaders = {}
    for split, dataset in datasets.items():

        # Add loader
        loaders[split] = DataLoader(
            dataset,
            shuffle= True if split == 'train' else False,
            collate_fn=collator,
            batch_size=batch_size,
            num_workers=num_workers
        )

    return loaders