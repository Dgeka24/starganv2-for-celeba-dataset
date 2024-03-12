from torch.utils.data import Dataset
from torchvision import transforms
import re
import numpy as np
import torch
import random
from collections import defaultdict
import pickle
from munch import Munch

class DefaultDataset(Dataset):
    def __init__(self, dataset_orig):
        self.dataset_orig = dataset_orig
    
    def __len__(self):
        return len(self.dataset_orig)
    
    def __getitem__(self, idx):
        img, vocab = self.dataset_orig[idx]
        attributes = []
        for i, val in enumerate(vocab['attributes']):
            if val.item() == 1:
                attributes.append(i)
        attribute = random.choice(attributes)
        return img, attribute

    
class ReferenceDataset(Dataset):
    def __init__(self, dataset_orig, path_to_vocab=None):
        self.dataset_orig = dataset_orig
        self.attributes_idx = defaultdict(list)
        if path_to_vocab is not None:
            with open(path_to_vocab, 'rb') as handle:
                self.attributes_idx = pickle.load(handle)
        else:
            for img, vocab in self.dataset_orig:
                idx = vocab['idx']
                attributes = vocab['attributes']
                for i, attribute in enumerate(attributes):
                    if attribute.item() == 1:
                        self.attributes_idx[i].append(idx)
    
    def __len__(self):
        return len(self.dataset_orig)
    
    def __getitem__(self, idx):
        img, vocab = self.dataset_orig[idx]
        attributes = []
        for i, val in enumerate(vocab['attributes']):
            if val.item() == 1:
                attributes.append(i)
        attribute = random.choice(attributes)
        idx1 = random.choice(self.attributes_idx[attribute])
        img1, _ = self.dataset_orig[idx1]
        return img, img1, attribute


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})