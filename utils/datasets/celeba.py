import os
import zipfile 
import gdown
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
import numpy as np
import torch

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class CelebADataset(Dataset):
    def __init__(self, root_dir=os.path.join(CUR_DIR, '../../data/celeba'), transform=None, chosen_indexes=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        dataset_folder = f'{root_dir}/img_align_celeba/'
        self.dataset_folder = os.path.abspath(dataset_folder)
        if not os.path.isdir(dataset_folder):
            download_url = 'https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
            download_path = f'{root_dir}/img_align_celeba.zip'
            gdown.download(download_url, download_path, quiet=False)
            with zipfile.ZipFile(download_path, 'r') as ziphandler:
                ziphandler.extractall(root_dir)

        image_names = os.listdir(self.dataset_folder)

        self.transform = transform
        self.chosen_indexes = chosen_indexes
        image_names = natsorted(image_names)
        
        self.filenames = []
        self.annotations = []
        with open(f'{root_dir}/list_attr_celeba.txt') as f:
            for i, line in enumerate(f.readlines()):
                line = re.sub(' *\n', '', line)
                if i == 0:
                    self.header = re.split(' +', line)
                else:
                    values = re.split(' +', line)
                    filename = values[0]
                    attrs = [int(v) for v in values[1:]]
                    flag = False
                    if self.chosen_indexes is None:
                        flag = True
                    else:
                        for idx in self.chosen_indexes:
                            if attrs[idx] == 1:
                                flag = True
                                break
                    if flag:
                        self.filenames.append(filename)
                        self.annotations.append([int(v) for v in values[1:]])
        self.annotations = np.array(self.annotations)    
        if self.chosen_indexes is not None:
            self.annotations = self.annotations[:, chosen_indexes]
              
    def __len__(self): 
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = self.annotations[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, {'filename': img_name, 'idx': idx, 'attributes': torch.tensor(img_attributes).long()}
    
