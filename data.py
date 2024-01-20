import os
import torch
from PIL import Image
from torchvision import transforms

class LazyLoader:
    def __init__(self, load_func):
        self.load_func = load_func
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = self.load_func()
        return self._data

PATH = 'deepglobe'
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_train_dataset():
    print('Loading training dataset...')
    with open(os.path.join(PATH, 'train_small.txt'), 'r') as f:
        train_dataset = []
        while (line := f.readline()) != '':
            sat, mask = line.rstrip().split(' ')
            sat = Image.open(os.path.join(PATH, sat)).convert('RGB')
            sat = to_tensor(sat)
            sat = normalize(sat)
            
            mask = Image.open(os.path.join(PATH, mask)).convert('L')
            mask = to_tensor(mask)
            mask = (mask > 0).byte().squeeze(dim=0)
            
            train_dataset.append((sat, mask))
        print(f'Load completed, dataset size = {len(train_dataset)}')
        return train_dataset

def load_val_dataset():
    print('Loading validation dataset...')
    with open(os.path.join(PATH, 'val.txt'), 'r') as f:
        val_dataset = []
        while (line := f.readline()) != '':
            sat, mask = line.rstrip().split(' ')
            sat = Image.open(os.path.join(PATH, sat)).convert('RGB')
            sat = to_tensor(sat)
            sat = normalize(sat)
            
            mask = Image.open(os.path.join(PATH, mask)).convert('L')
            mask = to_tensor(mask)
            mask = (mask > 0).byte().squeeze(dim=0)
            
            val_dataset.append((sat, mask))
        print(f'Load completed, dataset size = {len(val_dataset)}')
        return val_dataset

def load_test_dataset():
    print("Loading test dataset...")
    with open(os.path.join(PATH, 'test.txt'), 'r') as f:
        test_dataset = []
        while (line := f.readline()) != '':
            sat, mask = line.rstrip().split(' ')
            sat = Image.open(os.path.join(PATH, sat)).convert('RGB')
            sat = to_tensor(sat)
            sat = normalize(sat)
            
            mask = Image.open(os.path.join(PATH, mask)).convert('L')
            mask = to_tensor(mask)
            mask = (mask > 0).byte().squeeze(dim=0)
            
            test_dataset.append((sat, mask))
        print(f'Load completed, dataset size = {len(test_dataset)}')
        return test_dataset

train_dataset = LazyLoader(load_train_dataset)
val_dataset = LazyLoader(load_val_dataset)
test_dataset = LazyLoader(load_test_dataset)