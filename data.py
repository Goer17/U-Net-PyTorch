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

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def load_train_dataset():
    print('Loading training dataset...')
    # TODO
    return []

def load_val_dataset():
    print('Loading validation dataset...')
    # TODO
    return []

def load_test_dataset():
    print("Loading test dataset...")
    # TODO
    return []

train_dataset = LazyLoader(load_train_dataset)
val_dataset = LazyLoader(load_val_dataset)
test_dataset = LazyLoader(load_test_dataset)