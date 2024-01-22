# Road Segmentation

In this task, we wish to train a model to segment road from some areas in google map.

### Dataset: DeepGlobe Road Dataset

<img src="https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-21-134809.png" style="zoom:50%;" />

[Link](https://aistudio.baidu.com/datasetdetail/141168)

**Folder structure**:

```txt
deepglobe/
├── readme.md
├── test.txt
├── train/
├── train.txt
├── valid/
└── val.txt
```



Modify the `data.py` to load the dataset into memory:

```python
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

PATH = 'deepglobe'
def load_train_dataset():
    print('Loading training dataset...')
    with open(os.path.join(PATH, 'train.txt'), 'r') as f: # Here we only use a subset of the training dataset
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
```





### Training

**Device in this lab**:

<img src="https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-21-135426.png" style="zoom:67%;" />

#### Configurations & Results



##### Training Round-1

**Configuration**:

```yaml
model-conf:
  num_classes: 2

train-conf:
  w_c: True
  w_d: False
  num_epochs: 25
  batch_size: 2
  lr: 5e-5
```

**Dataset size**:

<img src="https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-21-165516.png" style="zoom:67%;" />

**Loss & MIoU curves**:

![Loss-Epoch](https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-21-164031.png)

![MIoU-Epoch](https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-21-164210.png)

**Test**:

We deployed the model trained at the 25th epoch to the test set and observed some of the results:

<img src="https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-22-021151.png" style="zoom:67%;" />



##### Training Round-2

**Configuration**:

```yaml
model-conf:
  num_classes: 2

train-conf:
  w_c: True
  w_d: False
  num_epochs: 50
  batch_size: 2
  lr: 5e-5
```

**Dataset Size**:

<img src="https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-21-170006.png" style="zoom:67%;" />

**Loss & MIoU curves**:

![Loss-Epoch](https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-22-021241.png)

![MIoU-EPoch](https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-22-021318.png)

**Test**:

We deployed the model trained at the 40th epoch to the test set and observed some of the results:

<img src="https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-22-022044.png" style="zoom:67%;" />