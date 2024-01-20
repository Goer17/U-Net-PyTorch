# An implementation of U-Net based on PyTorch

U-Net is a convolutional neural network architecture, originally developed for biomedical image segmentation. It features a symmetric, U-shaped structure with a contracting path to capture context and a symmetric expanding path for precise localization, enabling it to effectively segment images even with limited training data. U-Net is widely used in medical image analysis and other segmentation tasks.

[arXiv](https://arxiv.org/abs/1505.04597)

U-Net architecture in the original paper：

![U-Net](https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-20-144614.png)

#### Getting Started with Your Project

```txt
.
├── LICENSE
├── README.md
├── config.yaml
├── data.py
├── model
│   ├── Trainer.py
│   └── UNet.py
├── model-params
├── requirements.txt
├── result
└── train_model.py
```

1. **Install Dependencies**

   Begin by installing the required packages using `pip`

   ```shell
   pip install -r requirements.txt
   ```

2. **Configure Your Model**

   Next, tailor the `config.yaml` file to your specific requirements

   ```yaml
   model-conf:
     num_classes: 2 # Define the number of classes
   
   train-conf:
     w_c: True # Include class frequency in the loss function pixel weight
     w_d: False # Include pixel border distance in the loss function pixel weight (TODO)
     num_epochs: 20 # Set the number of training epochs
     batch_size: 8 # Specify the batch size
     lr: 0.0001 # Set the learning rate
   ```

3. **Prepare Your Dataset**

   In `data.py`, complete the dataset preparation stage:

   ```python
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
   ```

   > Ensure each function returns a list of tuples `(img, mask)`. Here, `img` should be a 3-channel tensor (e.g., (3, 512, 512)), and `mask` a 1-channel integer tensor (e.g., (512, 512)). The dimensions of `mask` should match `img`, and both height and width must be divisible by 32.

4. **Initiate Training**

   Start the training process with the following command:

   ```shell
   python train_model.py
   ```

   ![Training](https://typora-1313035735.cos.ap-nanjing.myqcloud.com/img/2024-01-20-153145.png)

   Upon completion, the training and MIoU curves will be saved in the `result` directory. The model parameters for each epoch are stored in `model-params` as `.pth` files.

This revised version maintains the original technical accuracy while enhancing clarity and formality.



#### Todo List

- [ ] Complete `w_d` generation task

- [ ] Complete road segmentation experiment

- [ ] Complete object segmentation experiment on VOC2012 dataset

	

