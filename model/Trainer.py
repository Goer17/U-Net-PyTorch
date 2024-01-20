import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.UNet import UNet
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model: UNet, train_dataset: list, val_dataset: list, w_c: bool, w_d: bool, num_epochs: int, batch_size: int, lr: float):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.w_c = None
        
        if w_c:
            self.w_c = 1 / self.__calculate_class_frequencies(self.val_dataset)
            
        if w_d:
            # TODO
            pass
    
    def __calculate_class_frequencies(self, dataset: list):
        class_counts = torch.zeros(self.model.num_classes).long()
        total_pixels = 0

        for _, mask in dataset:
            unique, counts = mask.unique(return_counts=True)
            class_counts.put_(unique.long(), counts.long(), accumulate=True)
            total_pixels += mask.numel()

        class_frequencies = class_counts / total_pixels

        return class_frequencies + 1e-3
    
    def __calculate_MIoU(self, pred: torch.tensor, target: torch.LongTensor):
        IoUs = []
        pred = torch.argmax(pred, dim=1).view(-1)
        target = target.view(-1)
        
        for cls in range(self.model.num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                IoUs.append(float('nan'))
            else:
                IoUs.append(float(intersection / union))
        
        IoUs = torch.tensor(IoUs)
        
        return IoUs[~torch.isnan(IoUs)].mean()
    
    def train_epoch(self, epoch: int):
        # Training
        self.model.train()
        avg_loss = 0
        avg_MIoU = 0
        for img, mask in tqdm(self.train_dataloader, desc=f'Training Epoch {epoch}', unit='batch'):
            self.optimizer.zero_grad()
            img, mask = img.to(self.device), mask.long().to(self.device)
            pred = self.model(img)
            loss = self.criterion(pred, mask)
            loss.backward()
            self.optimizer.step()
            
            avg_loss += loss
            with torch.no_grad():
                MIoU = self.__calculate_MIoU(pred, mask)
                avg_MIoU += MIoU
        avg_loss /= len(self.train_dataloader)
        self.train_loss_rec.append(avg_loss.data.cpu().item())
        avg_MIoU /= len(self.train_dataloader)
        self.train_MIoU_rec.append(avg_MIoU)
        
        # Validation
        self.model.eval()
        avg_loss = 0
        avg_MIoU = 0
        with torch.no_grad():
            for img, mask in self.val_dataloader:
                img, mask = img.to(self.device), mask.long().to(self.device)
                pred = self.model(img)
                
                loss = self.criterion(pred, mask)
                avg_loss += loss
                
                MIoU = self.__calculate_MIoU(pred, mask)
                avg_MIoU += MIoU
            avg_loss /= len(self.val_dataloader)
            self.val_loss_rec.append(avg_loss.data.cpu().item())
            avg_MIoU /= len(self.val_dataloader)
            self.val_MIoU_rec.append(avg_MIoU.data.cpu().item())
        
        print(f'Trainging Loss = {self.train_loss_rec[-1] :.4}, Training MIoU = {self.train_MIoU_rec[-1] :.4}')
        print(f'Validation Loss = {self.val_loss_rec[-1] :.4}, Validation MIoU = {self.val_MIoU_rec[-1] :.4}')    
        
    def train(self):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        if self.w_c is not None: self.w_c = self.w_c.to(device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=self.w_c)
        
        self.model.to(device=self.device)
        print(f'Start Training... (Device: {self.device})')
        self.train_loss_rec = []
        self.train_MIoU_rec = []
        self.val_loss_rec = []
        self.val_MIoU_rec = []
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for epoch in range(1, self.num_epochs + 1):
            self.train_epoch(epoch=epoch)
            # saving the model
            model_state_dict = self.model.state_dict()
            torch.save(model_state_dict, f'model-params/epoch-{epoch}.pth')
        
        epoch_t = range(1, self.num_epochs + 1)
        plt.figure()
        plt.title('Loss-Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(epoch_t, self.train_loss_rec, label='Training')
        plt.plot(epoch_t, self.val_loss_rec, label='Validation')
        plt.legend()
        plt.savefig('result/loss.png')
        
        plt.figure()
        plt.title('MIoU-Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MIoU')
        plt.plot(epoch_t, self.train_MIoU_rec, label='Training')
        plt.plot(epoch_t, self.val_MIoU_rec, label='Validation')
        plt.legend()
        plt.savefig('result/MIoU.png')