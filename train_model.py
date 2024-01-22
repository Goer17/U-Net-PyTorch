import os
import yaml
from model.UNet import UNet
from model.Trainer import Trainer
from data import train_dataset, val_dataset

if not os.path.exists('model-params'):
    os.makedirs('model-params')
if not os.path.exists('result'):
    os.makedirs('result')

with open('config.yaml', 'r') as conf:
    conf = yaml.safe_load(conf)
    model = UNet(num_classes=int(conf['model-conf']['num_classes']))
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset.data,
        val_dataset=val_dataset.data,
        w_c=bool(conf['train-conf']['w_c']),
        w_d=bool(conf['train-conf']['w_d']),
        num_epochs=int(conf['train-conf']['num_epochs']),
        batch_size=int(conf['train-conf']['batch_size']),
        lr=float(conf['train-conf']['lr'])
    )
    trainer.train()