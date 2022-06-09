import os
import shutil
from random import sample

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import Basic_Dataset
from model import Basic_Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Basic_Model()

SGD_optim = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(SGD_optim, T_max=10)

start_epoch = 0

if os.path.exists("checkpoints/last.pth"):
    checkpoint = torch.load("checkpoints/last.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    SGD_optim.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
elif not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

train_dataset = Basic_Dataset(r"../../datasets/plate")

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

best_acc_rate = -1

for epoch in range(start_epoch, 1000):
    acc = 0
    samples = 0
    total_loss = 0
    model.train()
    for i, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        output = model(image)
        loss = loss_fn(output, label)
        total_loss += loss.item()
        SGD_optim.zero_grad()
        loss.backward()
        SGD_optim.step()
        
        samples += len(image)
        acc += torch.sum(torch.argmax(output, dim=1) == label).item()
    acc_rate = acc / samples
    lr_scheduler.step()
    print("Epoch: {}, Loss: {}, Acc: {}".format(epoch, total_loss, acc_rate))
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": SGD_optim.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "loss": total_loss,
        "acc": acc_rate
    }
    torch.save(checkpoint, "checkpoints/last.pth")
    
    if acc_rate > best_acc_rate:
        best_acc_rate = acc_rate
        shutil.copyfile("checkpoints/last.pth", "checkpoints/best.pth")
    
        
        