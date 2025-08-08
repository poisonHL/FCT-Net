import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from Net3_FCT1 import Net
from dataloader1 import Datases_loader as dataloader
from Dice_BCEwithLogits import SoftDiceLoss as bcedice

def inverseDecayScheduler(step, initial_lr, gamma=10, power=0.9, max_iter=80):
    return initial_lr * ((1 - step / float(max_iter)) ** power)

def adjust_lr(optimizer, step, initial_lr):
    lr = inverseDecayScheduler(step, initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 4
lr = 0.0005
items = 40
model = Net().to(device)
criterion = bcedice()

# gpu_ids = [int(i) for i in list(cfg['gpu_ids'])]
# if len(gpu_ids) > 1:
#     model = nn.DataParallel(model, device_ids=gpu_ids)
# model = model.to(cfg["device"])

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=3e-5)
# optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=3e-5, lr=lr)

# savedir = r'/tmp/Segment/Hong/weights/Net3_315v2_260.pth'
# imgdir = r'/tmp/Segment/Deepcrack/CrackTree260/train_img'
# labdir = r'/tmp/Segment/Deepcrack/CrackTree260/train_lab'

# savedir = r'/tmp/Segment/Hong/weights/Net3_FCT478.pth'
# imgdir = r'/tmp/Segment/Deepcrack/CFTR478/train_img'
# labdir = r'/tmp/Segment/Deepcrack/CFTR478/train_lab'
savedir = r'/tmp/Segment/Hong/weights/Net3_FCT-9-6.pth'
imgdir = r'/tmp/Segment/Deepcrack/Deepcrack/train_img'
labdir = r'/tmp/Segment/Deepcrack/Deepcrack/train_lab'

imgsz = 512

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
trainsets = DataLoader(dataset, batch_size=batchsz, shuffle=True)

lossx = 0
# tp, tn, fp, fn = 0, 0, 0, 0
# accuracy, precision, recall, F1, ls_loss = [],[],[],[],[]
ls_loss = []
def train():
    for epoch in range(items):
        lossx = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        for idx, samples in enumerate(trainsets):
            img, lab = samples['image'], samples['mask']
            img, lab = img.to(device), lab.to(device)

            optimizer.zero_grad()
            pred = model(img)

            #BCE
            # pred = pred.view(-1, 1)
            # lab = lab.view(-1, 1)
            # loss = F.binary_cross_entropy_with_logits(pred, lab, reduction='mean')

            # CE
            # pred = pred.transpose(1, 3).transpose(1, 2).contiguous().view(-1, 2)
            # lab = lab.contiguous().view(-1)
            # lab = lab.long()
            # loss = criterion(pred, lab)

            #tversky
            # loss = tversky_loss(lab, pred)

            #Gfocal
            # B = pred.shape[0]
            # pred = pred.reshape(B, 512*512)
            # lab = lab.reshape(B, 512*512)
            # loss = criterion(pred, lab)

            loss = criterion(pred, lab)
            loss.backward()
            optimizer.step()

            lossx = lossx + loss
            # #CE
            # p = pred.argmax(1)
            # #
            # p = p.reshape(-1)
            # t = lab.reshape(-1)
            # tp_, fp_, tn_, fn_ = compute_confusion_matrix(p.detach().cpu().numpy(), t.detach().cpu().numpy())
            # tp = tp + tp_
            # fp = fp + fp_
            # tn = tn + tn_
            # fn = fn + fn_

            print('epoach:' + str(epoch+1) + '-----id:' + str(idx+1))

        # accuracy_, precision_, recall_, F1_ = compute_indexes(tp, fp, tn, fn)
        # accuracy.append(accuracy_)
        # precision.append(precision_)
        # recall.append(recall_)
        # F1.append(F1_)
        adjust_lr(optimizer, epoch, optimizer.param_groups[0]['lr'])
        lossx = lossx / dataset.num_of_samples()
        ls_loss.append(lossx.item())
        print('epoch'+str(epoch+1)+'---loss:'+str(lossx.item()))

    torch.save(model.state_dict(), savedir)

if __name__ == '__main__':
    train()

    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(F1)
    for i, loss in enumerate(ls_loss):
        print(loss, end='\n' if (i + 1) % 5 == 0 else ' ')
    str = 'loss = ' + str(ls_loss)
    # str = 'accuracy:' + str(accuracy) + '\nprecision:' + str(precision) + '\nrecall:' + str(recall) + '\nF1:' + str(F1) + '\nloss:' + str(ls_loss)
    filename = r'D:\Desktop\new2\weights\Net2_3.txt'
    with open(filename, mode='w', newline='') as f:
        f.writelines(str)