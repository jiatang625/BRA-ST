import torch
import csv
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn, device
import time
from torch.optim.lr_scheduler import StepLR
import os
from sklearn.metrics import precision_recall_fscore_support
device = torch.device("cuda:0"if torch.cuda.is_available() else"cpu")
from method import Ours
transform=transforms.Compose([transforms.Resize((224,224),antialias=True),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
transform1=transforms.Compose([transforms.Resize((224,224),antialias=True),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
#路径是自己电脑里所对应的路径
datapath1= r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/FLAVRUCF_2X/mergetrain1'
datapath2= r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/FLAVRUCF_2X/mergetrain'
datapath3= r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/IFRNetucf101_2x/mergetest1'
datapath4= r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/IFRNetucf101_2x/mergetest'
txtpath1 = r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/FLAVRUCF_2X/train1.txt'
txtpath2 = r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/FLAVRUCF_2X/train.txt'
txtpath3 = r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/IFRNetucf101_2x/test1.txt'
txtpath4 = r'/media/test/79019239-2054-4917-bd0a-9324b4811e87/tangjia/usedataset/Our/IFRNetucf101_2x/test.txt'
class MyDataset1(Dataset):
    def __init__(self,txtpath,datapath):
        imgs = []
        datainfo = open(txtpath, 'r')
        # print(datainfo)
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            temp = words[0].split('.')
            words0 = temp[0].split('i')
            n = int(words0[1])
            imagelist = []
            all_files_exist = True
            for i in range(5):
                s = '%03d' % n
                temp = words0[0] + 'i' + str(s) + '.jpg'
                image_path = os.path.join(datapath,temp)
                if not os.path.exists(image_path):
                    all_files_exist = False
                    break
                imagelist.append(temp)
                n += 1
            if all_files_exist:
               imgs.append((imagelist, words[1]))
        self.datapath = datapath
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        piclist,label = self.imgs[index]
        picr1 = Image.open(os.path.join(self.datapath,piclist[0]))
        picr1=transform(picr1)
        picr2= Image.open(os.path.join(self.datapath,piclist[1]))
        picr2 = transform(picr2)
        picr3 = Image.open(os.path.join(self.datapath,piclist[2]))
        picr3 = transform(picr3)
        picr4 = Image.open(os.path.join(self.datapath,piclist[3]))
        picr4 = transform(picr4)
        picr5 = Image.open(os.path.join(self.datapath,piclist[4]))
        picr5= transform(picr5)
        label = torch.tensor(int(label))
        return picr1,picr2,picr3,picr4,picr5,label
class MyDataset2(Dataset):
    def __init__(self,txtpath,datapath):
        imgs = []
        datainfo = open(txtpath, 'r')
        for line in datainfo:
            line = line.strip('\n')
            words = line.split()
            temp = words[0].split('.')
            words0 = temp[0].split('i')
            n = int(words0[1])
            imagelist = []
            all_files_exist = True
            for i in range(5):
                s = '%03d' % n
                temp = words0[0] + 'i' + str(s) + '.jpg'
                image_path = os.path.join(datapath,temp)
                if not os.path.exists(image_path):
                    all_files_exist = False
                    break
                imagelist.append(temp)
                n += 1
            if all_files_exist:
               imgs.append((imagelist, words[1]))
        self.datapath = datapath
        self.imgs = imgs
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        piclist,label = self.imgs[index]
        picn1 = Image.open(os.path.join(self.datapath,piclist[0]))
        picn1=transform1(picn1)
        picn2= Image.open(os.path.join(self.datapath,piclist[1]))
        picn2 = transform1(picn2)
        picn3 = Image.open(os.path.join(self.datapath,piclist[2]))
        picn3 = transform1(picn3)
        picn4 = Image.open(os.path.join(self.datapath,piclist[3]))
        picn4 = transform1(picn4)
        picn5 = Image.open(os.path.join(self.datapath,piclist[4]))
        picn5= transform1(picn5)
        label = torch.tensor(int(label))
        return picn1,picn2,picn3,picn4,picn5,label
data1 = MyDataset1(txtpath1,datapath1)
data_loader1 = DataLoader(data1,batch_size=16,shuffle=True)#32
data2 = MyDataset2(txtpath2,datapath2)
data_loader2 = DataLoader(data2,batch_size=16,shuffle=True)
data3 = MyDataset1(txtpath3,datapath3)
data_loader3 = DataLoader(data3)#32
data4 = MyDataset2(txtpath4,datapath4)
data_loader4 = DataLoader(data4)
model= Ours(depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,).to(device)
#weights = torch.tensor([1.0,3.0])
#loss_fn=nn.CrossEntropyLoss(weight=weights)
loss_fn=nn.CrossEntropyLoss()
loss_fn=loss_fn.to(device)
learning_rate=0.0001#0.0008345137614500873
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.00001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
total_train_step=0
total_test_step=0
# epoch=100
test_loss=[]
test_acc=[]
train_loss=[]
train_acc=[]
Epoch=[]
total_accuracy = 0
total_test_loss = 0
lr=0
acc=0
acc1=0
detections=[]
k=30#控制训练次数和写入excel
headers = ['epoch', 'train_acc', 'test_acc', 'precision', 'recall', 'f1score',  'train_loss', 'test_loss']
with open('result.csv', 'w', newline='') as f:  # 创建CSV文件
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    for i in range(k):
       correct = 0
       total = 0
       running_loss = 0
       print("第{}轮训练开始：".format(i+1))
       start_time=time.time()
       #训练步骤开始
       Epoch.append(i)
       model.train()
       for data1,data2 in zip(data_loader1,data_loader2):
           picr1, picr2, picr3, picr4, picr5, targerts1 = data1
           picn1, picn2, picn3, picn4, picn5, targerts2 = data2
           imgs1 = torch.cat((picr1, picr2, picr3, picr4, picr5), 1)
           imgs2 = torch.cat((picn1, picn2, picn3, picn4, picn5), 1)
           imgs1 = imgs1.to(device)
           imgs2 = imgs2.to(device)
           targerts1 = targerts1.to(device)
           targerts2 = targerts2.to(device)
           outputs = model(imgs1,imgs2)
           loss=loss_fn(outputs,targerts2)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           total_train_step=total_train_step+1
           if total_train_step%800==0:
               print("训练次数：{}，loss:{}".format(total_train_step,loss.item()))
           with torch.no_grad():
               y_pred = torch.argmax(outputs, dim=1)
               correct += (y_pred == targerts2).sum().item()
               total += targerts1.size(0)
               running_loss += loss.item()
       epoch_loss = running_loss / len(data_loader2.dataset)
       epoch_acc = correct / len(data_loader2.dataset)
       end_time=time.time()
       print("训练第{}次的时间是{}".format(i + 1, end_time - start_time))
       print("整体训练集上的Loss：{}".format(epoch_loss))
       print("整体训练集上的正确率：{}".format(epoch_acc))
       total_test_loss=0
       total_accuracy=0
       model.eval()
       with torch.no_grad():
           all_preds=[]
           all_labels=[]
           for data3,data4 in zip(data_loader3,data_loader4):
               picr1, picr2, picr3, picr4, picr5, targerts3 = data3
               picn1, picn2, picn3, picn4, picn5, targerts4 = data4
               imgs3 = torch.cat((picr1, picr2, picr3, picr4, picr5), 1)
               imgs4 = torch.cat((picn1, picn2, picn3, picn4, picn5), 1)
               imgs3 = imgs3.to(device)
               imgs4 = imgs4.to(device)
               targerts3 = targerts3.to(device)
               targerts4 = targerts4.to(device)
               outputs = model(imgs3, imgs4)
               loss=loss_fn(outputs,targerts4)
               total_test_loss=total_test_loss+loss.item()
               accuracy=(outputs.argmax(1)==targerts4).sum().item()
               total_accuracy = total_accuracy + accuracy
               all_preds.extend(outputs.argmax(1).cpu().numpy())
               all_labels.extend(targerts4.cpu().numpy())
           precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
       print("整体测试集上的Loss：{}".format(total_test_loss/len(data_loader4.dataset)))
       print("整体测试集上的正确率：{}".format(total_accuracy/len(data_loader4.dataset)))
       print("该轮训练的precision为:{}".format(precision))
       print("该轮训练的recall为:{}".format(recall))
       print("该轮训练的F1score为:{}".format(f1))
       total_test_step=total_test_step+1
       scheduler.step()
       torch.save(model, 'save-moudel/' + 'model' + str(i) + '.pkl')
       f_csv.writerow([str(i), str(epoch_acc), str(total_accuracy/len(data_loader4.dataset)), str(precision), str(recall), str(f1), str(epoch_loss), str(total_test_loss/len(data_loader4.dataset))])
f.close()
