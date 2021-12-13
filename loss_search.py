import argparse
import os
import sys
import yaml
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np

import torch

from utils.metrics import print_num_params

from core.trainer import train_epoch, eval_epoch
from core.utils import loss_adjust_cross_entropy, cross_entropy,loss_adjust_cross_entropy_cdt
from core.utils import get_init_dy, get_init_ly, get_train_w, get_val_w

from dataset.ImageNet_LT import ImageNetLTDataLoader
from dataset.cifar10 import load_cifar10
from dataset.iNaturalist import INAT
from dataset.cifar100 import load_cifar100

import torchvision.models as models
from models.ResNet import ResNet32

import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
assert torch.cuda.is_available()
assert torch.backends.cudnn.enabled


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config',
                    default="configs/config.yaml", type=str)
args = parser.parse_args()
with open(args.config, mode='r') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)
device = args["device"]
dataset = args["dataset"]
if dataset == 'Cifar10':
    num_classes = 10
    model = ResNet32(num_classes=num_classes)
    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_cifar10(
        train_size=args["train_size"], val_size=args["val_size"],
        balance_val=args["balance_val"], batch_size=args["low_batch_size"],
        train_rho=args["train_rho"],
        image_size=32, path=args["datapath"])
elif dataset == 'Cifar100':
    num_classes = 100
    model = ResNet32(num_classes=num_classes)
    # if torch.cuda.device_count()>1:
    #     model = nn.DataParallel(model, device_ids=[0, 1])
    train_loader, val_loader, test_loader, eval_train_loader, eval_val_loader, num_train_samples, num_val_samples = load_cifar100(
        train_size=args["train_size"], val_size=args["val_size"],
        balance_val=args["balance_val"], batch_size=args["low_batch_size"],
        train_rho=args["train_rho"],
        image_size=32, path=args["datapath"])
elif dataset == 'ImageNet':
    num_classes = 1000
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=[0, 1])

    train_loader = ImageNetLTDataLoader(
        args["datapath"],
        training=True, batch_size=args["low_batch_size"])
    val_loader = train_loader.split_validation()
    test_loader = ImageNetLTDataLoader(
        args["datapath"],
        training=False, batch_size=args["low_batch_size"])

    num_train_samples, num_val_samples = train_loader.get_train_val_size()
    eval_train_loader = ImageNetLTDataLoader(
        args["datapath"],
        training=True, batch_size=512)
    eval_val_loader = eval_train_loader.split_validation()

elif dataset == 'INAT':
    num_classes = 8142
    model = models.resnet50(pretrained=False, num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=[0, 1])
    train_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                        is_train=True, split=1)
    num_train_samples = train_loader.get_class_size()
    train_loader = DataLoader(train_loader, batch_size=args["low_batch_size"],num_workers=6)
    val_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                      is_train=True, split=2)
    num_val_samples = val_loader.get_class_size()
    val_loader = DataLoader(val_loader, batch_size=args["low_batch_size"],num_workers=6)

    test_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/val2018.json',
                       is_train=False, split=0)
    test_loader.get_class_size()
    test_loader= DataLoader(test_loader,batch_size=512,num_workers=6)

    eval_train_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                        is_train=False, split=1)
    eval_train_loader = DataLoader(eval_train_loader, batch_size=512,num_workers=6)
    eval_val_loader = INAT('/home/eeres/mili/data/inat_2018/', '/home/eeres/mili/data/inat_2018/train2018.json',
                      is_train=False, split=2)
    eval_val_loader = DataLoader(eval_val_loader, batch_size=512,num_workers=6)

args["num_classes"] = num_classes

print_num_params(model)

if args["checkpoint"] != 0:
    model = torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth')
    # model.load_state_dict(torch.load(f'{args["save_path"]}/epoch_{args.checkpoint}.pth'))
model = model.to(device)

criterion = nn.CrossEntropyLoss()

dy = get_init_dy(args, num_train_samples)
ly = get_init_ly(args, num_train_samples)
w_train = get_train_w(args, num_train_samples)
w_val = get_val_w(args, num_val_samples)

print(f"w_train: {w_train}\nw_val: {w_val}")
print('ly', ly, '\n dy', dy)

print("train data size",len(train_loader.dataset),len(train_loader))

if dataset == 'Cifar10' or dataset == 'Cifar100':
    up_start_epoch=args["up_configs"]["start_epoch"]
    if "low_lr_multiplier" in args:
        def warm_up_with_multistep_lr_low(epoch): return (epoch+1) / args["low_lr_warmup"] \
            if epoch < args["low_lr_warmup"] \
            else args["low_lr_multiplier"][len([m for m in args["low_lr_schedule"] if m <= epoch])]
    else:
        def warm_up_with_multistep_lr_low(epoch): return (epoch+1) / args["low_lr_warmup"] \
            if epoch < args["low_lr_warmup"] \
            else 0.1**len([m for m in args["low_lr_schedule"] if m <= epoch])

    def warm_up_with_multistep_lr_up(epoch): return (epoch-up_start_epoch+1) / args["up_lr_warmup"] \
        if epoch-up_start_epoch < args["up_lr_warmup"] \
        else 0.1**len([m for m in args["up_lr_schedule"] if m <= epoch])
    train_optimizer = optim.SGD(params=model.parameters(),
                                lr=args["low_lr"], momentum=0.9, weight_decay=1e-4)
    val_optimizer = optim.SGD(params=[{'params': dy}, {'params': ly}],
                              lr=args["up_lr"], momentum=0.9, weight_decay=1e-4)
    train_lr_scheduler = optim.lr_scheduler.LambdaLR(
        train_optimizer, lr_lambda=warm_up_with_multistep_lr_low)
    val_lr_scheduler = optim.lr_scheduler.LambdaLR(
        val_optimizer, lr_lambda=warm_up_with_multistep_lr_up)
elif dataset == 'ImageNet' or dataset == 'INAT':
    warm_up_with_cosine_lr_low = lambda epoch:\
         (epoch+1) / args["low_lr_warmup"] if epoch < args["low_lr_warmup"] \
            else 0.5*(math.cos((epoch - args["low_lr_warmup"]) / (args["epoch"]-args["low_lr_warmup"])*math.pi)+1)
    up_start_epoch=args["up_configs"]["start_epoch"]
    def warm_up_with_multistep_lr_low(epoch): return (epoch+1) / args["low_lr_warmup"] \
        if epoch < args["low_lr_warmup"] \
        else 0.1**len([m for m in args["low_lr_schedule"] if m <= epoch])

    def warm_up_with_multistep_lr_up(epoch): return (epoch-up_start_epoch+1) / args["up_lr_warmup"] \
        if epoch-up_start_epoch < args["up_lr_warmup"] \
        else 0.1**len([m for m in args["up_lr_schedule"] if m <= epoch])

    train_optimizer = optim.SGD(
        params=model.parameters(),
        lr=args["low_lr"], momentum=0.9, weight_decay=5e-4)
    val_optimizer = optim.SGD(params=[{'params': dy}, {'params': ly}],
                              lr=args["up_lr"], momentum=0.9, weight_decay=5e-4)
    train_lr_scheduler = optim.lr_scheduler.LambdaLR(
        train_optimizer, lr_lambda=warm_up_with_cosine_lr_low)
    val_lr_scheduler = optim.lr_scheduler.LambdaLR(
        val_optimizer, lr_lambda=warm_up_with_cosine_lr_low  )


if args["save_path"] is None:
    import time
    args["save_path"] = f'./results/{int(time.time())}'
if not os.path.exists(args["save_path"]):
    os.makedirs(args["save_path"])

assert(args["checkpoint"] == 0)

torch.save(model, f'{args["save_path"]}/init_model.pth')
logfile = open(f'{args["save_path"]}/logs.txt', mode='w')
dy_log = open(f'{args["save_path"]}/dy.txt', mode='w')
ly_log = open(f'{args["save_path"]}/ly.txt', mode='w')
err_log = open(f'{args["save_path"]}/err.txt', mode='w')
with open(f'{args["save_path"]}/config.yaml', mode='w') as config_log:
    yaml.dump(args, config_log)

save_data = {"ly": [], "dy": [], "w_train": [],
             "train_err": [], "balanced_train_err": [], 
             "val_err": [], "balanced_val_err": [], "test_err": [], "balanced_test_err": []}
with open(f'{args["save_path"]}/result.yaml', mode='w') as log:
    yaml.dump(save_data, log)

for i in range(args["checkpoint"], args["epoch"]+1):
    if i % args["checkpoint_interval"] == 0:
        torch.save(model, f'{args["save_path"]}/epoch_{i}.pth')

    if i % args["eval_interval"] == 0:
        if args["up_configs"]["dy_init"]=="CDT":
            print("CDT")
            text, loss, train_err, balanced_train_err = eval_epoch(eval_train_loader, model,
                                                               loss_adjust_cross_entropy_cdt, i, ' train_dataset', args,
                                                               params=[dy, ly, w_train])
                                                            
        else:
            text, loss, train_err, balanced_train_err = eval_epoch(eval_train_loader, model,
                                                               loss_adjust_cross_entropy, i, ' train_dataset', args,
                                                               params=[dy, ly, w_train])
        logfile.write(text+'\n')
        text, loss, val_err, balanced_val_err = eval_epoch(eval_val_loader, model,
                                                           cross_entropy, i, ' val_dataset', args, params=[dy, ly, w_val])
        logfile.write(text+'\n')

        text, loss, test_err, balanced_test_err = eval_epoch(test_loader, model,
                                                             cross_entropy, i, ' test_dataset', args, params=[dy, ly])
        logfile.write(text+'\n')
    save_data["train_err"].append(train_err)
    save_data["balanced_train_err"].append(balanced_train_err)
    save_data["val_err"].append(val_err)
    save_data["balanced_val_err"].append(balanced_val_err)
    save_data["test_err"].append(test_err)
    save_data["balanced_test_err"].append(balanced_test_err)

    save_data["dy"].append(dy.detach().cpu().numpy().tolist())
    save_data["ly"].append(ly.detach().cpu().numpy().tolist())
    save_data["w_train"].append(w_train.detach().cpu().numpy().tolist())

    with open(f'{args["save_path"]}/result.yaml', mode='w') as log:
        yaml.dump(save_data, log)

    print('ly', ly, '\n dy', dy, '\n')

    train_epoch(i, model, args,
                low_loader=train_loader, low_criterion=loss_adjust_cross_entropy,
                low_optimizer=train_optimizer, low_params=[dy, ly, w_train],
                up_loader=val_loader, up_optimizer=val_optimizer,
                up_criterion=cross_entropy, up_params=[dy, ly, w_val])

    logfile.write(str(dy)+str(ly)+'\n\n')
    dy_log.write(f'{dy.detach().cpu().numpy()}\n')
    ly_log.write(f'{ly.detach().cpu().numpy()}\n')
    err_log.write(f'{train_err} {val_err} {test_err}\n')
    logfile.flush()
    dy_log.flush()
    ly_log.flush()
    err_log.flush()
    train_lr_scheduler.step()
    val_lr_scheduler.step()

logfile.close()
dy_log.close()
ly_log.close()
err_log.close()
torch.save(model, f'{args["save_path"]}/final_model.pth')
