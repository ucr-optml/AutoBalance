from math import ceil
from PIL.Image import BICUBIC
from PIL import Image
from torchvision.datasets.cifar import CIFAR100, CIFAR10
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, Resize, RandomAffine
from torchvision.transforms import ToTensor, Normalize

from torch.utils.data import Subset,Dataset, Sampler
import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader
import numpy as np
import random

class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets]) # Actually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch
            
def load_cifar100(train_size=400,train_rho=0.01,val_size=100,val_rho=0.01,image_size=32,batch_size=128,num_workers=4,path='./data',num_classes=100,balance_val=False):
    train_transform = Compose([
        RandomCrop(32,padding=4),
        #Resize(image_size, BICUBIC),
        #RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02), shear=2, fillcolor=(124,117,104)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    test_transform = Compose([
        #Resize(image_size, BICUBIC),    
        ToTensor(),
        Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    train_dataset = CIFAR100(root=path, train=True, transform=train_transform, download=True)
    test_dataset = CIFAR100(root=path, train=False, transform=test_transform, download=True)
    train_x,train_y = np.array(train_dataset.data), np.array(train_dataset.targets)
    #test_x, test_y = test_dataset.data, test_dataset.targets
    
    # num_train_samples=[]
    # num_val_samples=[]
    # train_mu=train_rho**(1./(num_classes-1.))
    # val_mu=val_rho**(1./(num_classes-1.))
    # for i in range(num_classes):
    #     num_train_samples.append(round(train_size*(train_mu**i)))
    #     num_val_samples.append(round(val_size*(val_mu**i)))

    total_size=500
    num_total_samples=[]
    num_train_samples=[]
    num_val_samples=[]
    train_mu=train_rho**(1./(num_classes-1.))
    val_mu=val_rho**(1./(num_classes-1.))

    for i in range(num_classes):
        num_total_samples.append(round(total_size*(train_mu**i)))
        num_val_samples.append(max(1,round(100*(val_mu**i))))
        num_train_samples.append(num_total_samples[-1]-num_val_samples[-1])
        
        #num_total_samples.append(ceil(total_size*(train_mu**i)))
        #num_train_samples.append(ceil(400*(train_mu**i)))
        #num_val_samples.append(ceil(100*(val_mu**i)))


    train_index=[]
    val_index=[]
    #print(train_x,train_y)

    for i in range(num_classes):
        #print(np.where(train_y==i)[0].shape)
        train_index.extend(np.where(train_y==i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])
        #index.extend()

    total_index=[]
    total_index.extend(train_index)
    total_index.extend(val_index)
    total_index=list(set(total_index))
    random.shuffle(total_index)
    train_x, train_y=train_x[total_index], train_y[total_index]
    
    num_total_samples=[]
    num_train_samples=[]
    num_val_samples=[]

    train_mu=train_rho**(1./(num_classes-1.))
    val_mu=val_rho**(1./(num_classes-1.))
    for i in range(num_classes):
        num_total_samples.append(round(total_size*(train_mu**i)))
        if not train_size==500:
            num_train_samples.append(num_total_samples[-1]-num_val_samples[-1])
        else:
            num_train_samples.append(round(train_size*(train_mu**i)))
        num_val_samples.append(max(1,round(val_size*(val_mu**i))))

    print('Train samples: ',np.sum(num_train_samples), num_train_samples)
    print('Val samples: ',np.sum(num_val_samples),num_val_samples)

    train_index=[]
    val_index=[]
    #print(train_x,train_y)
    print(num_train_samples,num_val_samples)
    for i in range(num_classes):
        train_index.extend(np.where(train_y==i)[0][:num_train_samples[i]])
        val_index.extend(np.where(train_y==i)[0][-num_val_samples[i]:])

    random.shuffle(train_index)
    random.shuffle(val_index)
    
    train_data,train_targets=train_x[train_index],train_y[train_index]
    val_data,val_targets=train_x[val_index],train_y[val_index]

    if balance_val:
        print("Balanced Validation dataset")
        buckets = [[] for _ in range(num_classes)]
        for idx, label in enumerate(val_targets):
            buckets[label].append(idx)
        sampler = BalancedSampler(buckets, False)
    
    train_dataset = CustomDataset(train_data,train_targets,train_transform)
    val_dataset = CustomDataset(val_data,val_targets,train_transform)
    train_eval_dataset = CustomDataset(train_data,train_targets,test_transform)
    val_eval_dataset = CustomDataset(val_data,val_targets,test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=True, drop_last=False, pin_memory=True)
    if balance_val:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=False, drop_last=False, pin_memory=True, sampler=sampler)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=True, drop_last=False, pin_memory=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, 
                            shuffle=False, drop_last=False, pin_memory=True)

    eval_train_loader = DataLoader(train_eval_dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=False, drop_last=False, pin_memory=True)
    eval_val_loader = DataLoader(val_eval_dataset, batch_size=batch_size, num_workers=num_workers, 
                                shuffle=False, drop_last=False, pin_memory=True)

    return train_loader,val_loader,test_loader,eval_train_loader,eval_val_loader,num_train_samples,num_val_samples

class CustomDataset(Dataset):
    """CustomDataset with support of transforms.
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    def __len__(self):
        return len(self.data)
#load_cifar100()