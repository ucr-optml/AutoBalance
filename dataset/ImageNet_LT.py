import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from dataset.baseDataLoader import BaseDataLoader
from PIL import Image

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
            return sum([len(bucket) for bucket in self.buckets]) # Acrually we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num # Ensures every instance has the chance to be visited in an epoch

class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label

class ImageNetLTDataLoader(DataLoader):
    """
    ImageNetLT Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=8, training=True, balanced=False, retain_epoch_size=True):
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.dataset=LT_Dataset(data_dir, data_dir + '/ImageNet_LT_train.txt', train_trsfm)
        #self.val_dataset = LT_Dataset(data_dir, data_dir + '/ImageNet_LT_val.txt', test_trsfm)
        self.num_classes=len(np.unique(self.dataset.targets))
        assert self.num_classes == 1000
        num_train_samples =[0] * self.num_classes
        for label in self.dataset.targets:
            num_train_samples[label]+=1
        permutation=np.argsort(num_train_samples)[::-1]
        # self.n_samples = len(self.dataset)
        # train_size,val_size=self.get_train_val_size()
        # permutation=np.argsort(train_size)[::-1]

        
        if training:
            dataset = LT_Dataset(data_dir, data_dir + '/ImageNet_LT_train.txt', train_trsfm)
            val_dataset = LT_Dataset(data_dir, data_dir + '/ImageNet_LT_val.txt', test_trsfm)
            
        else: # test
            dataset = LT_Dataset(data_dir, data_dir + '/ImageNet_LT_test.txt', test_trsfm)
            val_dataset = None

        self.dataset = dataset
        self.val_dataset = val_dataset
        self.dataset.targets=[np.where(permutation==i)[0][0] for i in self.dataset.targets]
        self.dataset.labels=[np.where(permutation==i)[0][0] for i in self.dataset.labels]
        
        # if self.val_dataset:
        #     self.val_dataset.targets=[permutation[i] for i in self.val_dataset.targets]
        #     self.val_dataset.labels=[permutation[i] for i in self.val_dataset.labels]
        if self.val_dataset:
            self.val_dataset.targets=[np.where(permutation==i)[0][0] for i in self.val_dataset.targets]
            self.val_dataset.labels=[np.where(permutation==i)[0][0] for i in self.val_dataset.labels]
            
        self.n_samples = len(self.dataset)

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 1000
        self.num_classes=num_classes
        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list
        print("cls_num",cls_num_list)

        if balanced:
            if training:
                buckets = [[] for _ in range(num_classes)]
                for idx, label in enumerate(dataset.targets):
                    buckets[label].append(idx)
                sampler = BalancedSampler(buckets, retain_epoch_size)
                shuffle = False
            else:
                print("Test set will not be evaluated with balanced sampler, nothing is done to make it balanced")
        else:
            sampler = None
        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs, sampler=sampler) # Note that sampler does not apply to validation set

    def split_validation(self):
        # If you do not want to validate:
        # return None
        # If you want to validate:
        return DataLoader(dataset=self.val_dataset, **self.init_kwargs)
    def get_train_val_size(self):
        num_train_samples = [0 for _ in range(self.num_classes)]
        for _, label in enumerate(self.dataset.targets):
            num_train_samples[label]+=1
        num_val_samples = [0 for _ in range(self.num_classes)]
        for _, label in enumerate(self.val_dataset.targets):
            num_val_samples[label]+=1
        print(num_train_samples,num_val_samples)
        return num_train_samples,num_val_samples