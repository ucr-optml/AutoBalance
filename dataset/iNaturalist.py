from torch._C import dtype
import torch.utils.data as data
from PIL import Image
import os
import json
from torchvision import transforms
import random
import numpy as np
import math
import torch
def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic


class INAT(data.Dataset):
    def __init__(self, root, ann_file, is_train=True, split=0):
        train_filename="train2018.json"
        test_filename="val2018.json"

        # load train to get permutation
        train_path=f"{root}/{train_filename}"
        with open(train_path) as data_file:
            train_data=json.load(data_file)
        imgs = [aa['file_name'] for aa in train_data['images']]
        if 'annotations' in train_data.keys():
            self.classes = [aa['category_id'] for aa in train_data['annotations']]
        else:
            self.classes = [0]*len(imgs)
        self.num_classes=len(set(self.classes))
        num_train_samples = [0 for _ in range(self.num_classes)]
        for label in self.classes:
            num_train_samples[label]+=1
        permutation=np.argsort(num_train_samples)[::-1]

        #print(train_size)

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        self.num_classes=len(set(self.classes))
        
        selected_index=[]
        if split!=0:
            selected_index=[]
            if split==1:
                labels=np.array(self.classes)
                for i in range(self.num_classes):
                    #print((np.where(labels==i)[0]),len((np.where(labels==i)[0])),(np.where(labels==i)[0])*0.8)
                    selected_len=len((np.where(labels==i)[0]))-math.ceil(len((np.where(labels==i)[0]))*0.2)
                    selected_index.extend(np.where(labels==i)[0][:selected_len])
            elif split==2:
                labels=np.array(self.classes)
                for i in range(self.num_classes):
                    #print((np.where(labels==i)[0]),len((np.where(labels==i)[0])),(np.where(labels==i)[0])*0.8)
                    selected_len=math.ceil(len((np.where(labels==i)[0]))*0.2)
                    selected_index.extend(np.where(labels==i)[0][-selected_len:])
            np.random.shuffle(selected_index)
            self.imgs=np.array(self.imgs)[selected_index]
            self.classes=np.array(self.classes)[selected_index]
        # print out some stats
        print ('\t' + str(len(self.imgs)) + ' images')
        print ('\t' + str(len(set(self.classes))) + ' classes')
        #for i in self.classes:
        #    print(num_train_samples[i],np.where(permutation==i)[0][0])
        self.classes=[np.where(permutation==i)[0][0] for i in self.classes]
        

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # augmentation params
        #self.im_size = [299, 299]  # can change this to train on higher res
        self.im_size = [224, 224]
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.is_train:
            img = self.scale_aug(img)
            img = self.flip_aug(img)
            img = self.color_aug(img)
        else:
            img = self.center_crop(img)

        img = self.tensor_aug(img)
        img = self.norm_aug(img)
        #print(img.shape,species_id)
        return img,torch.tensor([species_id],dtype=torch.long).squeeze()
        return img, im_id, species_id, tax_ids

    def __len__(self):
        return len(self.imgs)
    
    def get_class_size(self):
        #from collections import Counter
        #values=list(Counter(self.classes).values())
        num_train_samples = [0 for _ in range(self.num_classes)]
        for label in self.classes:
            num_train_samples[label]+=1
        #values=np.flip(np.sort(values))
        print(len(num_train_samples),num_train_samples)
        return num_train_samples


# train_data=INAT('/home/eeres/mili/data/inat_2018','/home/eeres/mili/data/inat_2018/train2018.json',is_train=True)

# from collections import Counter
# values=list(Counter(train_data.classes).values())
# print(values)
# print(max(values),min(values))
# values=np.flip(np.sort(values))
# print(values)

# data=values

# import numpy as np
# from matplotlib import cm
# import matplotlib.pyplot as plt

# classes=len(values)

# print(data)

# color1=[]
# for i in range(classes):
#     if i<1000:
#         color1.append('tab:blue')
#     else:
#         color1.append('tab:blue')
# plt.bar(range(classes),data,color=color1,width=0.98)

# plt.xticks(fontsize=14)
# plt.yticks(fontsize=16)
# plt.grid('--')

# colors = {'Majority':'tab:blue', 'Minority':'tab:orange'}         
# labels = list(colors.keys())


# handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
# #handles = [artist(label) for label in labels]
# #plt.legend(handles, labels,fontsize=18)
# #plt.yscale('log')
# plt.savefig('results/figs/figure_intro_distribution_log.pdf')
# plt.show()


# train_data=INAT('/home/eeres/mili/data/inat_2018','/home/eeres/mili/data/inat_2018/train2018.json',is_train=True)

# from collections import Counter
# values=list(Counter(train_data.classes).values())
# for k in zip(train_data.classes,train_data.ids):
#     print(k)