import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class DatasetProcessing(Dataset):
    def __init__(self, data_path, img_path, img_filename, attribute_filename, transform=None):
        self.img_path = os.path.join(data_path, img_path)
        self.transform = transform
        
        # reading img file from file
        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        #print(self.img_filename)
        fp.close()
        
        # reading labels from file
        attribute_filepath = os.path.join(data_path, attribute_filename)
        att_labels = np.loadtxt(attribute_filepath) #int64
        self.att_label = att_labels

    def __getitem__(self, index):
        imagepath = os.path.join(self.img_path, self.img_filename[index])
        img = Image.open(imagepath)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        

        attribute_label = self.att_label[index]

        attribute_label = torch.LongTensor(attribute_label)
        
        return img, attribute_label
    def __len__(self):
        return len(self.img_filename)
