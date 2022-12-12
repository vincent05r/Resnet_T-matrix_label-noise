import numpy as np
import torch.utils.data as Data
from PIL import Image

import tools





class cifar_dataset(Data.Dataset):
    def __init__(self, path="./dataset/CIFAR.npz", train=True, transform=None, target_transform=None, split_per=0.8, random_seed=1, num_class=3):
            
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 

        self.dataset = np.load(path)   

        # #Training/Validation dataset
        # self.Xtr_val = self.dataset["Xtr"] #input
        # self.Str_val = self.dataset["Str"] #labels

        # #Testing dataset
        # self.Xts = self.dataset["Xts"] #input
        # self.Yts = self.dataset["Yts"] #labels

        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(self.dataset["Xtr"], self.dataset["Str"], split_per, random_seed)

        training_amount = self.train_data.shape[0]
        validation_amount = self.val_data.shape[0]

        if self.train: 
            self.train_data = self.train_data.reshape((training_amount,3,32,32)) #subject to testing
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) #what is this?
        
        else:
            self.val_data = self.val_data.reshape((validation_amount,3,32,32)) #subject to testing
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
           
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
            
        else:
            img, label = self.val_data[index], self.val_labels[index]
            
        img = Image.fromarray(img)
           
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    def __len__(self):
            
        if self.train:
            return len(self.train_data)
        
        else:
            return len(self.val_data)
        



class cifar_test_dataset(Data.Dataset):
    def __init__(self, path="./dataset/CIFAR.npz", transform=None, target_transform=None):
            
        self.transform = transform
        self.target_transform = target_transform

        self.dataset = np.load(path)   

        #Testing input
        self.test_data = self.dataset["Xts"] #input
        testing_amount = self.test_data.shape[0]
        self.test_data = self.test_data.reshape((testing_amount, 3, 32, 32))   #self.Xts.reshape((testing_amount, 3, 32, 32)) is outputting (32, 3, 32)
        self.test_data = self.test_data.transpose((0, 2, 3, 1)) 

        #testing labels
        self.test_labels = self.dataset["Yts"] #labels

        #testing data loader debugger
        print("testing amount is " + str(testing_amount))
        print("the first testing data is : " + str(self.test_data[0].shape))
        print("The first testing label is : " + str(self.test_labels[0].shape) + "  with actual label : {}".format(self.test_labels[0]))



    def __getitem__(self, index):
        
        img, label = self.test_data[index], self.test_labels[index]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
     
        return img, label
    
    def __len__(self):
        return len(self.test_data)





class FMN_dataset(Data.Dataset):
    def __init__(self, path="./dataset/FashionMINIST0.5.npz", train=True, transform=None, target_transform=None, split_per=0.8, random_seed=1, num_class=3):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.dataset = np.load(path)

        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(self.dataset["Xtr"], self.dataset["Str"], split_per, random_seed)

        training_amount = self.train_data.shape[0]
        validation_amount = self.val_data.shape[0]

        if self.train:
            self.train_data = self.train_data.reshape((training_amount,1,28,28)) #subject to testing
            self.train_data = self.train_data.transpose((0, 2, 3, 1)) #what is this?

        else:
            self.val_data = self.val_data.reshape((validation_amount,1,28,28)) #subject to testing
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]

        else:
            img, label = self.val_data[index], self.val_labels[index]

        img = np.squeeze(img, axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class FMN_test_dataset(Data.Dataset):
    def __init__(self, path="./dataset/FashionMINIST0.5.npz", transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        self.dataset = np.load(path)

        #Testing input
        self.test_data = self.dataset["Xts"] #input
        testing_amount = self.test_data.shape[0]
        self.test_data = self.test_data.reshape((testing_amount, 1, 28, 28))   #self.Xts.reshape((testing_amount, 3, 32, 32)) is outputting (32, 3, 32)
        self.test_data = self.test_data.transpose((0, 2, 3, 1))

        #testing labels
        self.test_labels = self.dataset["Yts"] #labels

        #testing data loader debugger
        print("testing amount is " + str(testing_amount))
        print("the first testing data is : " + str(self.test_data[0].shape))
        print("The first testing label is : " + str(self.test_labels[0].shape))



    def __getitem__(self, index):

        img, label = self.test_data[index], self.test_labels[index]

        img = np.squeeze(img, axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.test_data)