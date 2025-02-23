import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from imblearn.over_sampling import SMOTE  
import random
import itertools
import torchvision
from torchvision.io import read_image
import pandas as pd


class Reduce(Dataset):
    # reduces number of classes
    # takes in original dataset, target # of classes, which classes to be used 
    def __init__(self, original_dataset, num_classes, nums=(0,1), transform=None):  
        
        assert len(nums) == num_classes
       
        indices = np.isin(original_dataset.targets, nums) 
        self.images = torch.from_numpy(original_dataset.data[indices==1]).float()
        self.labels = torch.from_numpy(np.array(original_dataset.targets)[indices==1])

        self.nums = nums 
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        if self.labels[index]==self.nums[0]:
            label = 0
        elif self.labels[index]==self.nums[1]:
            label = 1
        else: # nums[2] if it exists
            label = 2
        if self.transform:
            image = self.transform(image)
        return (image, label)
    
    

class Ratio(Dataset):
    # assumes all classes are balanced 
    # takes in reduced or unreduced dataset 
    def __init__(self, original_dataset, num_classes, target_ratios, nums=(3,2,1), transform=None):
        assert len(target_ratios) == num_classes # target ratios is a list of the ratios between classes. Makes sure length of tr is equivalent to num classes
       
        self.nums=nums # nums = class labels
        
        class_indices = np.isin(original_dataset.targets, nums) # class indices are the indices of the original dataset samples that have the labels specified in nums
        
        targets = np.asarray(original_dataset.targets)[class_indices] # targets = numpy array of all the labels that have the right values (specified in nums)
        images = original_dataset.data[class_indices] # all the images with the right labels
        
        _, class_counts = np.unique(np.sort(targets), return_counts=True) # class counts = list with the number of samples in the class (for each class)
        
        max_index = target_ratios.index(max(target_ratios)) # the index of the smallest ratio in target ratios
        
        updated_ratios = tuple(ratio/target_ratios[max_index] for ratio in target_ratios) # scales the target ratios such that max ratio = 1
        
        ratio_class_counts = tuple(int(ratio*class_count) for ratio, class_count in zip(updated_ratios, class_counts)) # scale target ratios so max ratio = class count for max index
        
               
        reduced_images = []
        reduced_labels = []
                
        for i, num in enumerate(nums): # for each label in nums
            class_images = images[(targets == num)] # taking all the images whose labels are equal to the current label
            class_images = torch.from_numpy(class_images) # converts filtered class images from np array to tensor
            indices = np.random.choice(class_images.shape[0], ratio_class_counts[i], replace=False) # gets the indices of a random subset of images
            reduced_images.append(class_images[indices]) # append said images to the reduced images
            reduced_labels.append(torch.from_numpy(np.full(ratio_class_counts[i], i))) # append their labels to reduced labels
        
        self.images = torch.cat(reduced_images) #end up with a reduced set of images, with randomly selected subset of images for each label
        self.labels = torch.cat(reduced_labels).int()
        self.transform=transform
     
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float() # takes image from reduced images at given image
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return (image, label) 


class Ratio_CheXpert_version(Dataset):
    # assumes all classes are balanced 
    # takes in reduced or unreduced dataset 
    def __init__(self, labels_dataframe, num_classes, target_ratios, nums=(3,2,1), transform=None):
        # assert len(target_ratios) == num_classes # target ratios is a list of the ratios between classes. Makes sure length of tr is equivalent to num classes
       
        self.nums=nums # nums = class labels
        self.target_ratios = target_ratios
        self.transform = transform
               
        reduced_images_paths = []
        reduced_labels = []
        class_count = 0


                
        for i, num in enumerate(nums): # for each label in nums
            class_images = labels_dataframe.loc[labels_dataframe["Condition"]==num]
            class_images_paths = class_images["Path"].to_numpy()

            class_count = self.target_ratios[i]
            assert len(class_images) >= class_count
            indices = [random.randrange(0, class_count) for i in range(class_count)]

            reduced_imgs = class_images_paths[indices].tolist()
            reduced_images_paths.append(reduced_imgs) # append said images to the reduced images
            reduced_lbls = np.full(class_count, i).tolist()
            reduced_labels.append(reduced_lbls)


        self.images = [item for sublist in reduced_images_paths for item in sublist] #end up with a reduced set of images, with randomly selected subset of images for each label
        self.labels = [item for sublist in reduced_labels for item in sublist]
        # my_df = pd.DataFrame({'Image paths': self.images, 'Labels': self.labels})
        # print(my_df.head(n=50))


     
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index):
        # image = torchvision.io.read_image(self.images[index])
        image = np.zeros((1, 2320, 2828))
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return (image, label) 
    

NO_SMOTE_LABEL = 0
SMOTE_LABEL = 1


class Smote(Dataset): 
    # takes in imbalanced dataset 
    # apples SMOTE to the minority class(es)
    def __init__(self, ratio_dataset, target_shape, transform=None):
        
        shape = ratio_dataset.images.shape
                
        smote = SMOTE()
        
        self.images, self.labels = smote.fit_resample(ratio_dataset.images.reshape(shape[0], -1), ratio_dataset.labels)
        
        self.smote_labels = np.zeros(target_shape)
        
        self.smote_labels[shape[0]:] = SMOTE_LABEL 
        
        self.images = torch.from_numpy(self.images.reshape(-1, shape[1], shape[2], shape[3]))
        
        self.labels = torch.from_numpy(self.labels)
        
        self.transform=transform
       
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index): 
        image = self.images[index].float()
        label = self.labels[index]
        smote_label = self.smote_labels[index]
        if self.transform:
            image = self.transform(image)
        return (image, label, smote_label)

    
    
class ForTripletLoss(Dataset): 
    # modifies dataset to be used for triplet loss 
    # NOTE: only works with 2 class or 3 class at the moment 
    def __init__(self, dataset, smote=False, transform=None, nums=(0,1)):
        self.images = dataset.images.float()
        self.labels = dataset.labels
        self.smote = smote # whether dataset passed in uses SMOTE 
       
       
        num_classes = len(nums) 
        if smote: 
            self.smote_labels = dataset.smote_labels
            class0_smote_mask = np.full_like(self.labels, fill_value=False, dtype=bool)
            class1_smote_mask = np.full_like(self.labels, fill_value=False, dtype=bool)
            
            class0_smote_mask[self.smote_labels==NO_SMOTE_LABEL] = True
            class1_smote_mask[self.smote_labels==NO_SMOTE_LABEL] = True
            
            class0_smote_mask[self.labels!=nums[0]] = False
            class1_smote_mask[self.labels!=nums[1]] = False
            
            self.class0_images = self.images[class0_smote_mask]
            self.class1_images = self.images[class1_smote_mask]
            
            if num_classes == 3:
                class2_smote_mask = np.full_like(self.labels, fill_value=False, dtype=bool)
                class2_smote_mask[self.smote_labels==NO_SMOTE_LABEL] = True
                class2_smote_mask[self.labels!=nums[2]] = False
                self.class2_images = self.images[class2_smote_mask]
            
        else:
            self.class0_images = self.images[self.labels==nums[0]]
            self.class1_images = self.images[self.labels==nums[1]]
            if num_classes == 3:
                self.class2_images = self.images[self.labels==nums[2]] 
            
        self.transform=transform 
        self.num_classes = num_classes
        self.nums = nums
        
         
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        anchor_image = self.images[index]
        anchor_label = self.labels[index]
       
        if self.num_classes==2:
            if anchor_label == self.nums[0]:
                pos_image = random.choice(self.class0_images)
                neg_image = random.choice(self.class1_images)
            else:
                pos_image = random.choice(self.class1_images)
                neg_image = random.choice(self.class0_images)
        elif self.num_classes == 3:
            if anchor_label == self.nums[0]:
                pos_image = random.choice(self.class0_images)
                neg_image = random.choice(torch.cat((self.class1_images, self.class2_images)))
            elif anchor_label == self.nums[1]:
                pos_image = random.choice(self.class1_images)
                neg_image = random.choice(torch.cat((self.class0_images, self.class2_images)))
            else:
                pos_image = random.choice(self.class2_images)
                neg_image = random.choice(torch.cat((self.class0_images, self.class1_images)))
            
        
        if self.transform:
            anchor_image = self.transform(anchor_image)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)
        if self.smote: 
            anchor_smote_label = self.smote_labels[index]
            return (anchor_image, pos_image, neg_image, anchor_label, anchor_smote_label)
        return (anchor_image, pos_image, neg_image, anchor_label)