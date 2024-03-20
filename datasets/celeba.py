from torchvision.datasets import CelebA
from torch.utils.data import Dataset, Subset
from collections import Counter
import torchvision.transforms as T
import numpy as np
from functools import partial
import torch
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
import pandas
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import verify_str_arg

class CelebA_Attributes(Dataset):
    """ 
    subset holding pictures filtered by attributes
    """
    def __init__(self,
                 train,
                 attributes=None,
                 hidden_attributes=None,
                 ratio=None,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 download: bool = False
                 ):
        # Load default CelebA dataset
        self.celeba_attr = CustomCelebA(root=root,
                        split='all',
                        target_type='attr')
        

        # get attribute via index provided in config file (e.g. default_training.yaml)
        attributes = attributes  
        hidden_attributes = hidden_attributes 
        self.split_seed = split_seed

        # choose if attribute should be negated 
        # (e.g. to get people with beard using negation of no_beard attribute)
        attr_negation = False # default is false


        # create subsets for class 1 and class 2
        if (len(attributes) == 0):
            raise Exception('please specify 2 attributes in config file')
        c1_attr = attributes[0] # TODO maybe change to also allow attribute negations
        c2_attr = attributes[1]

        self.class1_idx = self.create_idx(c1_attr, hidden_attributes, ratio)
        self.class2_idx = self.create_idx(c2_attr, hidden_attributes, 0.5)
     
        # make class 1 and class 2 the same size
        if (len(self.class2_idx) > len(self.class1_idx)):
            self.class2_idx = self.class2_idx[:len(self.class1_idx)] 
        else:
            raise Exception('class 2 is smaller than class 1')

        # check for overlapping samples to discard 
        self.discarded_idx = self.discarded_samples()

        # sanity check for hidden attribute ratio
        c1 = 0
        for i in self.class1_idx:
            _, tensor_elements = self.celeba_attr[int(i)]
            c1 += tensor_elements[hidden_attributes[0]]
        print("ratio hidden attr class1: ", c1/len(self.class1_idx))

        c2 = 0
        for i in self.class2_idx:
            _, tensor_elements = self.celeba_attr[int(i)]
            c2 += tensor_elements[20]
        print("ratio hidden attr class2: ", c2/len(self.class2_idx))


        # define dataset 
        indices = np.concatenate([self.class1_idx, self.class2_idx])
        
        # assign all elements of class 1 the target value 1, and those of class 2 the target value 0
        targets_mapping = {
            indices[i]: 1 if i < len(self.class1_idx) else 0 
            for i in range(len(indices))
        }

        # remove samples with ambiguous labels
        if (len(self.discarded_idx) < 0):
            for sample_idx in self.discarded_idx:
                del targets_mapping[str(sample_idx)]
            print('removed ambiguous samples: ', self.discarded_idx)

     
        # shuffle dataset
        np.random.seed(self.split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices)) # take 90% of data for training
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        assert len(set.intersection(set(train_idx), set(test_idx))) == 0 

        # Set transformations
        self.transform = transform


        # Split dataset
        if train:
            self.targets =np.array([targets_mapping[x] for x in train_idx]) 
            self.dataset = Subset(self.celeba_attr, train_idx)
            self.name = 'CelebA_Attributes_train'
            
         
        else:
            self.targets = np.array([targets_mapping[x] for x in test_idx])
            self.dataset = Subset(self.celeba_attr, test_idx)
            self.name = 'CelebA_Attributes_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]

    def create_idx(self, attr, hidden_attr=None, ratio=None):
            if hidden_attr is None:
                # e.g. puts all blond people in a class without checking hidden_attribute
                # NOTE: it is advised to provide the hidden attribute and a ratio of 0.5 to guarantee equal distribution
                attr_mask = self.celeba_attr.attr[:,attr] > 0 #e.g all blond people
                class_idx = torch.where(attr_mask)[0] 
                return class_idx

            else:
                attr_mask = self.celeba_attr.attr[:,attr] > 0
                # aditionally filter for hidden attribute
                hidden_attr_mask = self.celeba_attr.attr[:, hidden_attr[0]] >0  #e.g. male
                if (len(hidden_attr)==1):
                    neg_hidden_attr_mask = ~hidden_attr_mask #invert mask e.g. female = all entries where 'male' is not 1
                if(len(hidden_attr)==2):
                    neg_hidden_attr_mask = self.celeba_attr.attr[:, hidden_attr[1]] >0 
            
                hidden_pos = torch.where(attr_mask & hidden_attr_mask)[0] #e.g. all 'blond' and 'male'
                hidden_neg = torch.where(attr_mask & neg_hidden_attr_mask)[0] #e.g. all 'blond' and not 'male'

                ratio = float(ratio) # percentage of samples holding the hidden attribute

                # balance hidden attribute in samples according to ratio
                # max nr of samples --> ratio of hidden attribute = 1.0
                total_samples = len(hidden_pos)

                hidden_pos_idx = hidden_pos[:int(total_samples*ratio)]
                hidden_neg_idx = hidden_neg[:int(total_samples*(1-ratio))]
                # TODO check what happens if there are not enough neg samples

                class_idx = np.concatenate([hidden_pos_idx, hidden_neg_idx])

                #shuffle to distribute hidden attribute among idx
                np.random.seed(self.split_seed)
                np.random.shuffle(class_idx)
              
                return class_idx
    
    def discarded_samples(self):        
            # find samples with ambiguous samples (labeled true for class 1 AND class 2)
            class1_idx = self.class1_idx
            class2_idx = self.class2_idx
            discarded_idx = set.intersection(set(class1_idx), set(class2_idx))
            if(len(discarded_idx) > 5):
                raise Exception("Too many samples with ambiguous labels")
            else:
                return discarded_idx


        
class CelebA1000(Dataset):
    """ 
    subset holding all pictures of the 1000 most frequent celebreties
    """
    def __init__(self,
                 train,
                 split_seed=42,
                 transform=None,
                 root='data/celeba',
                 download: bool = False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root,
                        split='all',
                        target_type="identity")
        celeba.targets = celeba.identity

        # Select the 1,000 most frequent celebrities from the dataset
        targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(targets).items(),
                   key=lambda item: item[1],
                   reverse=True))
        sorted_targets = list(ordered_dict.keys())[:1000]

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(targets, sorted_targets))[0]
        
        np.random.seed(self.split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        target_mapping = {
            sorted_targets[i]: i
            for i in range(len(sorted_targets))
        }

        print("target_mapping: ")
        for key, value in list(target_mapping.items())[:5]:
            print(f"{key}: {value}")

        self.target_transform = T.Lambda(lambda x: target_mapping[x])

        # Split dataset
        if train:
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(targets)[train_idx]
            print("shuffeled train_targets", train_targets[:5])
            self.targets = [self.target_transform(t) for t in train_targets]
            print("transformed targets", self.targets[:5])
            self.name = 'CelebA1000_train'
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(targets)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebA1000_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root: str,
            split: str = "all",
            target_type: Union[List[str], str] = "identity",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ):
        super(CustomCelebA, self).__init__(root, transform=transform,
                                     target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(self.attr + 1, 2, rounding_mode='floor') # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba", self.filename[index])
        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)



'''

print("INSPECTION CELEBA1000")
inspection_set = CelebA1000(train=True)


#print(inspection_set[0])
#_,idx = inspection_set[0]
#print("idx: " + str(idx))

print("INSPECTION ATTRIBUTES BASE CLASS")
my_test = CustomCelebA(root='data/celeba',
                        split='all',
                        target_type="attr")

#print(my_test.attr_names)
#print(my_test.attr.shape)
print(my_test[949])

_,attributes = my_test[3]
print(attributes.shape)

print("INSPECTION CELAB A ATTRIBUTES CLASS")
testinstance = CelebA_Attributes(train=True, attributes=[9,8], hidden_attributes=[20], ratio = 0.5)
print(len(testinstance))
print(testinstance[680])

print("INSPECTION ATTRIBUTES BASE CLASS")
my_test = CustomCelebA(root='data/celeba',
                        split='all',
                        target_type="attr")

#print(my_test.attr_names)
#print(my_test.attr.shape)
#print(my_test[680])

_,attributes = my_test[680]
print('black hair: ', int(attributes[8]))
print('blond hair: ', int(attributes[9]))
print('male: ', int(attributes[20]))

'''