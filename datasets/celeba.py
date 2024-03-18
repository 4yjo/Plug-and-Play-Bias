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
        my_celeba = CustomCelebA(root=root,
                        split='all',
                        target_type='attr')
        
       # my_celeba.targets=my_celeba.attr #WHAT DOES THAT DO TODO

        # provide index/indices for attributes 
        attributes = attributes  # TODO assumes [c1 attr, c2 attr]
        hidden_attributes = hidden_attributes # TODO assumes [male] or [blond, brown]
        print("Attributes: ", attributes)
        print("Hidden Attributes: ", hidden_attributes)

        # choose if attribute should be negated 
        # (e.g. to get people with beard using negation of no_beard attribute)
        attr_negation = False # default is false

       
        def create_idx(attr, hidden_attr=None, ratio=None):
            if hidden_attr is None:
                attr_mask = my_celeba.attr[:,attr] > 0 #e.g all blond people
                class_idx = torch.where(attr_mask)[0] 
                return class_idx

            else:
                attr_mask = my_celeba.attr[:,attr] > 0
                # aditionally filter for hidden attribute
                hidden_attr_mask = my_celeba.attr[:, hidden_attr[0]] >0  #e.g. male
                if (len(hidden_attr)==1):
                    neg_hidden_attr_mask = ~hidden_attr_mask #invert mask e.g. female = all entries where 'male' is not 1
                if(len(hidden_attr)==2):
                    neg_hidden_attr_mask = my_celeba.attr[:, hidden_attr[1]] >0 
            
                hidden_pos = torch.where(attr_mask & hidden_attr_mask)[0] #e.g. all 'blond' and 'male'
                hidden_neg = torch.where(attr_mask & neg_hidden_attr_mask)[0] #e.g. all 'blond' and not 'male'

                ratio = float(ratio) # percentage of samples holding the hidden attribute

                # balance samples for hidden attribute according to ratio
                # max nr of samples --> ratio of hidden attribute = 1.0
                total_samples = len(hidden_pos)

                hidden_pos_idx = hidden_pos[:int(total_samples*ratio)]
                hidden_neg_idx = hidden_neg[:int(total_samples*(1-ratio))]
                # TODO check what happens if there are not enough neg samples

                class_idx = np.concatenate([hidden_pos_idx, hidden_neg_idx])
                return class_idx

         # create class 1 and class two data samples

        c1_attr = attributes[0] # TODO maybe change to also allow attribute negations
        c2_attr = attributes[1]

        class1_idx = create_idx(c1_attr, hidden_attributes, ratio)
        class2_idx = create_idx(c2_attr)
            
        '''

        # get indices of image that are true for (all) attribute(s) from celeba attr tensor
        if (len(attributes) == 0):
            raise ValueError('no attributes given to filter subset')
        if (len(attributes) == 1):
            attr_mask = my_celeba.attr[:,attributes[0]] > 0  # takes indices of given attribute 
            hidden_attr_mask = my_celeba.attr[:,hidden_attributes[0]] > 0 
        
    
        if (len(attributes) > 1):
            attr_x_indices = [] # array to store bool values for each attribute
            for i in range(len(attributes)):
                attr_x_indices.append(my_celeba.attr[:,attributes[i]] > 0)

            attr_mask = torch.zeros(202599, dtype= torch.bool) #initialize tensor of size celeba.attr
            for i in range(len(attr_x_indices)):
                attr_mask = attr_mask | attr_x_indices[i] #bitwise or to select indices that hold at least one attribute
        
        
        # get image ids (= index of images according to true/false value in attr mask)
        if not attr_negation:
            class1_idx = np.concatenate([class1_idx, class2_idx])
            class2_idx = torch.where(~attr_mask)[0] 
        else:
            #TODO this wont work here
            class1_idx = torch.where(~attr_mask)[0]
            class2_idx = torch.where(attr_mask)[0] 
        '''
       
        # balance samples 50:50 to make class 1 and class 2 the same size
        if (len(class2_idx) > len(class1_idx)):
            class2_idx = class2_idx[:len(class1_idx)] 
        else:
            raise Exception('class 2 is smaller than class 1')

        
        # Assert that there are no overlapping datasets
        amb_samples = set.intersection(set(class1_idx), set(class2_idx))
        if(len(amb_samples) > 5):
            raise Exception("Too many samples with ambiguous labels")
      
        #assert len(set.intersection(set(class1_idx), set(class2_idx))) == 0

        # check balance of hidden attribute for class 
        counter = 0
        for i in class2_idx:
            _, tensor_elements = my_celeba[int(i)]
            counter += tensor_elements[20]
        print("ratio hidden attr class2: ", counter/len(class2_idx))
       
        
        indices = np.concatenate([class1_idx, class2_idx])
        
        # map targets and indices
        targets_mapping = {
            indices[i]: 1 if i < len(class1_idx) else 0 
            for i in range(len(indices))
        }

        # remove samples with ambiguous labels
        if (len(amb_samples) < 0):
            for sample_idx in amb_samples:
                del targets_mapping[str(sample_idx)]
            print('removed ambiguous samples: ', amb_samples)

     
        # shuffle dataset
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices)) # take 90% of data for training
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]


        # Assert that there are no overlapping datasets
        print("Assertion: ")
        print(set.intersection(set(train_idx), set(test_idx)))
        #assert len(set.intersection(set(train_idx), set(test_idx))) == 0
       
       

        # Set transformations
        self.transform = transform


        # Split dataset
        if train:
            #print("INDICES in CELEB A given to parser")
            #print(train_idx)
            self.targets =np.array([targets_mapping[x] for x in train_idx]) 
            #print(self.targets)
            #print('---')
            self.dataset = Subset(my_celeba, train_idx)
            self.name = 'CelebA_Attributes_train'
            #print("INDICES created from shuffled Subset")
            
         
        else:
            self.targets = np.array([targets_mapping[x] for x in test_idx])
            self.dataset = Subset(my_celeba, test_idx)
            self.name = 'CelebA_Attributes_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]



        
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
        print("indices ", indices)
        print(len(indices))
        
        np.random.seed(split_seed)
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