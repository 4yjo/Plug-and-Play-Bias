import torch
import numpy as np
import matplotlib.pyplot as plt

from celeba import CustomCelebA

###########################################
## different helpers to inspect celeb a ##
##########################################

print("INSPECTION ATTRIBUTES BASE CLASS")
celeba = CustomCelebA(root='data/celeba',
                        split='all',
                        target_type="attr")

attr = celeba.attr
#print(attr.shape)

# get all 40 features from celeb a attribute tensor
attr_names = celeba.attr_names
#print(attr_names)

# print index of selected attribute
index = attr_names.index('Eyeglasses')
print(index)

# store number of samples for attribute i
sample_dist = [] 

# iterate through attribute tensor and count occurance of features
for i in range(attr.shape[1]):
    sample_dist.append(torch.sum(attr[:,i] >0).item())

# print amount of samples for an attribute
print(sample_dist[index])

def plot_distribution():
    # plot sample distribution

    plt.bar(attr_names, sample_dist)
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.xlabel('Attribute')
    plt.ylabel('Number of Samples')
    plt.title('Sample Distribution of CelebA by Attributes')
    plt.tight_layout()
    plt.savefig('sample_distribution_plot.png')
    plt.show()

