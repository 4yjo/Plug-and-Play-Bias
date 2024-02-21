import torch
from torchvision.datasets import CelebA

# Load the CelebA dataset
celeb_dataset = CelebA(root='path_to_celeba_dataset', download=True)

# Image ids you want to use for training
image_ids = [3132, 396, 62570, 2090, 60365, 57, 3837, 67058, 59184, 2718]

# Create a subset of CelebA dataset using the specified image ids
subset_dataset = torch.utils.data.Subset(celeb_dataset, image_ids)