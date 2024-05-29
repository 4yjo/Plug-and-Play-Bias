import subprocess

# automation script to train target model using different subsets of celebA dataset 
# ratio defines the percentage of images in class1 that hold the hidden attribute 
#                       (0.0 -> no images with hidden_attribute, 1.0 all images with hidden_attribute)

# make sure that indices given attributes and hidden_attributes are what you want to filter the data
# get the indices for attributes eg. 20: 'Male' from datasets/inspect_dataset.py

ratios = [i/10 for i in range(6,11)]

# Iterate over the ratios and run names
for ratio in ratios:
    command = f"python train_model.py  -c=configs/training/default_training.yaml --attributes 20  --hidden_attributes 21 --ratio {ratio} --run_name=\"blond_{ratio}\"" 
    subprocess.run(command, shell=True)