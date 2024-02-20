import subprocess
import wandb

# automation script to train target model using different subsets of celeb a 
# ratio defines the percentage of images in training data that hold the given 
# attribute (0.0 -> no images with attribute, 1.0 all images with attribute)

ratios = [i/10 for i in range(0,11)]

# Iterate over the ratios and run names
for ratio in ratios:
    command = f"python ./train_model.py -c=configs/training/default_training.yaml --ratio={ratio} --run_name=\"Beard_{ratio}\""
    subprocess.run(command, shell=True)



# Initialize your project (if not already done)
#wandb.init(project="plugandplay")

# Create an API instance
api = wandb.Api()

# Get all runs from the specified project
runs = api.runs(path=f"plugandplay/Beard_subsets")

# Extract run IDs
run_ids = [run.id for run in runs]

print("List of run IDs:")
print(run_ids)

#caution: run id for evaluation  model is specified in attack config 'CelebA_Attr.yaml'

for run in run_ids:
    command = f"python ./train_model.py -c=configs/training/default_training.yaml --ratio={ratio} --run_name=\"Beard_{ratio}\""
    subprocess.run(command, shell=True)
