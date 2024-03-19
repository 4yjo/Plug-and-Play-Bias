import subprocess

# automation script to train target model using different subsets of celeb a 
# ratio defines the percentage of images in training data that hold the given 
# attribute (0.0 -> no images with attribute, 1.0 all images with attribute)

ratios = [i/10 for i in range(0,11)]

# Iterate over the ratios and run names
for ratio in ratios:
    command = f"python ./train_model.py -c=configs/training/default_training.yaml --ratio={ratio} --run_name=\"Beard_{ratio}\""
    subprocess.run(command, shell=True)