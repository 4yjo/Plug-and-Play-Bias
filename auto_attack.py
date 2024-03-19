import subprocess
import wandb

# automation script to attack the target models from  wandb project

# Create an API instance
api = wandb.Api()

# Get all runs from the specified project
runs = api.runs(path=f"plugandplay/Beard_subsets")


# Extract run IDs
run_ids = [run.id for run in runs]

print("List of run IDs:")
print(run_ids)

#note: run id for evaluation  model is specified in attack config 'CelebA_Attr.yaml'

for run in run_ids:
    command = f"python ./attack.py -c=configs/attacking/CelebA_Attr.yaml --run_id={run} --run_name=\"Men_{run.index()/10}\""
    subprocess.run(command, shell=True)
