import subprocess
import wandb

# automation script to attack the target models from  wandb project

###################################################################
### put info here:                                              ###
###################################################################
#project = 


# Create an API instance
api = wandb.Api()

# Get all run-ids from the specified project
runs = api.runs(path=f"plugandplay/Targets_Haircolor_Gender")
run_ids = [run.id for run in runs]

print("List of run IDs: ", run_ids)

# note: run id for evaluation  model is specified in attack config 'CelebA_Attr.yaml'
# make sure the evaluation  model is trained on suitable data/same data as training #TODO double check what data should be used for evaluation 

for idx, run in enumerate(run_ids):
    command = f"python ./attack.py -c=configs/attacking/CelebA_Attr.yaml --run_id={run} --run_name=\"Male_{idx/10}\""
    subprocess.run(command, shell=True)
