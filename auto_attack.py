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
runs = api.runs(path=f"plugandplay/TargetsHaircolorGender")
run_ids = [run.id for run in runs]

print("List of run IDs: ", run_ids)


for idx, run in enumerate(run_ids):
    command = f"python ./attack.py -c=configs/attacking/CelebA_Attr.yaml --wandb_target_run=\"TargetsHaircolorGender/{run}\" --run_name=\"Male_{1-(idx/10)}\""
    subprocess.run(command, shell=True)
