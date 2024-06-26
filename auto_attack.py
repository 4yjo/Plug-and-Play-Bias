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
runs = api.runs(path=f"plugandplay/TargetsHaircolorGender-Balanced") 

run_ids = [run.id for run in runs]

# correctec order run id TargetsMouthopenGlasses
#run_ids = ['msamlj5b', 'rst7y49z', 'gzvuu2z7', 'y7mkq56v', 'zd1u6ttm', 'whqh6iak','ogm7z9vu', '9r5vfm1c', 'wf17w28h', 'ca1yjanc', '752za6cn']

print("List of run IDs: ", run_ids)

# note: run id for evaluation  model is specified in attack config 'CelebA_Attr.yaml'
# make sure the evaluation  model is trained on suitable data/same data as training #TODO double check what data should be used for evaluation 

for idx, run in enumerate(run_ids):
    command = f"python ./attack.py -c=configs/attacking/CelebA_Attr.yaml --wandb_target_run=\"TargetsMouthopenGlasses/{run}\" --run_name=\"Glasses_{1-(int(idx/10))}\""
    subprocess.run(command, shell=True)

