import pandas as pd
import wandb

api = wandb.Api()
entity, project = "plugandplay", "AttacksHaircolorGender"
runs = api.runs(entity + "/" + project)

c1_list, c2_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary = run.summary._json_dict
    c1_list.append(summary.get("c1_male"))

    c2_list.append(summary.get("c2_male"))

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    #config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    { "name": name_list, "c1_male": c1_list, "c2_male": c2_list}
)

runs_df.to_csv("AttacksHaircolorGender.csv")