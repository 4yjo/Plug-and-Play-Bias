import pandas as pd
import wandb

api = wandb.Api()
entity, project = "plugandplay", "AttacksHaircolorGender-NotBalanced"
runs = api.runs(entity + "/" + project)

c1_list, c2_list, c1_list_10, c1_list_25, c2_list_10, c2_list_25, name_list = [], [], [], [], [],[], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary = run.summary._json_dict
    c1_list.append(summary.get("a male face in Class 1"))
    c1_list_10.append(summary.get("c1-10"))
    c1_list_25.append(summary.get("c1-25"))

    c2_list.append(summary.get("a male face in Class 2"))
    c2_list_10.append(summary.get("c2-10"))
    c2_list_25.append(summary.get("c2-25"))


 

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    #config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    { "name": name_list, "c1_male": c1_list, "c1-10": c1_list_10, "c1-25": c1_list_25,"c2_male": c2_list,"c2-10": c2_list_10, "c2-25": c2_list_25,}
)

runs_df.to_csv("AttacksHaircolorGender-NotBalanced.csv")