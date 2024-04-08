import yaml
from rtpt.rtpt import RTPT
import wandb

class AttrIdentConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config


    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=1)
        return rtpt
    

    @property
    def wandb_attack_run(self):
        return self._config['wandb_attack_run']
    
    @property
    def image_location(self):
        return self._config['image_location']
    
    @property
    def attribute(self):
        return self._config['attribute']
    
    
    @property
    def prompts(self):
        return self._config['prompts']


    @property
    def stylegan_model(self):
        return self._config['stylegan_model']

    @property
    def seed(self):
        return self._config['seed']
    
    @property
    def rtpt(self):
        return self._config['rtpt']

    
    @property
    def wandb(self):
        return self._config['wandb']
    
    @property
    def wandb_project(self):
        return self._config['wandb']['wandb_init_args']['project']
    
    @property
    def wandb_name(self):
        return self._config['wandb']['wandb_init_args']['name']
