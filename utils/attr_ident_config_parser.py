import yaml
from rtpt.rtpt import RTPT

class AttrIdentConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    #TODO create function to generate images from wandb run
    
    #TODO prompts

    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'])
        return rtpt
    
    @property
    def wandb_attack_run(self):
        return self._config['wandb_attack_run']

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
