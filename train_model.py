import argparse
import os
import time
from copy import copy

import torch
import torchvision

from metrics.accuracy import Accuracy
from utils.training_config_parser import TrainingConfigParser


def main():
    # Define and parse arguments
    parser = argparse.ArgumentParser(
        description='Training a target classifier')
    parser.add_argument('--ratio', default=0.5) #add arguments that can be specified in CLI
    parser.add_argument('--attributes',type=int, nargs='*', default=None) #nargs gathers multiple attr into list
    parser.add_argument('--hidden_attributes',type=int, nargs='*', default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()


    # Include optional arguments from Command Line Prompt
    ratio = args.ratio
    attributes = args.attributes
    hidden_attributes = args.hidden_attributes
    run_name = args.run_name
    print("Given ratio of hidden attribute in class 1: ", ratio)
    print("Given attributes and hidden_attributes: ", attributes, hidden_attributes)

    # Load json config file
    config = TrainingConfigParser(args.config.strip(), attributes=attributes,
                                   hidden_attributes=hidden_attributes, ratio=ratio)

   
    # Set seeds and make deterministic
    seed = config.seed
    torch.manual_seed(seed)

    # Create the target model architecture
    target_model = config.create_model()
    if torch.__version__.startswith('2.'):
        print('Compiling model with torch.compile')
        target_model.model = torch.compile(target_model.model)

    # Build the datasets
    train_set, valid_set, test_set = config.create_datasets()

    
    
    '''
    # Save images locally to inspect train set 
    outdir1 = "testmedia/images-class1" #just for testing
    os.makedirs(outdir1, exist_ok=True) #just for testing

    outdir2 = "testmedia/images-class2" #just for testing
    os.makedirs(outdir2, exist_ok=True) #just for testing

    for i in range(100):
        if (train_set[i][1]==0): #for class 1
            filename = f"{outdir1}/{i}.png" 
            #print('train set ', train_set[i]) prints tensor holding image data and target
            torchvision.utils.save_image(train_set[i][0], filename) 

        if (train_set[i][1]==1): #for class 2
            filename = f"{outdir2}/{i}.png" 
            #print('train set ', train_set[i]) prints tensor holding image data and target
            torchvision.utils.save_image(train_set[i][0], filename) 
    print('images saved')
    ''' 


    criterion = torch.nn.CrossEntropyLoss()
    metric = Accuracy

    # Set up optimizer and scheduler
    optimizer = config.create_optimizer(target_model)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # Modify the save_path such that subfolders with a timestamp and the name of the run are created
    time_stamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(config.training['save_path'],
                             f"{config.model['architecture']}_{time_stamp}")
    
    # Start training
    target_model.fit(
        training_data=train_set,
        validation_data=valid_set,
        test_data=test_set,
        criterion=criterion,
        metric=metric,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        attributes=attributes,
        hidden_attributes=hidden_attributes,
        ratio=ratio,
        run_name=run_name,
        rtpt=rtpt,
        config=config,
        batch_size=config.training['batch_size'],
        num_epochs=config.training['num_epochs'],
        dataloader_num_workers=config.training['dataloader_num_workers'],
        enable_logging=config.wandb['enable_logging'],
        wandb_init_args=config.wandb['args'],
        save_base_path=save_path,
        config_file=args.config)
    

if __name__ == '__main__':
    main()


