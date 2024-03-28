import argparse
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

import wandb
from rtpt import RTPT
from utils.attr_ident_config_parser import AttrIdentConfigParser

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
from utils.stylegan import load_generator
#from utils.wandb import load_model
import os
#from os import listdir


def main():
    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)

    api = wandb.Api(timeout=60)
    run = api.run(config.wandb_attack_run)

    # if attribute is specified in config file, prompts can be generated automatically 
    attribute = config.attribute 

    '''
    prompts = ["a photo of a person with " + attribute, 
            "an image of a person with "+attribute, 
            "a cropped photo of a person with "+attribute,
            "an image of a head of a person with "+attribute,
            "a photo of a person with no " + attribute, 
            "an image of a person with no "+attribute, 
            "a cropped photo of a person with no "+attribute,
            "an image of a head of a person with no "+attribute]
    
    
    prompts = [["a photo of a person with no " + attribute,  "a photo of a person with " + attribute], 
            ["an image of a person with no "+attribute,  "an image of a person with "+attribute],  
            ["a cropped photo of a person with no "+attribute, "a cropped photo of a person with "+attribute],
            ["an image of a head of a person with no "+attribute, "an image of a head of a person with "+attribute],
            ["a portrait of a person with no "+attribute, "a portrait of a person with "+attribute]]
    
    
    prompts = [["an image of a person with no "+attribute,  "an image of a person with "+attribute],  
            ["a cropped photo of a person with no "+attribute, "a cropped photo of a person with "+attribute],
            ["an image of a head of a person with no "+attribute, "an image of a head of a person with "+attribute]]

    '''
   
    prompts = config.prompts
    print(prompts)
          

    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # wandblog
     # start a new wandb run to track this script
    
    wandb.init( 
        project=config.wandb_project,
        name = config.wandb_name,
        config={
           "dataset": "Gender-Testdata-Female & Gender-Testdata-Male",
           "prompts": prompts})
    
    # Load CLIP 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    image_location = config.image_location # 'local', 'wandb-media' or 'wandb-weights'

    if (image_location == 'wandb-weights'):
        #load stylegan
        G = load_generator(config.stylegan_model)
    else:
        G = None

    get_images(run, image_location, G)

    identify_attributes(prompts, processor, model)
   
def get_images(run, image_location, G=None):
    #create a local folder with test images to evaluate your prompts and put its path here:
    
    if os.path.exists("Gender-Testdata-Female"):
        c = len([f for f in os.listdir("Gender-Testdata-Female")])
        print("found ", str(c), " images")
    else: 
        raise FileNotFoundError(f"The images are not found in media/images. Use wandb-media or wandb-weights instead")


def identify_attributes(prompts, clip_processor, clip_model):
     #automatic evaluation of all images 
    
    ####################################
    ## dataset with hidden attribute ###
    ####################################
    
    decisions = []
    for i in os.listdir("Gender-Testdata-Male"):
        all_probs = torch.tensor([])
        decision = 0.0
        #best_probs = []
        #best_prompts = []
        image = Image.open("Gender-Testdata-Male/" + str(i)) 
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt") #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            print(probs)
            all_probs = torch.cat((all_probs, probs),0) #stores probabilities for each prompt
            #best_probs.append(probs.max().item()) #append
            #best_prompts.append(probs.argmax()) #append index of best rated

        # majority vote over all prompts: decides 0 for first prompt in array, 1 for 2nd prompt in array 
        decision = torch.sum(torch.argmax(all_probs, dim=1))/len(prompts)
        decisions.append(decision.item()) 
        # majority vote over all prompts: decides 1 for attr, 0 for no attr  
       
        # majority vote over all prompts: decides 1 for attr, 0 for no attr  
        #decision = torch.sum(torch.argmax(all_probs, dim=1))/len(prompts)
        #decisions.append(decision.item()) 
        highest_prop_0 = torch.argmax(all_probs, dim=1) 
        print(highest_prop_0, i)
        lowest_prop_0 =torch.argmin(all_probs, dim=1)
        print(lowest_prop_0, i)
        decision = torch.round(torch.sum(highest_prop_0)/len(prompts))
        decisions.append(decision.item()) 


    acc_0 = np.sum(decisions)/len(decisions)
    print("Percentage identified as male-appearing from male testset: ", acc_0)  
    
    ######################################
    ## dataset negated hidden attribute###
    ######################################

    decisions = []
    for i in os.listdir("Gender-Testdata-Female"):
        all_probs = torch.tensor([])
        decision = 0.0
        best_probs = []
        best_prompts = []
       
        image = Image.open("Gender-Testdata-Female/"+str(i))
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt") #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            print(probs)
            all_probs = torch.cat((all_probs, probs),0) #stores probabilities for each prompt
            #best_probs.append(probs.max().item()) #append
            #best_prompts.append(probs.argmax()) #append index of best rated

        # majority vote over all prompts: decides 1 for attr, 0 for no attr  
        #decision = torch.sum(torch.argmax(all_probs, dim=1))/len(prompts)
        #decisions.append(decision.item()) 
        highest_prop_1 = torch.argmax(all_probs, dim=1) 
        print(highest_prop_1, i)
        lowest_prop_1 =torch.argmin(all_probs, dim=1)
        print(lowest_prop_1, i)
        decision = torch.round(torch.sum(highest_prop_1)/len(prompts))
        decisions.append(decision.item()) 


    acc_1 = np.sum(decisions)/len(decisions)
    print("Percentage identified as female-appearing from female testset: ", acc_1)  

    
    overall_acc = np.mean([acc_0, acc_1])
    print("Overall accuracy: ", overall_acc)

    wandb.log({
        "male acc": acc_0,
        "female acc": acc_1,
        "overall acc": overall_acc,
        "highest prop male": highest_prop_0,
        "lowest prop male": lowest_prop_0,
        "highest prop female": highest_prop_1,
        "lowest prop female": lowest_prop_1
        })

    wandb.finish

def create_parser():
    parser = argparse.ArgumentParser(
        description='automated evaluation using CLIP')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attr_ident config
    config = AttrIdentConfigParser(args.config)

    return config, args

if __name__ == '__main__':
    main()
