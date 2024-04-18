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


    '''
    Examples for prompts
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
    wandb.init( 
        project=config.wandb_project,
        name = config.wandb_name,
        config={
           "dataset": "Eyeglasses-Testdata-With & Eyeglasses-Testdata-Without",
           "prompts": prompts})
    
    # Load CLIP 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    #tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    #prompt_accuracy(prompts[0], processor, model) # use this to calc acc for prompts

    identify_attributes(prompts, processor, model) # use this for acc of majority vote

   
   
def prompt_accuracy(prompt, clip_processor, clip_model):
    # evaluates CLIP Accuracy for given prompt on Testdata
    # prompts should be provided as array of 2 strings, where the first prompt describes hidden attribute and second its negation
    # eg ['a photo of a man', 'a photo of a woman']

    ####################################
    ## dataset with hidden attribute ###
    ####################################

    highest_prop_0 = 0.0
    lowest_prop_0 = 1.0
    decision_0 = 0.0
    counter_0 = 0.0

    for i in os.listdir("Eyeglasses-Testdata-With"):
        image = Image.open("Eyeglasses-Testdata-With/" + str(i)) 
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # CLIP similarity score
        prob = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
        prob_0 = prob[0,0].item()
        prob_1 = prob[0,1].item()
        decision_0 += 1 if prob.argmax().item() == 0 else 0 #if CLIP decides for prompt with idx 0 add 1
        counter_0 += 1

        #update highest and lowest probabilities for hidden attribute

        if float(prob_0) > highest_prop_0:
            highest_prop_0 = float(prob_0)

        if float(prob_0) < lowest_prop_0:
            lowest_prop_0 = float(prob_0)
    
    acc_0 = decision_0/counter_0

    print('accuracy 0: ', acc_0)
    print("highest prob 0: ", highest_prop_0)
    print("lowest prob 0: ", lowest_prop_0)

    ####################################
    ## dataset without hidden attribute ###
    ####################################

    highest_prop_1 = 0.0
    lowest_prop_1 = 1.0
    decision_1 = 0.0
    counter_1 = 0.0

    for i in os.listdir("Eyeglasses-Testdata-Without"):
        image = Image.open("Eyeglasses-Testdata-Without/" + str(i)) 
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image # CLIP similarity score
        prob = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
        prob_0 = prob[0,0].item()
        prob_1 = prob[0,1].item()
        decision_1 += 1 if prob.argmax().item() == 1 else 0 #if CLIP decides for prompt with idx 0 add 1
        counter_1 += 1

        # update highest and lowest probabilities for hidden attribute
        if float(prob_1) > highest_prop_1:
            highest_prop_1 = float(prob_1)

        if float(prob_1) < lowest_prop_1:
            lowest_prop_1 = float(prob_1)
    
    acc_1 = decision_1/counter_1

    print('accuracy 1: ', acc_1)
    print("highest prob 1: ", highest_prop_1)
    print("lowest prob 1: ", lowest_prop_1)

    overall_acc = np.mean([acc_0, acc_1])

 
    wandb.log({
        "with glasses acc": acc_0,
        "without glasses acc": acc_1,
        "overall acc": overall_acc,
        "highest prop with glasses": highest_prop_0,
        "lowest prop with glasses": lowest_prop_0,
        "highest prop without glasses": highest_prop_1,
        "lowest prop without glasses": lowest_prop_1
        })

def identify_attributes(prompts, clip_processor, clip_model):
     #automatic evaluation of all images 
    
    ####################################
    ## dataset with hidden attribute ###
    ####################################
    
    decisions = []
    for i in os.listdir("Eyeglasses-Testdata-With"):
        all_probs = torch.tensor([])
        decision = 0.0

        image = Image.open("Eyeglasses-Testdata-With/" + str(i)) 
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            all_probs = torch.cat((all_probs, probs),0) #stores probabilities for each prompt
        
        # majority vote over all prompts: decides 0 for first prompt in array, 1 for 2nd prompt in array 
        decision = 1 if torch.sum(torch.argmax(all_probs, dim=1))/len(prompts) > 0.5 else 0
        decisions.append(decision) 
       
    acc_0 = (len(decisions)-np.sum(decisions))/len(decisions) 

    print("Percentage identified as wearing glasses from with glasses testset: ", acc_0)  
    
    ######################################
    ## dataset negated hidden attribute###
    ######################################

    decisions = []
    for i in os.listdir("Eyeglasses-Testdata-Without"):
        all_probs = torch.tensor([])
        decision = 0.0
        image = Image.open("Eyeglasses-Testdata-Without/" + str(i)) 
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            all_probs = torch.cat((all_probs, probs),0) #stores probabilities for each prompt
        
        # majority vote over all prompts: decides 0 for first prompt in array, 1 for 2nd prompt in array 
        decision = 1 if torch.sum(torch.argmax(all_probs, dim=1))/len(prompts) > 0.5 else 0
        decisions.append(decision) 

    acc_1 = np.sum(decisions)/len(decisions)
    print("Percentage identified as not wearing glasses from testset without glasses: ", acc_1)  

    
    overall_acc = np.mean([acc_0, acc_1])
    print("Overall accuracy: ", overall_acc)

    wandb.log({
        "with glasses acc": acc_0,
        "without glasses acc": acc_1,
        "overall acc": overall_acc,
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
