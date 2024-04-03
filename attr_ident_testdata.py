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
    '''
    # wandblog 
    wandb.init( 
        project=config.wandb_project,
        name = config.wandb_name,
        config={
           "dataset": "Gender-Testdata-Female & Gender-Testdata-Male",
           "prompts": prompts})
    '''
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

    #prompt_accuracy(prompts[0], processor, model) # TODO for all prompts
   
    identify_attributes(prompts, processor, model)
   
def get_images(run, image_location, G=None):
    #create a local folder with test images to evaluate your prompts and put its path here:
    
    if os.path.exists("Gender-Testdata-Female"):
        c = len([f for f in os.listdir("Gender-Testdata-Female")])
        print("found ", str(c), " images")
    else: 
        raise FileNotFoundError(f"The images are not found in media/images. Use wandb-media or wandb-weights instead")

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

    for i in os.listdir("Gender-Testdata-Male"):
        image = Image.open("Gender-Testdata-Male/" + str(i)) 
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt") #process using CLIP
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

    for i in os.listdir("Gender-Testdata-Female"):
        image = Image.open("Gender-Testdata-Female/" + str(i)) 
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt") #process using CLIP
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

    overall_acc = (acc_0+acc_1)/2

 
    wandb.log({
        "male acc": acc_0,
        "female acc": acc_1,
        "overall acc": overall_acc,
        "highest prop male": highest_prop_0,
        "lowest prop male": lowest_prop_0,
        "highest prop female": highest_prop_1,
        "lowest prop female": lowest_prop_1
        })

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
        prop_0 = torch.amax(all_probs, dim=1).item() #gets current prob for prompt with index 0
        #decision = torch.round(torch.sum(highest_prop_0)/len(prompts))
        #decisions.append(decision.item()) 

    #update highest and lowest probabilities
    if float(prop_0) > highest_prop_0:
        highest_prop_0 = float(prop_0)

    if float(prop_0) < lowest_prop_0:
        lowest_prop_0 = float(low_prop_0)

    print("highest prob: ", highest_prop_0)
    print("lowest_prob: ", lowest_prop_0)

    print('sum dec', np.sum(decisions))
    print(' len dec', len(decisions))

    #acc_0 counts percentage of decision = 0, where 0 -> index in prompt e.g. 'a photo of a man'
    acc_0 = (len(decisions)-np.sum(decisions))/len(decisions) 

    print("Percentage identified as male-appearing from male testset: ", acc_0)  
    
    ######################################
    ## dataset negated hidden attribute###
    ######################################

    decisions = []
    for i in os.listdir("Gender-Testdata-Female"):
        all_probs = torch.tensor([])
        decision = 0.0
        highest_prop_1 = 0.0
        lowest_prop_1 = 0.0
        best_probs = []
        best_prompts = []
       
        image = Image.open("Gender-Testdata-Female/"+str(i))
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt") #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            all_probs = torch.cat((all_probs, probs),0) #stores probabilities for each prompt
            #best_probs.append(probs.max().item()) #append
            #best_prompts.append(probs.argmax()) #append index of best rated

        # majority vote over all prompts: decides 1 for attr, 0 for no attr  
        decision = torch.sum(torch.argmax(all_probs, dim=1))/len(prompts)
        decisions.append(decision.item()) 
        high_prop_1 = torch.amax(all_probs, dim=1).item()
        low_prop_1 =torch.amin(all_probs, dim=1).item()
        #decision = torch.round(torch.sum(highest_prop_1)/len(prompts))
        #decisions.append(decision.item()) 

    #update highest and lowest probabilities
    if float(high_prop_1) > highest_prop_1:
        highest_prop_1 = float(high_prop_1)

    if float(low_prop_1) > lowest_prop_1:
        lowest_prop_1 = float(low_prop_1)

    print("highest prob: ", highest_prop_1)
    print("lowest_prob: ", lowest_prop_1)

    acc_1 = np.sum(decisions)/len(decisions)
    print("Percentage identified as female-appearing from female testset: ", acc_1)  

    
    overall_acc = np.mean([acc_0, acc_1])
    print("Overall accuracy: ", overall_acc)
'''
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
'''
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
