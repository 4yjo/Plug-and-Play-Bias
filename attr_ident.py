import argparse
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

import wandb
from rtpt import RTPT
from utils.attr_ident_config_parser import AttrIdentConfigParser

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from utils.stylegan import load_generator

import os
#from os import listdir


def main():
    """Use to inspect images or vectors from former wandb runs.
    Evaluation during attack is already implemented in attack.py"""


    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)

    api = wandb.Api(timeout=60)
    run = api.run(config.wandb_attack_run)

    
    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # Load pretrained CLIP 
    clip_model, clip_processor = load_clip()
    
    image_location = config.image_location # 'local', 'wandb-media' or 'wandb-weights'
   

    if (image_location == 'wandb-weights'):
        #load stylegan
        G = load_generator(config.stylegan_model)
    else:
        G = None
        
    get_images(run, image_location, G)
    print("All images loaded from ", str(image_location))

    prompts = config.prompts

    for prompt in prompts:
        if not isinstance(prompt, list):
            raise ValueError("prompts must be 2d array, e.g. [['a boy','a girl']]")

    
    
    # identifies bias attribute in images, e.g. male and counts number of images with
    # the attribute for each class e.g. class 1 = blond hair, class2 = black hair
    identify_attributes(prompts, clip_processor, clip_model, run)


def load_clip():
    # use transformers to load pretrained clip model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor
   
def get_images(run, image_location, G=None):
    #gets images from wandb attack run and stores them in media/images
    if (image_location == 'local'):
        if os.path.exists("media/images"):
            c = len([f for f in os.listdir("media/images")])
            print('Found ', c, ' images in media/images')
        else: 
            raise FileNotFoundError(f"The images are not found in media/images. Use wandb-weights instead")
        

    elif (image_location == 'wandb-weights'):
        print('using wandb weight vector to generate images for CLIP evaluation')
       
        # make local directory to store generated images
        outdir = "media/images" 
        os.makedirs(outdir, exist_ok=True)

        # Set devices
        torch.set_num_threads(24)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_devices = [i for i in range(torch.cuda.device_count())]

        # initialize stylegan syntezesis network 
        synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
        synthesis.num_ws = G.num_ws

        synthesis.eval()
        
        # get weights from wandb
        for file in run.files():
            if file.name.startswith("results/optimized_w_selected"):    
                w_file = file.download(exist_ok=True) #wandb only downloads if file does not already exist
                print('weights downloaded')
                
                w = torch.load(w_file.name).cuda() #loads tensor from file 
                print(w.shape)

                # copy data to match dimensions
                if w.shape[1] == 1:
                    w_expanded = torch.repeat_interleave(w,
                                                    repeats=synthesis.num_ws,
                                                    dim=1)
                else: 
                    w_expanded = w
        
        print(w_expanded.shape)
        x = synthesis(w_expanded, noise_mode='const', force_fp32=True)

        print(x.shape)
        # crop and resize
        x = F.resize(x, 224, antialias=True)
        #x = F.center_crop(x, (800, 800)) #crop images
        x = (x * 0.5 + 128 / 224).clamp(0, 1) #maps from [-1,1] to [0,1]
        print(x.shape)
        
        #save images
        for i in range(x.shape[0]):
            torchvision.utils.save_image(x[i], f'{outdir}/{i}.png') 

def identify_attributes(prompts, clip_processor, clip_model, run):
     #automatic evaluation of all images in "media/images" 


    # split image directory to group images by class 1 and class 2
    all_img = sorted(os.listdir("media/images"), key=lambda x: int(x.split('.')[0])) # lambda ensures numerical sorting of files with naem 0.png, 1.png etc
    
    c1_img = all_img[:int(len(all_img)/2)]
    c1_decisions = []

    c2_img = all_img[int(len(all_img)/2):]
    c2_decisions = []

    for  i in c1_img:
        image = Image.open("media/images/" +str(i)) 

        c1_all_probs = torch.tensor([])
        c1_decision = 0.0
    
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            c1_all_probs = torch.cat((c1_all_probs, probs),0) #stores probabilities for each prompt

        # majority vote over all prompts
        # decision = 0 for first prompt in array, 1 for 2nd prompt in array 
        c1_decision = 1 if torch.sum(torch.argmax(c1_all_probs, dim=1))/len(prompts) > 0.5 else 0
        c1_decisions.append(c1_decision) 

        
    for  i in c2_img:
        image = Image.open("media/images/" +str(i)) 
        c2_all_probs = torch.tensor([])
        c2_decision = 0.0
    
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            c2_all_probs = torch.cat((c2_all_probs, probs),0) #stores probabilities for each prompt
    
        
        # majority vote over all prompts
        # decision = 0 for first prompt in array, 1 for 2nd prompt in array 
        c2_decision = 1 if torch.sum(torch.argmax(c2_all_probs, dim=1))/len(prompts) > 0.5 else 0
        c2_decisions.append(c2_decision) 


    c1_attr_count = (len(c1_decisions)-np.sum(c1_decisions))/len(c1_decisions)  # -> get percentage of images with attribute described in 1st prompt(s)
    c2_attr_count = (len(c2_decisions)-np.sum(c2_decisions))/len(c2_decisions)  # -> get percentage of images with attribute described in 1st prompt(s)


    c1_total = np.sum(c1_decisions)/len(c1_decisions)
    c2_total = np.sum(c2_decisions)/len(c1_decisions)
    
    print(f'Identified as {prompt[0]} in Class 1: {c1_total}')
    print(f'Identified as {prompt[0]} in Class 2: {c2_total}')
    print(f'c1-10 {np.sum(c1_decisions[:10])/10}')
    print(f'c1-25 {np.sum(c1_decisions[:25])/25}')
    print(f'c2-10 {np.sum(c2_decisions[:10])/10}')
    print(f'c2-25 {np.sum(c2_decisions[:25])/25}')



    # add results to wandb attack run logs
    run.summary.update({f'{prompt[0]} in Class 1': c1_total})
    run.summary.update({f'{prompt[0]} in Class 2': c2_total})
    run.summary.update({'c1-10': np.sum(c1_decisions[:10])/10})
    run.summary.update({'c1-25': np.sum(c1_decisions[:25])/25})
    run.summary.update({'c2-10': np.sum(c2_decisions[:10])/10})
    run.summary.update({'c2-25': np.sum(c2_decisions[:25])/25})
    run.config.update({'prompts': prompt})


    return c1_attr_count, c2_attr_count

    
    '''
    # Define the bin edges
    bin_edges = np.arange(0, 1.1, 0.1)

    #   Count the number of elements in each bin
    bin_counts, _ = np.histogram(cp_0, bins=bin_edges)

    # Plot the histogram
    plt.bar(range(len(bin_counts)), bin_counts, tick_label=[f'{i}0-{i+1}0' for i in range(10)])
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.title('Distribution class prob test data with beard')
    plt.tight_layout()
    plt.savefig('dist-with-beard.png')

    # Plot the histogram
    plt.bar(range(len(bin_counts2)), bin_counts2, tick_label=[f'{i}0-{i+1}0' for i in range(10)])
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.title('Distribution similarity score test data with beard')
    plt.tight_layout()
    plt.savefig('similarity-with-beard.png')
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
