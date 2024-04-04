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

    
    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # Load pretrained CLIP 
    clip_model, clip_processor = load_clip()
    
    image_location = config.image_location # 'local', 'wandb-media' or 'wandb-weights'

    wandb.init(
        project=config.wandb_project,
        name = config.wandb_name,
        config={
            "dataset": config.wandb_attack_run,
            "prompts": config.prompts,
            }
    )


    if (image_location == 'wandb-weights'):
        #load stylegan
        G = load_generator(config.stylegan_model)
    else:
        G = None
        
    get_images(run, image_location, G)
    print("All images loaded from ", str(image_location))

    prompts = config.prompts
    print(prompts)
    

    acc_attr = identify_attributes(prompts, clip_processor, clip_model)

    print("identified as male appearing: ", acc_attr)
    print("identified as female appearing: ", acc_attr)
    wandb.log({"accurracy male appearing: ", 1-acc_attr})
    wandb.finish()

def load_clip():
    # use transformers to load pretrained clip model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor
   
def get_images(run, image_location, G=None):
    #gets images from wandb attack run and stores them in media/images
    if (image_location == 'local'):
        print('using locally stored images for CLIP evaluation')
        #if os.path.exists("media/images"):
         #   c = len([f for f in os.listdir("media/images")]) #TODO uncomment
        if os.path.exists("with_beard"):
            c = len([f for f in os.listdir("with_beard")])
            #print("found ", str(c), " images locally in media/images")
        else: 
            raise FileNotFoundError(f"The images are not found in media/images. Use wandb-media or wandb-weights instead")
        
    
    elif (image_location == 'wandb-media'):
        print('using images on wandb run for CLIP evaluation')
        # if image is stored on wandb: download it to local file
        c = 0
        for file in run.files():
            if file.name.startswith("media/images/final_"):  
                file.download(exist_ok=True) #wandb only downloads if file does not already exist
                c +=1
            else:
                raise FileNotFoundError(f"The images are not found on wandb. Use wandb-weights instead")
        print("downloaded ", str(c), " images from wandb")

    elif (image_location == 'wandb-weights'):
        print('using wandb weight vector to generate images for CLIP evaluation')
       
        # make local directory to store generated images
        outdir = "media/images-100"
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
    
        x = synthesis(w_expanded, noise_mode='const', force_fp32=True)

        print(x.shape)
        # crop and resize
        x = F.resize(x, 224, antialias=True)
        #x = F.center_crop(x, (800, 800)) #crop images
        x = (x * 0.5 + 128 / 224).clamp(0, 1) #maps from [-1,1] to [0,1]

        #save images
        for i in range(x.shape[0]):
            torchvision.utils.save_image(x[i], f'{outdir}/img-{i}.png') 

def identify_attributes(prompts, clip_processor, clip_model):
     #automatic evaluation of all images in "media/images" 
    decisions = []
    for i in os.listdir("media/images"):
        image = Image.open("media/images/" +str(i)) 
        all_probs = torch.tensor([])
        decision = 0.0
    
        for prompt in prompts:
            inputs = clip_processor(text=prompt, images=image, return_tensors="pt") #process using CLIP
            outputs = clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # CLIP similarity score
            probs = logits_per_image.softmax(dim=1) #softmax to get probability for prompts
            all_probs = torch.cat((all_probs, probs),0) #stores probabilities for each prompt
        
        # majority vote over all prompts: decides 0 for first prompt in array, 1 for 2nd prompt in array 
        decision = 1 if torch.sum(torch.argmax(all_probs, dim=1))/len(prompts) > 0.5 else 0
        decisions.append(decision) 
       
    acc_attr = (len(decisions)-np.sum(decisions))/len(decisions) 

    print("Percentage identified as male-appearing: ", acc_attr)  
    return acc_attr 

    
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
