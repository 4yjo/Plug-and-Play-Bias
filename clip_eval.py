from PIL import Image
import random
import numpy as np
import torch
#import torch_utils
#import pickle
#import dnnlib

import wandb
from rtpt import RTPT

from transformers import CLIPProcessor, CLIPModel
 
from utils.stylegan import create_image, load_discrimator, load_generator
import os
from os import listdir


def main():
   

    # Define and parse attack arguments
    # TODO parser = create_parser()
    # TODO config, args = parse_arguments(parser)

    # Set seeds
    #torch.manual_seed(42)
    # TODO torch.manual_seed(config.seed)

    #random.seed(config.seed)
    #np.random.seed(config.seed)

    random.seed(42)
    np.random.seed(42)
    

    api = wandb.Api(timeout=60)
    run = api.run("model_inversion_attacks/ga0mt8yu")

    # Load CLIP 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    
    #image = Image.open("/workspace/media_images_final_images_0_d263f1d46c6d5902de56.png")


    # Create and start RTPT object
    #rtpt = config.create_rtpt()
    #rtpt.start()

    image_location = 'wandb-weights' # local, wandb-media, wandb-weights
    get_images(run, image_location)

    # TODO übergebe prompt und image folder
    #identify_attributes()


def get_images(run, image_location):
    if (image_location == 'local'):
        print('using locally stored images for CLIP evaluation')
        pass 
        # TODO maybe double check path
    
    elif (image_location == 'wandb-media'):
        print('using images on wandb run for CLIP evaluation')
        # if image is stored on wandb: download it to local file
        c = 1
        for file in run.files():
            if file.name.startswith("media/images/final_"):  
                file.download(exist_ok=True) #wandb only downloads if file does not already exist
                c +=1
            # TODO throw error if no file found
        print("downloaded ", str(c), " images from wandb")

    elif (image_location == 'wandb-weights'):
        print('using wandb weight vector to generate images for CLIP evaluation')
        # if image is not stored on wandb: get weights from run and create image using pretrained stylegan
       
       # Load pre-trained StyleGan2 components to generate images if they don't already exist
        #G = load_generator(config.stylegan_model)

        stylegan_model = "stylegan2-ada-pytorch/ffhq.pkl"

        # Set devices
        torch.set_num_threads(24)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_devices = [i for i in range(torch.cuda.device_count())]

        G = load_generator(stylegan_model)
        #D = load_discrimator(config.stylegan_model)
        num_ws = G.num_ws # ?

        synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
        synthesis.num_ws = num_ws
        #discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

        # make local directory to store generated images
        outdir = "media/images"
        os.makedirs(outdir, exist_ok=True)

        for file in run.files():
            if file.name.startswith("results/optimized_w_selected"):    
                w_file = file.download(exist_ok=True) #wandb only downloads if file does not already exist
                print('weights downloaded')
        
                w = torch.load(w_file.name) #loads tensor from file 
                print(w.shape)

                # TODO put weights in right format
                img = create_image(w,
                            synthesis,
                            #crop_size=config.attack_center_crop,
                            #resize=config.attack_resize)
                            crop_size= 800,
                            resize= 224)
                print("IMAGE CREATED")  
                # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
                # TODO double-check image saving > numpy permute as in attack
                # print("IMAGE SAVED")

            

def identify_attributes():
    # TODO take prompts from config file
    prompts = ["a photo of a person without beard", "a photo of a person with beard"]

    #automatic evaluation of all images saved to local folder
    for i in os.listdir("media/images"):
        image = Image.open("media/images/" + str(i))

        inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True) #process using CLIP

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        #print(logits_per_image[0])
        print("IMAGE ", str(i), probs)

        # TODO log to wandb
        # TODO throw error if no images in the folder


if __name__ == '__main__':
    main()
