from PIL import Image
import random
import numpy as np
import torch


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

    image_location = 'wandb-weights' # local, wandb-media, wandb-weights
    get_images(run, image_location)

    # TODO übergebe prompt und image folder
    #identify_attributes()

    # Create and start RTPT object
    #rtpt = config.create_rtpt()
    #rtpt.start()

    # TODO wandblog


def get_images(run, image_location):
    if (image_location == 'local'):
        print('using locally stored images for CLIP evaluation')
        pass 
        # TODO test - maybe double check path
    
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
        # Load pre-trained StyleGan2 components to generate images if they don't already exist
        #TODO: put in config
        stylegan_model = "stylegan2-ada-pytorch/ffhq.pkl"

        # Set devices
        torch.set_num_threads(24)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_devices = [i for i in range(torch.cuda.device_count())]

        # make local directory to store generated images
        outdir = "media/images"
        os.makedirs(outdir, exist_ok=True)

        G = load_generator(stylegan_model)
        #G = load_generator(config.stylegan_model)
        num_ws = G.num_ws # ?

        # initialize synthesis network to generate image from given w
        synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
        synthesis.num_ws = num_ws

        # get optimized w from wandb attack run
        for file in run.files():
            if file.name.startswith("results/optimized_w_selected"):    
                w_file = file.download(exist_ok=True) #wandb only downloads if file does not already exist
                print('weights downloaded')
        
                w = torch.load(w_file.name) #loads tensor from file 
                print(w.shape)

                # re-generate image 
                img = create_image(w,
                            synthesis,
                            crop_size= 800, #crop_size=config.attack_center_crop, #TODO put in config
                            resize= 224) #resize=config.attack_resize) #TODO put in config

                print('img',img.shape)

                #convert to numpy array and permute order from batch, channel, hight, width to batch, height, width,channel
                img_np = img.permute(0,2,3,1).cpu().numpy() 
                print('img_np', img_np.shape)

                #PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
                Image.fromarray(img_np[40], 'RGB').save(f'{outdir}/test.png')
                print("IMAGE SAVED") 

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
