import argparse
import torch
import torchvision


import wandb
from rtpt import RTPT
from utils.attr_ident_config_parser import AttrIdentConfigParser

from transformers import CLIPProcessor, CLIPModel
 
from utils.stylegan import load_generator
#from utils.wandb import load_model
import os
#from os import listdir


def main():
    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    #torch.manual_seed(42)
    torch.manual_seed(config.seed)

    api = wandb.Api(timeout=60)
    run = api.run(config.wandb_attack_run)

    # Load CLIP 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


    # make local directory to store generated images
    outdir = "media/images"
    os.makedirs(outdir, exist_ok=True)


    image_location = config.image_location # 'local', 'wandb-media' or 'wandb-weights'

    if (image_location == 'wandb-weights'):
        #load stylegan
        G = load_generator(config.stylegan_model)
    else:
        G = None

    get_images(run, image_location, G)

    # TODO übergebe prompt und image folder
    #identify_attributes()

    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # TODO wandblog
'''
def load_w_from_wandb(run, generator):
    # get optimized w from wandb attack run
    for file in run.files():
        if file.name.startswith("results/optimized_w_selected"):    
            w_file = file.download(exist_ok=True) #wandb only downloads if file does not already exist
            print('weights downloaded')
            
            w = torch.load(w_file.name).cuda() #loads tensor from file 
            print(w.shape)

            # copy data to match dimensions
            if w.shape[1] == 1:
                w_expanded = torch.repeat_interleave(w,
                                                repeats=generator.num_ws,
                                                dim=1)
            else: 
                w_expanded = w
    return w_expanded
'''
def get_images(run, image_location, G=None):
    #gets images from wandb attack run and stores them in media/images
    if (image_location == 'local'):
        print('using locally stored images for CLIP evaluation')
        if os.path.exists("media/images"):
            c = len([f for f in os.listdir("media/images")])
            print("found ", str(c), " images locally in media/images")
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
        # Load pre-trained StyleGan2 components to generate images if they don't already exist
       
        # make local directory to store generated images
        outdir = "media/images"
        os.makedirs(outdir, exist_ok=True)

         # Set devices
        torch.set_num_threads(24)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        gpu_devices = [i for i in range(torch.cuda.device_count())]

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

        # save image
        torchvision.utils.save_image(x[0], f'{outdir}/test.png') # TODO name images correct

    

'''
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


        # TODO throw error if no images in the folder
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
