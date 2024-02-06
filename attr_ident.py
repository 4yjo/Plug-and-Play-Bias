import argparse
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np

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
    #torch.manual_seed(42)
    torch.manual_seed(config.seed)

    api = wandb.Api(timeout=60)
    run = api.run(config.wandb_attack_run)

    attribute = config.attribute 
    prompt = ["a photo of a person with no " + attribute, "a photo of a person with " + attribute]
    

    # Create and start RTPT object
    rtpt = config.create_rtpt()
    rtpt.start()

    # wandblog
     # start a new wandb run to track this script
    
    wandb.init(
        project=config.wandb_project,
        name = config.wandb_name,
        config={
            "dataset": "test-data-beard",
            "prompts": prompt,
            "acc_above": "0.8",
            }
        )
    
    # Load CLIP 
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image_location = config.image_location # 'local', 'wandb-media' or 'wandb-weights'

    if (image_location == 'wandb-weights'):
        #load stylegan
        G = load_generator(config.stylegan_model)
    else:
        G = None

    get_images(run, image_location, G)

    identify_attributes(prompt, processor, model)


   


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
    
        x = synthesis(w_expanded, noise_mode='const', force_fp32=True)

        print(x.shape)
        # crop and resize
        x = F.resize(x, 224, antialias=True)
        #x = F.center_crop(x, (800, 800)) #crop images
        x = (x * 0.5 + 128 / 224).clamp(0, 1) #maps from [-1,1] to [0,1]

        #save images
        for i in range(x.shape[0]):
            torchvision.utils.save_image(x[i], f'{outdir}/img-{i}.png') 

    


def identify_attributes(prompt,clip_processor, clip_model):
     #automatic evaluation of all images saved to local folder
    print(prompt)
    sim_scores= []
    class_probs = []
    for i in os.listdir("media/images"):
        image = Image.open("media/images/" + str(i))
        inputs = clip_processor(text=prompt, images=image, return_tensors="pt", padding=True) #process using CLIP
        outputs = clip_model(**inputs)
        #logits_per_image = outputs.logits_per_image.cpu().detach().numpy()  # get image-text similarity score
        logits_per_image = outputs.logits_per_image
        np_score = logits_per_image.cpu().detach().numpy()
        sim_scores.append(np_score)
        prob = logits_per_image.softmax(dim=1).cpu().detach().numpy()  # softmax to get the label probabilities 
        class_probs.append(prob)
        #all_probs_0.append(probs[0][0])
        #all_probs_1.append(probs[0][1])
    

    sc = np.squeeze(sim_scores)
    print('sc', sc[:5])
    cp = np.squeeze(class_probs)
    print('cp', cp[:5])
    
    cp_0 = [i for i in cp[0]]
    print("cp0",len(cp_0))
    print("mean",np.mean(cp_0))

    wandb.log({"mean similarity score": np.mean(sc), "lowest similarity score": np.amin(sc), "highest similarity score": np.amax(sc)})
    #todo get indices of low scores
    wandb.finish()

    #print(len(all_probs_0))
    #print(prob)
    #all_probs = np.squeeze(all_probs_0)
    #print(all_probs[0])

    #wandb.log({"all_prob_0": all_probs_0, "mean_prob_0": np.mean(all_probs_0), "lowest_prob_0": np.amin(all_probs_0), "highest_prob_0": np.amax(all_probs_0)})
    
     #calc accuracy 
    #benchmark = 0.8
    #counter = np.where(all_probs > benchmark)[0]
    #acc_0 = len(counter)/len(all_probs_0)
    #print("accuracy_prompt_0: ", acc_0)
    
    #wandb.finish() 

    #acc_1 = (torch.sum(all_probs_1 > benchmark).item())/len(all_probs_1)
    #print("accuracy_prompt_1: ", acc_1)

    #wandb.log({"acc_prompt_0": acc_0, "acc_prompt_1":acc_1})

'''

    all_scores = torch.cat(log_scores, dim=0)
    lowest_score = round(all_scores.min(), 4)
    lowest_score_idx = all_scores.argmin()
    highest_score = round(all_scores.max(), 4)
    highest_score_idx = all_scores.argmax()
    mean_scores = round(all_scores.mean(), 4)

    wandb.log({"mean_score": mean_scores,
               "lowest_score": lowest_score,
               "lowest_score_idx": lowest_score_idx,
               "highest_score": highest_score,
               "highest_score_idx": highest_score_idx})
    
    all_probs = torch.cat(log_probs, dim=0)
    lowest_probs = round(all_probs.min(), 4)
    lowest_probs_idx = all_probs.argmin()
    highest_probs = round(all_probs.max(), 4)
    highest_probs_idx = all_probs.argmax()
    mean_probs = round(all_probs.mean(), 4)

    wandb.log({"mean_probs": mean_probs,
               "lowest_probs": lowest_probs,
               "lowest_probs_idx": lowest_probs_idx,
               "highest_probs": highest_probs,
               "highest_probs_idx": highest_probs_idx})
    
    # calculate accuracy
    benchmark = 80
    acc = (torch.sum(all_probs > benchmark).item())/len(all_probs)
    print("accuracy: ", acc)

    wandb.log({"accuracy": acc})

    #all_probs = log_probs.detach().numpy()
    #wandb.log({"all_prob": all_probs, "mean_prob": np.mean(all_probs), "lowest_prob": np.amin(all_probs), "highest_prob": np.amax(all_probs)})
    wandb.finish()         
  

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
