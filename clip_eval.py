from PIL import Image

import wandb
from transformers import CLIPProcessor, CLIPModel
from utils.stylegan import create_image 

import os
from os import listdir

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#if image is already in folder
#image = Image.open("/workspace/media_images_final_images_0_d263f1d46c6d5902de56.png")

#reconstruct images from wandb run 
api = wandb.Api(timeout=60)
run = api.run("model_inversion_attacks/ga0mt8yu")

for file in run.files():
    if file.name.startswith("media/images/final_"):
    #if file.name.startswith("results/optimized_w_selected"):    
        file.download(exist_ok=True) #wandb only downloads if file does not already exist


prompts = ["a photo of a person without beard", "a photo of a person with beard"]

for i in os.listdir("media/images"):
    image = Image.open("media/images/" + str(i))
    



 #img_results = create_image(final_w,
  #                          generator,
                            #crop_size=config.attack_center_crop,
                            #resize=config.attack_resize,
   #                         device=device).cpu()



    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    #print(logits_per_image[0])
    print("IMAGE ", str(i), probs)

