import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from utils.stylegan import adjust_gen_images


from transformers import CLIPProcessor, CLIPModel
import torchvision
import os
from PIL import Image

def find_initial_w(generator,
                   target_model,
                   targets,
                   search_space_size,
                   clip=True,
                   center_crop=768,
                   resize=224,
                   horizontal_flip=True,
                   filepath=None,
                   truncation_psi=0.7,
                   truncation_cutoff=18,
                   batch_size=25,
                   seed=0):
    """Find good initial starting points in the style space.

    Args:
        generator (torch.nn.Module): StyleGAN2 model
        target_model (torch.nn.Module): [description]
        target_cls (int): index of target class.
        search_space_size (int): number of potential style vectors.
        clip (boolean, optional): clip images to [-1, 1]. Defaults to True.
        center_crop (int, optional): size of the center crop. Defaults 768.
        resize (int, optional): size for the resizing operation. Defaults to 224.
        horizontal_flip (boolean, optional): apply horizontal flipping to images. Defaults to true.
        filepath (str): filepath to save candidates.
        truncation_psi (float, optional): truncation factor. Defaults to 0.7.
        truncation_cutoff (int, optional): truncation cutoff. Defaults to 18.
        batch_size (int, optional): batch size. Defaults to 25.

    Returns:
        torch.tensor: style vectors with highest confidences on the target model and target class.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.from_numpy(
        np.random.RandomState(seed).randn(search_space_size,
                                          generator.z_dim)).to(device)
    c = None
    target_model.eval()
    five_crop = None

    # initialize CLIP #TODO put somewhere else later

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    prompt = ['a person with glasses', 'a person with no glasses']


    # TODO for testing only
    outdir = "media/test" 
    os.makedirs(outdir, exist_ok=True)

    with torch.no_grad():
        confidences = []
        bias_attributes = []
        final_candidates = []
        final_confidences = []
        candidates = generator.mapping(z,
                                       c,
                                       truncation_psi=truncation_psi,
                                       truncation_cutoff=truncation_cutoff)
        candidate_dataset = torch.utils.data.TensorDataset(candidates)
        for w in tqdm(torch.utils.data.DataLoader(candidate_dataset,
                                                  batch_size=batch_size),
                      desc='Find initial style vector w'):
            imgs = generator.synthesis(w[0],
                                       noise_mode='const',
                                       force_fp32=True)
            # Adjust images and perform augmentation
            if clip:
                lower_bound = torch.tensor(-1.0).float().to(imgs.device)
                upper_bound = torch.tensor(1.0).float().to(imgs.device)
                imgs = torch.where(imgs > upper_bound, upper_bound, imgs)
                imgs = torch.where(imgs < lower_bound, lower_bound, imgs)
            if center_crop is not None:
                imgs = F.center_crop(imgs, (center_crop, center_crop))
            if resize is not None:
                imgs = [F.resize(imgs, resize, antialias=True)]
            if horizontal_flip:
                imgs.append(F.hflip(imgs[0]))
            if five_crop:
                cropped_images = []
                for im in imgs:
                    cropped_images += list(F.five_crop(im))
                imgs = cropped_images

            target_conf = None
            for im in imgs:
                if target_conf is not None:
                    target_conf += target_model(im).softmax(dim=1) / len(imgs)
                else:
                    target_conf = target_model(im).softmax(dim=1) / len(imgs)
            confidences.append(target_conf)

           #bias_attr = [] # TODO what happens here? Initialize new list for each w?
            for im in imgs:
                for i in range(len(im)):
                    im[i] = (im[i] * 0.5 + 128 / 224).clamp(0, 1) #maps from [-1,1] to [0,1]

                    #TODO make this more elegant?
                    # match dimensions for CLIP processor
                    perm= im[i].permute(1, 2, 0) 
                    perm_rescale = perm.mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8)

                    # reference 
                    torchvision.utils.save_image(im[i], 'media/test/test.png')
                    test1 = Image.open('media/test/test.png')
                    test1_arr = np.asarray(test1) 

            
                    #torchvision.utils.save_image(im[i], 'media/test/temp_img.png') #save image for testing
                    #inputs = clip_processor(text=prompt, images=Image.open('media/test/test.png'), return_tensors="pt", padding=True)

                    inputs = clip_processor(text=prompt, images=perm_rescale, return_tensors="pt", padding=True)
                    
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                    probs = logits_per_image.softmax(dim=1)  # 0 -> with glasses, 1 -> with no glasses
                    bias_attr = torch.argmax(probs).item()
                    bias_attributes.append(bias_attr)
            
        #print('bias attributes', bias_attributes)


        results_test_image = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
        results_perm_magic =  [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]

        print(np.array_equal(results_perm_magic,results_test_image))
        print(np.array_equal(perm_rescale.detach().cpu().numpy(), test1_arr))

        print('conf shape',len(confidences))
        confidences = torch.cat(confidences, dim=0)
        print('conf shape cat',len(confidences))
        #bias_attributes = torch.cat(bias_attributes, dim=0)

        for target in targets:
            # find candidate with highest confidence for each target
            sorted_conf, sorted_idx = confidences[:,
                                                  target].sort(descending=True)
            
            #while bias_counter < 0.5: # TODO import actual ratio
            
            # TODO implement CLIP check for bias attribute here as second point for choice
        

            # only keep images with bias attribute as long as ratio is not representative
            # if has attribute:
            #   bias_counter += 1
            # else:
            #   sorted_conf.pop()
            #   sorted_idx.pop()

            # check for balance of bias attr in candidate selection
            #bias_counter = torch.sum(bias_attributes[sorted_idx[0]].unsqueeze(0))

            #diff = ratio*nr candidates per target - bias_counter
            #while diff > 0:
            #    sorted_idx.pop()
            #    sorted_conf.pop()

            final_candidates.append(candidates[sorted_idx[0]].unsqueeze(0)) #get image with hightes confidence
            final_confidences.append(sorted_conf[0].cpu().item())
            # Avoid identical candidates for the same target
            confidences[sorted_idx[0], target] = -1.0


    final_candidates = torch.cat(final_candidates, dim=0).to(device)
    final_confidences = [np.round(c, 2) for c in final_confidences]

    # TODO change selection process -> add CLIP evaluation for biased attribute


    print(f'Found {final_candidates.shape[0]} initial style vectors.')

    if filepath:
        torch.save(final_candidates, filepath)
        print(f'Candidates have been saved to {filepath}')
    return final_candidates
