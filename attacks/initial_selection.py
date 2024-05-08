import numpy as np
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from utils.stylegan import adjust_gen_images


from transformers import CLIPProcessor, CLIPModel
import torchvision
import os
from PIL import Image

def find_initial_w():
    #TODO put original code here
    pass

def find_bal_initial_w(generator,
                   target_model,
                   targets,
                   ratio,
                   num_cand,
                   prompt,
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

    # initialize CLIP 
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    prompt = prompt

    with torch.no_grad():
        confidences = []
        bias_attributes = []
        final_candidates = []

        classes = np.arange(int(len(targets)/num_cand)) #define nr of classes to be used instead of targets
        print('classes', classes)
        ratios = [ratio, 0.5] # note: ratio for reference class (target'1') is always fixed
        print('ratios', ratios)
        nr_with = [int(ratios[0]*num_cand),int(ratios[1]*num_cand)] 
        nr_without = [int((1-ratios[0])*num_cand),int((1-ratios[1])*num_cand)]
        print('counter', nr_with, nr_without)

        counter_with = np.zeros(len(classes))
        counter_without = np.zeros(len(classes))
            
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
            #if horizontal_flip:
                #imgs.append(F.hflip(imgs[0])) # adds copy of horizontallly flipped images but does not work for clip -> len(imgs) changes from 1 to 2 
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
           

            bias_attr = []
            for im in imgs: 
                for i in range(len(im)):
                    im[i] = (im[i] * 0.5 + 128 / 224).clamp(0, 1) #maps from [-1,1] to [0,1]

                    # match dimensions for CLIP processor
                    perm= im[i].permute(1, 2, 0) 
                    perm_rescale = perm.mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8)

                    # reference 
                    #torchvision.utils.save_image(im[i], 'media/test/test.png')
                    #test1 = Image.open('media/test/test.png')
                    #test1_arr = np.asarray(test1) 
                    #print(np.array_equal(perm_rescale.detach().cpu().numpy(), test1_arr))

            
                    #torchvision.utils.save_image(im[i], 'media/test/temp_img.png') #save image for testing
                    #inputs = clip_processor(text=prompt, images=Image.open('media/test/test.png'), return_tensors="pt", padding=True)

                    inputs = clip_processor(text=prompt, images=perm_rescale, return_tensors="pt", padding=True)
                    
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
                    probs = torch.flatten(torch.round(logits_per_image.softmax(dim=1)))
        
                    bias_attr.append(probs[0].item()) # 1 if has attribute described by prompt with idx 0 (eg. male) - 0 if not

            bias_attributes.extend(bias_attr)
      
    
        confidences = torch.cat(confidences, dim=0)
      

        for class_idx in classes:
            # find candidate with highest confidence for each target
            sorted_conf, sorted_idx = confidences[:,
                                                  class_idx].sort(descending=True)
    
            # filter for bias attribute
            for idx in sorted_idx: 
                if (bias_attributes[sorted_idx[idx]] == 1) and (counter_with[class_idx] < nr_with[class_idx]): #& (counter_with[target] < numbers_per_target[target])): # has glasses
                    final_candidates.append(candidates[sorted_idx[0]].unsqueeze(0))
                    # TODO add confidences?
                    counter_with[class_idx]+=1
                elif (bias_attributes[sorted_idx[0]] == 0) and (counter_without[class_idx] < nr_without[class_idx]): # & (counter_without[target] < numbers_per_target[target])):
                    final_candidates.append(candidates[sorted_idx[0]].unsqueeze(0))
                    counter_without[class_idx]+=1
                
       
    
    print('cand with bias attr', counter_with)
    print('cand without bias attr', counter_without)

    final_candidates = torch.cat(final_candidates, dim=0).to(device)
    
    
    # check if enough vectors have been found
    if final_candidates.shape[0] < len(targets):
        raise ValueError('Too few initial vecotrs with bias attribute. Maximize search space or reduce num_cand')


    print(f'Found {final_candidates.shape[0]} initial style vectors.')

    if filepath:
        torch.save(final_candidates, filepath)
        print(f'Candidates have been saved to {filepath}')
    return final_candidates
