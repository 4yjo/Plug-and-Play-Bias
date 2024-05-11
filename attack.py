import argparse
import csv
import math
import random
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import wandb
from facenet_pytorch import InceptionResnetV1
from rtpt import RTPT
from torch.utils.data import TensorDataset

from attacks.final_selection import perform_final_selection
from attacks.optimize import Optimization
from datasets.custom_subset import ClassSubset, Subset
from metrics.classification_acc import ClassificationAccuracy
from metrics.fid_score import FID_Score
from metrics.prcd import PRCD
from utils.attack_config_parser import AttackConfigParser
from utils.datasets import (create_target_dataset, get_facescrub_idx_to_class,
                            get_stanford_dogs_idx_to_class)
from utils.stylegan import create_image, load_discrimator, load_generator
from utils.wandb import *

import os
import torchvision
import torchvision.transforms.functional as F

from transformers import CLIPProcessor, CLIPModel


def main():
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_devices = [i for i in range(torch.cuda.device_count())]

    # Define and parse attack arguments
    parser = create_parser()
    parser.add_argument('--wandb_target_run', type=str, default=None) # "project/run-id"
    parser.add_argument('--run_name', type=str, default=None)
    config, args = parse_arguments(parser)
    
    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load idx to class mappings
    idx_to_class = None
    if config.dataset.lower() == 'facescrub':
        idx_to_class = get_facescrub_idx_to_class()
    elif config.dataset.lower() == 'stanford_dogs':
        idx_to_class = get_stanford_dogs_idx_to_class()
    else:

        class KeyDict(dict):

            def __missing__(self, key):
                return key

        idx_to_class = KeyDict()

    # Load pre-trained StyleGan2 components
    G = load_generator(config.stylegan_model)
    D = load_discrimator(config.stylegan_model)
    num_ws = G.num_ws

    # Load target model and set dataset
    wandb_target_run = args.wandb_target_run
    api = wandb.Api(timeout=60)
    run = api.run(wandb_target_run)
    ratio = run.config['Ratio']
    attributes = run.config['Attributes']
    hidden_attributes = run.config['Hidden Attributes']
    print("information loaded from target model config with wandb run id ",wandb_target_run)
    print("Ratio:",ratio, ", Attributes:", attributes, ", Hidden Attributes:", hidden_attributes)
    target_model = config.create_target_model(wandb_target_run)
    target_model_name = target_model.name
    target_dataset = config.get_target_dataset() #note: takes ratio from wandb config of specified target model run id
    num_cand = config.candidates['num_candidates'] 
    prompt = config.prompt

    # TODO 
   # for prompt in prompts:
    #    if not isinstance(prompt, list):
     #       raise ValueError("prompts must be 2d array, e.g. [['a boy','a girl']]")


    # Distribute models
    target_model = torch.nn.DataParallel(target_model, device_ids=gpu_devices)
    target_model.name = target_model_name
    synthesis = torch.nn.DataParallel(G.synthesis, device_ids=gpu_devices)
    synthesis.num_ws = num_ws
    discriminator = torch.nn.DataParallel(D, device_ids=gpu_devices)

    synthesis.eval() 

    # Load basic attack parameters
    num_epochs = config.attack['num_epochs']
    batch_size_single = config.attack['batch_size']
    batch_size = config.attack['batch_size'] * torch.cuda.device_count()
    targets = config.create_target_vector()

    '''
    # Create initial style vectors (unbalanced)
    w, w_init, x, V = create_initial_vectors(config, G, target_model, targets,
                                             device)
    del G

    save_as_image(w_init, synthesis)
    '''
    

    # Create balanced distribution of bias attribute in latent space
    w, w_init, x, V = create_bal_initial_vectors(config, G, target_model, targets, device)
    del G

    # save vector as images for visualization
    #save_as_img(w_init, synthesis, 'media/images')

    # Initialize wandb logging
    if config.logging:
        optimizer = config.create_optimizer(params=[w])
        wandb_run = init_wandb_logging(optimizer, target_model_name, config,
                                       args)
        run_id = wandb_run.id

    # Print attack configuration
    print(
        f'Start attack against {target_model.name} optimizing w with shape {list(w.shape)} ',
        f'and targets {dict(Counter(targets.cpu().numpy()))}.')
    print(f'\nAttack parameters')
    for key in config.attack:
        print(f'\t{key}: {config.attack[key]}')
    print(
        f'Performing attack on {torch.cuda.device_count()} gpus and an effective batch size of {batch_size} images.'
    )

    # Initialize RTPT
    rtpt = None
    if args.rtpt:
        max_iterations = math.ceil(w.shape[0] / batch_size) \
            + int(math.ceil(w.shape[0] / (batch_size * 3))) \
            + 2 * int(math.ceil(config.final_selection['samples_per_target'] * len(set(targets.cpu().tolist())) / (batch_size * 3))) \
            + 2 * len(set(targets.cpu().tolist()))
        rtpt = RTPT(name_initials='AM',
                    experiment_name='Model_Inversion_Attack',
                    max_iterations=max_iterations)
        rtpt.start()

    # Load pretrained CLIP 
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Log initial vectors
    if config.logging:
        Path("results").mkdir(parents=True, exist_ok=True)
        init_w_path = f"results/init_w_{run_id}.pt"
        torch.save(w.detach(), init_w_path)
        wandb.save(init_w_path)

    # Create attack transformations
    attack_transformations = config.create_attack_transformations()

    ####################################
    #         Attack Iteration         #
    ####################################
    optimization = Optimization(target_model, synthesis, discriminator,
                                attack_transformations, num_ws, config)

    # Collect results
    w_optimized = []

    # Prepare batches for attack
    for i in range(math.ceil(w.shape[0] / batch_size)):
        w_batch = w[i * batch_size:(i + 1) * batch_size].cuda()
        targets_batch = targets[i * batch_size:(i + 1) * batch_size].cuda()
        print(
            f'\nOptimizing batch {i+1} of {math.ceil(w.shape[0] / batch_size)} targeting classes {set(targets_batch.cpu().tolist())}.'
        )

        # Run attack iteration
        torch.cuda.empty_cache()
        w_batch_optimized = optimization.optimize(w_batch, targets_batch,
                                                  num_epochs).detach().cpu()

        if rtpt:
            num_batches = math.ceil(w.shape[0] / batch_size)
            rtpt.step(subtitle=f'batch {i+1} of {num_batches}')

        # Collect optimized style vectors
        w_optimized.append(w_batch_optimized)

    # Concatenate optimized style vectors
    w_optimized_unselected = torch.cat(w_optimized, dim=0)
    torch.cuda.empty_cache()
    del discriminator

    # Log optimized vectors
    if config.logging:
        optimized_w_path = f"results/optimized_w_{run_id}.pt"
        torch.save(w_optimized_unselected.detach(), optimized_w_path)
        wandb.save(optimized_w_path)
    

    ####################################
    #          Filter Results          #
    ####################################

    # Filter results
    if config.final_selection:
        print(
            f'\nSelect final set of max. {config.final_selection["samples_per_target"]} ',
            f'images per target using {config.final_selection["approach"]} approach.'
        )
        final_w, final_targets = perform_final_selection(
            w_optimized_unselected,
            synthesis,
            config,
            targets,
            target_model,
            device=device,
            batch_size=batch_size * 10,
            **config.final_selection,
            rtpt=rtpt)
        print(f'Selected a total of {final_w.shape[0]} final images ',
              f'of target classes {set(final_targets.cpu().tolist())}.')

    else:
        final_targets, final_w = targets, w_optimized_unselected
    del target_model

    # Log selected vectors
    if config.logging:
        optimized_w_path_selected = f"results/optimized_w_selected_{run_id}.pt"
        torch.save(final_w.detach(), optimized_w_path_selected)
        wandb.save(optimized_w_path_selected)
        wandb.config.update({'w_path': optimized_w_path})

   


    ####################################
    #  Attack Evaluation for Bias      #
    ####################################

    # Count number of images in attack results that hold bias attribute 
    classes = np.arange(int(len(targets)/num_cand))  #define nr of classes to be used instead of targets
    print('classes', classes)

    counter = []
    log_imgs_class_1=[]
    log_imgs_class_2=[]
    
    # copy data to match dimensions
    if final_w.shape[1] == 1:
        final_w_expanded = torch.repeat_interleave(final_w,repeats=synthesis.num_ws,
                                                    dim=1)
    else: 
        final_w_expanded = w
    
    # create images for CLIP evaluation
    imgs = synthesis(final_w_expanded,noise_mode='const',force_fp32=True)

    # crop and resize
    imgs = F.resize(imgs, 224, antialias=True)

    for i in range(imgs.shape[0]):
        # maps from [-1,1] to [0,1]
        imgs[i] = (imgs[i] * 0.5 + 128 / 224).clamp(0, 1)

        # match dimensions for CLIP processor
        perm = imgs[i].permute(1, 2, 0) 
        perm_rescale = perm.mul(255).add_(0.5).clamp_(0, 255).type(torch.uint8)

        inputs = clip_processor(text=prompt, images=perm_rescale, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = torch.flatten(torch.round(logits_per_image.softmax(dim=1)))
            
        counter.append(probs[0].item()) # 1 if has attribute described by prompt with idx 0 (eg. male) - 0 if not

        # save first 10 images of each class to wandb
        cpu_image = perm_rescale.detach().cpu().numpy()
        if i < 10:
            wand_img = wandb.Image(cpu_image, caption=f'class 1')
            log_imgs_class_1.append(wand_img)

        if (i >= num_cand ) and (i < num_cand+10):
            wand_img = wandb.Image(cpu_image, caption=f'class 2')
            log_imgs_class_2.append(wand_img)

    wandb.log({'class 1': log_imgs_class_1})
    wandb.log({'class 2': log_imgs_class_2})
   
    # count per class
    split_idx = int(len(counter)/2)  # note: only for 2 classes
    counter_class_1 = np.sum(counter[:split_idx])/num_cand
    counter_class_2 = np.sum(counter[split_idx:])/num_cand
    
    print(f'Identified as {prompt[0]} in Class 1: {counter_class_1}')
    print(f'Identified as {prompt[0]} in Class 2: {counter_class_2}')

    # add results to wandb attack run logs
    wandb.summary.update({f'{prompt[0]} in Class 1': counter_class_1})
    wandb.summary.update({f'{prompt[0]} in Class 2': counter_class_2})
    
    wandb.config.update({'prompts': prompt})

    '''

    ####################################
    #         Attack Accuracy          #
    ####################################

    # Compute attack accuracy with evaluation model on all generated samples
    try:
        evaluation_model = config.create_evaluation_model()
        evaluation_model = torch.nn.DataParallel(evaluation_model)
        evaluation_model.to(device)
        evaluation_model.eval()
        class_acc_evaluator = ClassificationAccuracy(evaluation_model,
                                                     device=device)
  
        acc_top1, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            w_optimized_unselected,
            targets,
            synthesis,
            config,
            batch_size=batch_size * 2,
            resize=299,
            rtpt=rtpt)

        if config.logging:
            try:
                filename_precision = write_precision_list(
                    f'results/precision_list_unfiltered_{run_id}',
                    precision_list)
                wandb.save(filename_precision)
            except:
                pass
        #print(
        #    f'\nUnfiltered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
        #    f', accuracy@5={acc_top5:4f}, correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
        #)
        print(
           f'\nUnfiltered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}',
           f', correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
        )

        # Compute attack accuracy on filtered samples
        if config.final_selection:
            #acc_top1, acc_top5, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
            acc_top1, predictions, avg_correct_conf, avg_total_conf, target_confidences, maximum_confidences, precision_list = class_acc_evaluator.compute_acc(
                final_w,
                final_targets,
                synthesis,
                config,
                batch_size=batch_size * 2,
                resize=299,
                rtpt=rtpt)
            if config.logging:
                filename_precision = write_precision_list(
                    f'results/precision_list_filtered_{run_id}',
                    precision_list)
                wandb.save(filename_precision)

            print(
                f'Filtered Evaluation of {final_w.shape[0]} images on Inception-v3: \taccuracy@1={acc_top1:4f}, ',
                f'correct_confidence={avg_correct_conf:4f}, total_confidence={avg_total_conf:4f}'
            )
            
        del evaluation_model

    except Exception:
        print(traceback.format_exc())

    ####################################
    #    FID Score and GAN Metrics     #
    ####################################

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None
    try:
        # set transformations
        crop_size = config.attack_center_crop
        target_transform = T.Compose([
            T.ToTensor(),
            T.Resize((299, 299), antialias=True),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # create datasets
        attack_dataset = TensorDataset(final_w, final_targets)
        attack_dataset.targets = final_targets
        training_dataset = create_target_dataset(target_dataset,
                                             target_transform, attributes, hidden_attributes, ratio)
       
    
        training_dataset = ClassSubset(
            training_dataset,
            target_classes=torch.unique(final_targets).cpu().tolist())

        # compute FID score
        fid_evaluation = FID_Score(training_dataset,
                                   attack_dataset,
                                   device=device,
                                   crop_size=crop_size,
                                   generator=synthesis,
                                   batch_size=batch_size * 3,
                                   dims=2048,
                                   num_workers=8,
                                   gpu_devices=gpu_devices)
        fid_score = fid_evaluation.compute_fid(rtpt)
        print(
            f'FID score computed on {final_w.shape[0]} attack samples and {config.dataset}: {fid_score:.4f}'
        )
    
        # compute precision, recall, density, coverage
        prdc = PRCD(training_dataset,
                    attack_dataset,
                    device=device,
                    crop_size=crop_size,
                    generator=synthesis,
                    batch_size=batch_size * 3,
                    dims=2048,
                    num_workers=8,
                    gpu_devices=gpu_devices)
        precision, recall, density, coverage = prdc.compute_metric(
            num_classes=config.num_classes, k=1, rtpt=rtpt)  #TODO put k=3 if more than 2 classes
        print(
            f' Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}'
        )

    except Exception:
        print(traceback.format_exc())

    ####################################
    #         Feature Distance         #
    ####################################
    avg_dist_inception = None
    avg_dist_facenet = None
    try:
        # Load Inception-v3 evaluation model and remove final layer
        evaluation_model_dist = config.create_evaluation_model()
        evaluation_model_dist.model.fc = torch.nn.Sequential()
        evaluation_model_dist = torch.nn.DataParallel(evaluation_model_dist,
                                                      device_ids=gpu_devices)
        evaluation_model_dist.to(device)
        evaluation_model_dist.eval()

        # Compute average feature distance on Inception-v3
        evaluate_inception = DistanceEvaluation(evaluation_model_dist,
                                                synthesis, 299,
                                                config.attack_center_crop,
                                                target_dataset, config.seed, 
                                                attributes, hidden_attributes, ratio)
        avg_dist_inception, mean_distances_list = evaluate_inception.compute_dist(
            final_w,
            final_targets,
            batch_size=batch_size_single * 5,
            rtpt=rtpt)

        if config.logging:
            try:
                filename_distance = write_precision_list(
                    f'results/distance_inceptionv3_list_filtered_{run_id}',
                    mean_distances_list)
                wandb.save(filename_distance)
            except:
                pass

        print('Mean Distance on Inception-v3: ',
              avg_dist_inception.cpu().item())
        # Compute feature distance only for facial images
        if target_dataset in [
                'facescrub', 'celeba_identities', 'celeba_attributes'
        ]:
            # Load FaceNet model for face recognition
            facenet = InceptionResnetV1(pretrained='vggface2')
            facenet = torch.nn.DataParallel(facenet, device_ids=gpu_devices)
            facenet.to(device)
            facenet.eval()

            # Compute average feature distance on facenet
            evaluater_facenet = DistanceEvaluation(facenet, synthesis, 160,
                                                   config.attack_center_crop,
                                                   target_dataset, config.seed,
                                                   attributes, hidden_attributes, ratio)
            avg_dist_facenet, mean_distances_list = evaluater_facenet.compute_dist(
                final_w,
                final_targets,
                batch_size=batch_size_single * 8,
                rtpt=rtpt)
            if config.logging:
                filename_distance = write_precision_list(
                    f'results/distance_facenet_list_filtered_{run_id}',
                    mean_distances_list)
                wandb.save(filename_distance)

            print('Mean Distance on FaceNet: ', avg_dist_facenet.cpu().item())
    except Exception:
        print(traceback.format_exc())
    '''
    


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
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
        
        (
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


def create_initial_vectors(config, G, target_model, targets, device):
    with torch.no_grad():
        w = config.create_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)
        w_init = deepcopy(w)
        x = None
        V = None
    return w, w_init, x, V

def create_bal_initial_vectors(config, G, target_model, targets, device):
    with torch.no_grad():
        w = config.create_bal_candidates(G, target_model, targets).cpu()
        if config.attack['single_w']:
            w = w[:, 0].unsqueeze(1)
        w_init = deepcopy(w)
        x = None
        V = None
    return w, w_init, x, V


def write_precision_list(filename, precision_list):
    filename = f"{filename}.csv"
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        for row in precision_list:
            wr.writerow(row)
    return filename


def log_attack_progress(loss,
                        target_loss,
                        discriminator_loss,
                        discriminator_weight,
                        mean_conf,
                        lr,
                        imgs=None,
                        captions=None):
    if imgs is not None:
        imgs = [
            wandb.Image(img.permute(1, 2, 0).numpy(), caption=caption)
            for img, caption in zip(imgs, captions)
        ]
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr,
            'samples': imgs
        })
    else:
        wandb.log({
            'total_loss': loss,
            'target_loss': target_loss,
            'discriminator_loss': discriminator_loss,
            'discriminator_weight': discriminator_weight,
            'mean_conf': mean_conf,
            'learning_rate': lr
        })

def save_as_img(vector, synthesis, outdir=None):
    # make local directory to store generated images
    if outdir is not None:
        os.makedirs(outdir)
    else: 
        os.makedirs('media/images/results')

    # copy data to match dimensions
    if vector.shape[1] == 1:
        vector_expanded = torch.repeat_interleave(vector,
                                    repeats=synthesis.num_ws,
                                    dim=1)
    else: 
        vector_expanded = vector

    # create images from init vectors for testing
    print(vector_expanded.shape)
    x = synthesis(vector_expanded, noise_mode='const', force_fp32=True)

    print(x.shape)
    # crop and resize
    x = F.resize(x, 224, antialias=True)
    x = (x * 0.5 + 128 / 224).clamp(0, 1) #maps from [-1,1] to [0,1]
    print(x.shape)
        
    for i in range(x.shape[0]):
        torchvision.utils.save_image(x[i], f'{outdir}/{i}.png') 
    print('images saved to ', str(outdir))


def init_wandb_logging(optimizer, target_model_name, config, args):
    lr = optimizer.param_groups[0]['lr']
    optimizer_name = type(optimizer).__name__
    if not 'name' in config.wandb['wandb_init_args']:
        config.wandb['wandb_init_args'][
            'name'] = f'{optimizer_name}_{lr}_{target_model_name}'
    wandb_config = config.create_wandb_config()
    run = wandb.init(config=wandb_config, **config.wandb['wandb_init_args'])
    wandb.config.update(args) # adds all of the arguments as config variables to attack loggin on wandb

    wandb.save(args.config)
    return run


def intermediate_wandb_logging(optimizer, targets, confidences, loss,
                               target_loss, discriminator_loss,
                               discriminator_weight, mean_conf, imgs, idx2cls):
    lr = optimizer.param_groups[0]['lr']
    target_classes = [idx2cls[idx.item()] for idx in targets.cpu()]
    conf_list = [conf.item() for conf in confidences]
    if imgs is not None:
        img_captions = [
            f'{target} ({conf:.4f})'
            for target, conf in zip(target_classes, conf_list)
        ]
        log_attack_progress(loss,
                            target_loss,
                            discriminator_loss,
                            discriminator_weight,
                            mean_conf,
                            lr,
                            imgs,
                            captions=img_captions)
    else:
        log_attack_progress(loss, target_loss, discriminator_loss,
                            discriminator_weight, mean_conf, lr)


def log_nearest_neighbors(imgs, targets, eval_model, model_name, dataset,
                          img_size, seed, attributes, hidden_attributes, ratio):
    # Find closest training samples to final results
    evaluater = DistanceEvaluation(eval_model, None, img_size, None, dataset,
                                   seed, attributes, hidden_attributes, ratio) 
    closest_samples, distances = evaluater.find_closest_training_sample(
        imgs, targets)
    closest_samples = [
        wandb.Image(img.permute(1, 2, 0).cpu().numpy(),
                    caption=f'distance={d:.4f}')
        for img, d in zip(closest_samples, distances)
    ]
    wandb.log({f'closest_samples {model_name}': closest_samples})


def log_final_images(imgs, targets,predictions, max_confidences, target_confidences,
                     idx2cls):
   
    wand_imgs = [
        wandb.Image(
            img.permute(1, 2, 0).numpy(),
            caption=
            f'class: {targets}'
            #f'pred={idx2cls[pred.item()]} ({max_conf:.2f}), target_conf={target_conf:.2f}'
        ) for img, target, pred, max_conf, target_conf in zip(
            imgs.cpu(), targets, predictions, max_confidences, target_confidences)
    ]
    print('wandb imgs', wand_imgs)
    wandb.log({'final_images': wand_imgs})




def final_wandb_logging(avg_correct_conf, avg_total_conf, acc_top1, 
                        avg_dist_facenet, avg_dist_eval, fid_score, precision,
                        recall, density, coverage):
    wandb.save('attacks/gradient_based.py')
    wandb.run.summary['correct_avg_conf'] = avg_correct_conf
    wandb.run.summary['total_avg_conf'] = avg_total_conf
    wandb.run.summary['evaluation_acc@1'] = acc_top1
    #wandb.run.summary['evaluation_acc@5'] = acc_top5
    wandb.run.summary['avg_dist_facenet'] = avg_dist_facenet
    wandb.run.summary['avg_dist_evaluation'] = avg_dist_eval
    wandb.run.summary['fid_score'] = fid_score
    wandb.run.summary['precision'] = precision
    wandb.run.summary['recall'] = recall
    wandb.run.summary['density'] = density
    wandb.run.summary['coverage'] = coverage

    wandb.finish()


if __name__ == '__main__':
    main()
