#!/usr/bin/env python
# coding: utf-8
import os
import random
import sys
from tqdm import tqdm
import torch
import torch.nn.functional as F
from datasets.imagenet import ImageNet
from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from utils import *
from trainers import *

# --- IMPORTS ---
try:
    from trainers.rehark import ReHARK
except ImportError:
    pass

try:
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    try:
        from gpt_utils import get_gpt3_weights
        import gpt_lib
        ALL_TEMPLATES = gpt_lib.ALL_TEMPLATES
except ImportError as e:
    print(f"Warning: Could not import GPT tools: {e}")
    ALL_TEMPLATES = {}

def run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path, classnames=None, label_mapping=None, device='cuda:0'):
    vecs = []
    labels = []
    try:
        cache = torch.load(shots_path, map_location=device)
        vecs, labels = cache['vecs'].to(device), cache['labels'].to(device)
    except Exception as e:
        print(f"Cache miss, generating shots...")
        cache = {}
        for _ in range(cfg["augment_epoch"]):
            for image, target in tqdm(train_loader_cache):
                image, target = image.to(device), target.to(device)
                with torch.no_grad():  
                    image_features = clip_model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                vecs.append(image_features)
                labels.append(target)
        vecs = torch.cat(vecs)
        labels = torch.cat(labels)
        torch.save({'vecs':vecs.cpu(), 'labels':labels.cpu()}, shots_path)
    
    # --- ReHARK LOGIC ---
    if cfg['hp_selection'] == 'ReHARK' or 'ReHARK' in str(classifier):
        # 1. Recover Classnames if missing
        if classnames is None:
             if cfg['dataset'] == 'imagenet' and 'imagenet' in ALL_TEMPLATES:
                 classnames = list(ALL_TEMPLATES['imagenet'].keys())
             else:
                 num_classes = len(torch.unique(test_labels))
                 classnames = [str(i) for i in range(num_classes)]

        # 2. GPT Caching Mechanism (Speed Boost)
        gpt_cache_path = shots_path.replace(f"shots_s{cfg['seed']}_k{cfg['shots']}.pt", "gpt_weights.pt")
        
        try:
            if os.path.exists(gpt_cache_path):
                print("   Loading cached GPT weights...")
                gpt3_weights = torch.load(gpt_cache_path, map_location=device)
            else:
                raise FileNotFoundError
        except:
            # Generate and Save
            gpt3_weights = get_gpt3_weights(cfg['dataset'], classnames, clip_model, device)
            torch.save(gpt3_weights, gpt_cache_path)
        
        test_logits = classifier(vecs, labels, val_features, val_labels, test_features, clip_weights, gpt3_weights, cfg['dataset'], shots=cfg['shots'], seed=cfg['seed'], hp_selection=cfg['hp_selection'], backbone=cfg['backbone'])
    
    else:
        test_logits = classifier(vecs, labels, val_features, val_labels, test_features, clip_weights, cfg['dataset'], shots=cfg['shots'], seed=cfg['seed'], hp_selection=cfg['hp_selection'], backbone=cfg['backbone'])
    
    if label_mapping is not None: 
        notune_acc = cls_acc(test_logits @ label_mapping.to(test_logits.device), test_labels)  
    else: 
        notune_acc = cls_acc(test_logits, test_labels)    
    return notune_acc

def main(args):
    # Ensure ReHARK is loaded
    if args.method == 'ReHARK':
        try: from trainers.ReHARK import ReHARK
        except: pass 
            
    classifier = eval(args.method) 
    dataset = args.dataset
    hp_selection = 'ReHARK' if args.method == 'ReHARK' else args.hp_selection

    cfg = {'root_path':args.dataset_path, 'subsample_classes':'all', 'dataset':dataset, 'augment_epoch':args.augment_epoch, 'backbone':args.backbone, 'hp_selection':hp_selection, 'device':args.device}
    print("\nRunning config: ")
    print(cfg, "\n")
    backbone_names = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT-B-32': 'ViT-B/32', 'ViT-B-16': 'ViT-B/16', 'ViT-L-14': 'ViT-L/14'}
    
    test_path = os.path.join(args.test_path, args.backbone)
    if not os.path.exists(test_path): os.makedirs(test_path)
    test_path = os.path.join(args.test_path, args.backbone, cfg['dataset'])
    if not os.path.exists(test_path): os.makedirs(test_path)
        
    if os.path.exists(args.cache_dir):
        clip_model, preprocess = clip.load(backbone_names.get(cfg['backbone'], cfg['backbone']), device=args.device, download_root=args.cache_dir)
        clip_model.eval()
        clip_model = clip_model.float().to(args.device)
        for p in clip_model.parameters(): p.requires_grad = False
    else:
        clip_model, preprocess = None, None        
    
    accs = {"1": [], "2": [], "3": []}
    for seed in args.seeds:
        cfg["seed"] = seed
        random.seed(seed)
        torch.manual_seed(seed) 
        print(f"---- Seed {seed} ----")
        for shots in args.shots:
            shots_path = os.path.join(args.shots_path, args.backbone, f'augment{cfg["augment_epoch"]}')
            if not os.path.exists(shots_path): os.makedirs(shots_path)
            shots_path = os.path.join(args.shots_path, args.backbone, f'augment{cfg["augment_epoch"]}', cfg['dataset'])
            if not os.path.exists(shots_path): os.makedirs(shots_path)
            shots_path = os.path.join(args.shots_path, args.backbone,f'augment{cfg["augment_epoch"]}', cfg['dataset'])
            if not os.path.exists(shots_path): os.makedirs(shots_path)
            clip_weights_path = os.path.join(args.shots_path, args.backbone,f'augment10', cfg['dataset'], f'textweights_s1_k1.pt')
            cfg["shots"] = shots
            
            # --- DATA LOADING ---
            if cfg['dataset'] != "imagenet":
                dataset = build_dataset(cfg, cfg['dataset'], cfg['root_path'], cfg['shots']) 
                train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256, tfm=train_tranform if cfg['augment_epoch']>1 else train_tranform_clean, is_train=True, shuffle=False)
                test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False)
                test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)
                val_features, val_labels = pre_load_features(clip_model, val_loader, load_path=os.path.join(test_path, f'val_s{seed}_k{shots}.pt'), device=args.device, n_shots=-1 if args.hp_selection == 'tip-adapter' else shots)
                
                classnames, template = dataset.classnames, dataset.template
            else:
                try: 
                    if not os.path.exists(os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt')):
                        raise FileNotFoundError("No shot cache")
                    train_loader_cache, test_loader = None, None
                    test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
                    
                    dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
                    classnames, template = dataset.classnames, dataset.template
                    
                except Exception as e: 
                    dataset = ImageNet(cfg, cfg['root_path'], cfg['shots'], preprocess)
                    train_loader_cache = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=False)
                    test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=64, num_workers=8, shuffle=False)
                    test_features, test_labels = pre_load_features(clip_model, test_loader, load_path=os.path.join(test_path, f'test_s{seed}_k{shots}.pt'), device=args.device)   
                    classnames, template = dataset.classnames, dataset.template
                
                val_features, val_labels = test_features, test_labels 
                
            test_features = test_features.cpu()
            test_labels = test_labels.cpu()
            val_features = val_features.cpu()
            val_labels = val_labels.cpu()
            
            try:
                clip_weights = torch.load(clip_weights_path, map_location=args.device).to(args.device)
            except Exception as e:
                clip_weights = get_clip_weights(classnames, template, clip_model, device=args.device)   
                torch.save(clip_weights.cpu(), clip_weights_path)
            
            acc = run(classifier, cfg, train_loader_cache, test_features, test_labels, val_features, val_labels, clip_weights, clip_model, shots_path=os.path.join(shots_path, f'shots_s{seed}_k{shots}.pt'), classnames=classnames, device=args.device)
            accs[str(cfg["seed"])].append(acc)
            print(f"{shots}-shots : {acc:.2f}%")
    accuracies = []
    for seed in ["1", "2", "3"]:
        accuracies.append(accs[seed])
    accuracies = torch.tensor(accuracies)
    return accuracies

if __name__ == '__main__':
    args = get_arguments()
    print("Evaluate on dataset:", args.dataset)
    res = main(args)
    print(f'{args.method} on {args.dataset}:', {k:round(v, 2) for k,v in zip(args.shots, res.mean(dim=0).tolist())})
