
""" This is an argparse wrapper for clip_inference.py, to edit behaviore or see code please go to functions at clip_inference.py"""
import os
import sys
import pickle 
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from pathlib import Path

from utils import save_features_and_labels


from models.MGCA.mgca.models.mgca import mgca_module
from dataset import ChexpertSmall, ChexpertSmallInspect


def load_dataset(args):

    if args.label_set == "chexpert":
        dataset  = ChexpertSmall(
            root        = args.data_dir, 
            mode        = args.split,
            return_path = True)         
    elif args.label_set == "inspect":
        dataset  = ChexpertSmallInspect(
            root        = args.data_dir, 
            mode        = args.split,
            return_path = True)

    dataset_len = len(dataset)
    return dataset,dataset_len

def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a given model')
    parser.add_argument('--model',      type=str, default="biomedclip",                help='Name of model')
    parser.add_argument('--split',      type=str, default='valid',                     help='Dataset split to evaluate (default is "validation")')
    parser.add_argument('--output_dir', type=str, default='outputs/embeddings',        help='Output directory to save evaluation results (default is "outputs")') 
    parser.add_argument('--data_dir',   type=str, default='/pasteur/data/ChexPertV01', help='Dataset path')
    parser.add_argument('--label_set',  type=str, default='inspect',                   help='labels set to use ')
    args = parser.parse_args()
    model_dict  = {}
    
    dataset, dataset_len = load_dataset(args)
    model_dict["name"]  = args.model 
    
    if args.model =="biomedclip":
        from vlms.contrastive.biomedclip import BioMedCLIP
        model_dict["model"] = BioMedCLIP()
    
    elif args.model =="clip":
        from vlms.contrastive.clip import CLIP
        model_dict["model"] = CLIP()

    elif args.model == "resnet":
       pass 
    
    elif args.model == "mgca":
        checkpoint_path = '/pasteur/u/ale9806/Repositories/chexpert/weights/vit_base.ckpt'  # Replace with the actual path to your checkpoint file
        model_dict["model"] =  mgca_module.MGCA.load_from_checkpoint(checkpoint_path)

    print(f"Loaded { model_dict['name'] }")

    output_dir = Path(args.output_dir)
    image_features_list, labels_list = [], []
    #import pdb;pdb.set_trace()
    for i in tqdm(range(dataset_len)):
       
        img, attr, _   = dataset[i]
        img            = os.path.join(args.data_dir ,img)
   
        image_features = model_dict["model"].forward_vision_only(img).to("cpu")
        image_features_list.append(image_features);labels_list.append(attr)
       
    features_array = np.array(image_features_list)
    labels_array   = np.array(labels_list)
    output_file = output_dir /  f"{args.label_set}_{args.model}_embeddings_{args.split}.pkl"
    save_features_and_labels(output_file, features_array, labels_array)
   
if __name__ == "__main__":
    main()
