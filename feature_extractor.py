
""" This is an argparse wrapper for clip_inference.py, to edit behaviore or see code please go to functions at clip_inference.py"""
import os
import sys
import pickle 
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from pathlib import Path

from dataset import ChexpertSmall
from vlms.contrastive.biomedclip import BioMedCLIP
from utils import save_features_and_labels

def main():
    parser = argparse.ArgumentParser(description='Evaluate a dataset using a given model')
    parser.add_argument('--model',      type=str, default="biomedclip",                help='Name of model')
    parser.add_argument('--split',      type=str, default='valid',                     help='Dataset split to evaluate (default is "validation")')
    parser.add_argument('--output_dir', type=str, default='outputs',                   help='Output directory to save evaluation results (default is "outputs")') 
    parser.add_argument('--data_dir',   type=str, default='/pasteur/data/ChexPertV01', help='Dataset path')
    args = parser.parse_args()
    model_dict  = {}

    model_dict["name"]  = args.model
    model_dict["model"] = BioMedCLIP()
    output_dir = Path(args.output_dir)

    dataset  = ChexpertSmall(
        root        = args.data_dir, 
        mode        = args.split,
        return_path = True)

    dataset_len = len(dataset)
    image_features_list, labels_list = [], []

    for i in tqdm(range(dataset_len)):
        img, attr, patient_id = dataset[i]
        img = os.path.join(args.data_dir ,img)
        image_features = model_dict["model"].forward_vision_only(img).to("cpu")
        image_features_list.append(image_features);labels_list.append(attr)
       
    features_array = np.array(image_features_list)
    labels_array   = np.array(labels_list)
    output_file = output_dir /  f"{args.model}_embeddings_{args.split}.pkl"
    save_features_and_labels(output_file, features_array, labels_array)
   
if __name__ == "__main__":
    main()


