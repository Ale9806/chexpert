import os
import sys
from urllib import request
import zipfile
import json
import math

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset
import pdb

def extract_patient_ids(dataset, idxs):
    # extract a list of patient_id for prediction/eval results as ['CheXpert-v1.0-small/valid/patient64541/study1', ...]
    #    extract from image path = 'CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg'
    #    NOTE -- patient_id is non-unique as there can be multiple views under the same study
    return dataset.data['Path'].loc[idxs].str.rsplit('/', expand=True, n=1)[0].values


def compute_mean_and_std(dataset):
    m = 0
    s = 0
    k = 1
    for img, _, _ in tqdm(dataset):
        x = img.mean().item()
        new_m = m + (x - m)/k
        s += (x - m)*(x - new_m)
        m = new_m
        k += 1
    print('Number of datapoints: ', k)
    return m, math.sqrt(s/(k-1))

class ChexpertSmall(Dataset):
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # folder to match the filename
    attr_all_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
                      'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
                      'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
    # select only the competition labels
    attr_names = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

    def __init__(self, root, mode='train', transform=None, data_filter=None, mini_data=None,return_path=False):
        assert mode in ['train', 'valid', 'test', 'vis']
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.mode = mode
        self.return_path = return_path
        self._maybe_download_and_extract()
        self._maybe_process(data_filter)

        data_file = os.path.join(self.root, self.dir_name, f'{mode}.pt')
        self.data:pd.DataFrame = torch.load(data_file) # Loads preprocessed CSV

        if mini_data is not None:
            # truncate data to only a subset for debugging
            self.data = self.data[:mini_data]


        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]
    
        

    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        img = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr)

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[idx]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
                                    # self.data.index(idx) pulls the index in the original dataframe and not the subset

        if self.return_path:
            img = img_path 
            
        return img, attr, idx

    def __len__(self):
        return len(self.data)

    def _maybe_download_and_extract(self):
        fpath = os.path.join(self.root, os.path.basename(self.url))
        # if data dir does not exist, download file to root and unzip into dir_name
        if not os.path.exists(os.path.join(self.root, self.dir_name)):
            # check if zip file already downloaded
            if not os.path.exists(os.path.join(self.root, os.path.basename(self.url))):
                print('Downloading ' + self.url + ' to ' + fpath)
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\r>> Downloading %s %.1f%%' % (fpath,
                        float(count * block_size) / float(total_size) * 100.0))
                    sys.stdout.flush()
                request.urlretrieve(self.url, fpath, _progress)
                print()
            print('Extracting ' + fpath)
            with zipfile.ZipFile(fpath, 'r') as z:
                z.extractall(self.root)
                if os.path.exists(os.path.join(self.root, self.dir_name, '__MACOSX')):
                    os.rmdir(os.path.join(self.root, self.dir_name, '__MACOSX'))
            os.unlink(fpath)
            print('Dataset extracted.')
            print("Test set has to be extrqacted from the orignial Chexpert Website :)")

    def _maybe_process(self, data_filter):
        # Dataset labels are: blank for unmentioned, 0 for negative, -1 for uncertain, and 1 for positive.
        # Process by:
        #    1. fill NAs (blanks for unmentioned) as 0 (negatives)
        #    2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        #    3. apply attr filters as a dictionary {data_attribute: value_to_keep} e.g. {'Frontal/Lateral': 'Frontal'}

        # check for processed .pt files
        train_file = os.path.join(self.root, self.dir_name, 'train.pt')
        valid_file = os.path.join(self.root, self.dir_name, 'valid.pt')
        test_file  = os.path.join(self.root, self.dir_name, 'test.pt')


        if not (os.path.exists(train_file) and os.path.exists(valid_file)):
            # load data and preprocess training data
            valid_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'valid.csv'), keep_default_na=True)
            train_df = self._load_and_preprocess_training_data(os.path.join(self.root, self.dir_name, 'train.csv'), data_filter)

            # save
            torch.save(train_df, train_file)
            torch.save(valid_df, valid_file)

        if not os.path.exists(test_file):
            test_df = pd.read_csv(os.path.join(self.root, self.dir_name, 'test.csv'), keep_default_na=True)
            test_df['Path'] =  'CheXpert-v1.0-small/' + test_df['Path'] # add CheXpert-v1.0-small/ to every string in Path in chexpert_labelss datafame      
            torch.save(test_df,test_file)



    def _load_and_preprocess_training_data(self, csv_path, data_filter):
        train_df = pd.read_csv(csv_path, keep_default_na=True)

        # 1. fill NAs (blanks for unmentioned) as 0 (negatives)
        # attr columns ['No Finding', ..., 'Support Devices']; note AP/PA remains with NAs for Lateral pictures
        train_df[self.attr_names] = train_df[self.attr_names].fillna(0)

        # 2. fill -1 as 1 (U-Ones method described in paper)  # TODO -- setup options for uncertain labels
        train_df[self.attr_names] = train_df[self.attr_names].replace(-1,1)

        if data_filter is not None:
            # 3. apply attr filters
            # only keep data matching the attribute e.g. df['Frontal/Lateral']=='Frontal'
            for k, v in data_filter.items():
                train_df = train_df[train_df[k]==v]

            with open(os.path.join(os.path.dirname(csv_path), 'processed_training_data_filters.json'), 'w') as f:
                json.dump(data_filter, f)

        return train_df

class ChexpertSmallInspect(Dataset):
    url = 'http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip'
    dir_name = os.path.splitext(os.path.basename(url))[0]  # folder to match the filename
    attr_all_names = ["12_month_mortality",  
                        "12_month_readmission", 
                        "1_month_readmission", 
                        "6_month_readmission",
                        "12_month_PH",         
                        "1_month_mortality",   
                        "6_month_mortality"]


    # select only the competition labels
    attr_names =["12_month_mortality",  
                        "12_month_readmission", 
                        "1_month_readmission", 
                        "6_month_readmission",
                        "12_month_PH",         
                        "1_month_mortality",   
                        "6_month_mortality"]
   

    def __init__(self, root, mode='train', transform=None, data_filter=None, mini_data=None,return_path=False):
        assert mode in ['train', 'valid', 'test']
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.mode = mode
        self.return_path = return_path
        self.data = self.merge_data(mode)

        # store index of the selected attributes in the columns of the data for faster indexing
        self.attr_idxs = [self.data.columns.tolist().index(a) for a in self.attr_names]

        #import pdb;pdb.set_trace()
    

    def merge_data(self,mode:str,how:str="inner") -> pd.DataFrame:
        inspect_labels:pd.DataFrame  = self.get_labels(mode)
        data_filepath:str            = os.path.join(self.root, self.dir_name, f'{mode}.pt')
        chexpert_labels:pd.DataFrame = torch.load(data_filepath)
        merged_df:pd.DataFrame  = pd.merge(chexpert_labels, inspect_labels, on="Path", how="inner")

        #simport pdb;pdb.set_trace()
        #merged_labels.to_csv('out.csv')  

        merged_df = merged_df.fillna(-1)
        return merged_df


    def split_path_column(self,df,attribute,chexpert:str="CheXpert-v1.0-small"):

        # Check if 'Path' column exists
        if 'Path' not in df.columns:
            raise ValueError("The DataFrame does not contain a 'Path' column.")
     
        df["value"] = df["value"].astype(int)
        df.rename(columns={'value':attribute}, inplace=True)   
       
        # Initialize new columns
        df['split'] = ''
        df['CPath'] = ''

        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            path_split = row['Path'].split('/')
            
            # Assign new values to the DataFrame
            df.at[index, 'split'] = path_split[1]
            df.at[index, 'CPath'] = f"{chexpert}/{path_split[1]}/{path_split[2]}/{path_split[3]}/{path_split[4]}"
        
        # Remove unecssary columns
        df.drop(['label_type',"Path","patient_id",'prediction_time', 'split'], axis = 1, inplace = True) 
       

    def get_labels(self,mode:str=None) -> dict:
        merged_df = None
        for attribute in self.attr_names:
            data_path = os.path.join(self.root,"CheXpert-v1.0-small","inspect_labels",f"{attribute}.csv")
            df        =   pd.read_csv(data_path, keep_default_na=True)
            self.split_path_column(df,attribute)

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on="CPath", how="outer")

        merged_df.rename(columns={"CPath":"Path"}, inplace=True)
        merged_df['Path'] = merged_df['Path'].str.replace('test-patient', 'test')
        merged_df['Path'] = merged_df['Path'].str.replace('valid-patient', 'valid')

        if mode:
            merged_df = merged_df[merged_df['Path'].str.contains(mode)]
  

        return merged_df
        

 


    def __getitem__(self, idx):
        # 1. select and load image
        img_path = self.data.iloc[idx, 0]  # 'Path' column is 0
        img = Image.open(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        # 2. select attributes as targets
        attr = self.data.iloc[idx, self.attr_idxs].values.astype(np.float32)
        attr = torch.from_numpy(attr)

        # 3. save index for extracting the patient_id in prediction/eval results as 'CheXpert-v1.0-small/valid/patient64541/study1'
        #    performed using the extract_patient_ids function
        idx = self.data.index[idx]  # idx is based on len(self.data); if we are taking a subset of the data, idx will be relative to len(subset);
                                    # self.data.index(idx) pulls the index in the original dataframe and not the subset

        if self.return_path:
            img = img_path 
            
        return img, attr, idx

    def __len__(self):
        return len(self.data)






if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Data directory.')
    args = parser.parse_args()

    ds = ChexpertSmall(root=args.data_dir, mode='train')
    print('Train dataset loaded. Length: ', len(ds))

    output_dir = 'results/test/'

    # output a few images from the validation set and display labels
    if True:
        import torchvision.transforms as T
        from torchvision.utils import save_image
        ds = ChexpertSmall(root=args.data_dir, mode='valid',
                transform=T.Compose([T.CenterCrop(320), T.ToTensor(), T.Normalize(mean=[0.5330], std=[0.0349])]))
        print('Valid dataset loaded. Length: ', len(ds))
        for i in range(10):
            img, attr, patient_id = ds[i]
            save_image(img, 'test_valid_dataset_image_{}.png'.format(i), normalize=True, scale_each=True)
            print('Patient id: {}; labels: {}'.format(patient_id, attr))

        #import pdb;pdb.set_trace()

    if False:
        ds = ChexpertSmall(root=args.data_dir, mode='train', transform=T.Compose([T.CenterCrop(320), T.ToTensor()]))
        m, s = compute_mean_and_std(ds)
        print('Dataset mean: {}; dataset std {}'.format(m, s))
        # Dataset mean: 0.533048452958796; dataset std 0.03490651403764978
