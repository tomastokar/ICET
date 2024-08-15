import re
import numpy as np
import pandas as pd

from typing import Union
from dataclasses import dataclass
from datasets import load_dataset, Dataset, Value # HugginFace interface

from sklearn.datasets import make_blobs
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from torchvision.transforms import Compose, ToTensor, Resize


@dataclass
class DataCard:
    
    # Variables
    img_vars: list
    txt_vars: list
    cat_vars: list
    num_vars: list
        
    # Reponse
    response: str = None
        
    # Num. of levels of categoricals
    cat_levels: Union[list, int] = None
    
    # Resolution
    resolution: int = 32
    
    def __post_init__(self):
        super(DataCard, self).__init__()
        
        # All variables
        self.all_vars = self.img_vars + self.txt_vars + self.cat_vars + self.num_vars
                
        # Features
        self.features = [v for v in self.all_vars if v != self.response]
                
        # Cat levels
        if isinstance(self.cat_levels, int):
            self.cat_levels = {
                v : self.cat_levels for v in self.cat_vars
            }
            
        else:
            assert len(self.cat_levels) == len(self.cat_vars)
            self.cat_levels = {
                self.cat_vars[i]:l for i, l in enumerate(self.cat_levels)
            }

# ---------------
# Aux. functions
# ---------------
def image_transform(examples, transform):
    examples["image"] = [transform(img.convert("RGB")) for img in examples["image"]]
    return examples


def clean_string(s, max_length = 200):
    s = ' ' if s is None else s
    s = re.sub('\xa0', ' ', s)
    s = re.sub(r'[^a-zA-Z0-9]+',' ', s)
    s = ' '.join(s.split())
    s = s.split()
    s = s[:max_length]
    s = ' '.join(s)
    return s


def process_text(example, vars):
    for var in vars:
        example[var] = clean_string(example[var])
    return example


def split_dataset(dataset, test_size = 0.1, valid_size = 0.1):
    n = len(dataset)
    indices = np.arange(n)
    
    train, test = train_test_split(indices, test_size=test_size)
    train, validation = train_test_split(train, test_size=valid_size)
    
    datasets = {}
    datasets['train'] = dataset.select(i for i in train)
    datasets['validation'] = dataset.select(i for i in validation)
    datasets['test'] = dataset.select(i for i in test)
    
    return datasets


# --------------------
# XOR blobs synthetic
# --------------------
def make_xor_blobs(n):
    centers = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    XY, c = make_blobs(
        n_samples = n, 
        cluster_std = 0.25, 
        centers = centers
    )    
    c = c % 2
    XY = MinMaxScaler().fit_transform(XY)
    x = XY[:,0]
    y = XY[:,1]
    return x, y, c

def load_xor_blobs():
    
    # Split samples
    splits = {
        'train' : 1000, 
        'validation' : 100, 
        'test' : 100
    }
    
    # Generate data
    datasets = {}
    for split, n in splits.items():
        x, y, c = make_xor_blobs(n)
        datasets[split] = Dataset.from_dict(
            {
                'x' : x,
                'y' : y,
                'c' : c
            }
        )
    
    return datasets 


# ------------------------
# Purely tabular datasets
# ------------------------

def load_compas():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_cat_compas-two-years'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Split dataset
    datasets = split_dataset(dataset)
            
    return datasets


def load_electricity():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_cat_electricity'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Create maps for relabeling
    label_map = {l : i for i, l in enumerate(set(dataset['class']))}
        
    # Relabeling class
    def relabel(example):        
        example['class'] = label_map[example['class']]
        return example
    
    # Relabel categorical        
    dataset = dataset.map(lambda x : relabel(x))

    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets


def load_eye_movements():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_cat_eye_movements'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets


def load_telescope():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_MagicTelescope'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Create maps for relabeling
    label_map = {l : i for i, l in enumerate(set(dataset['class']))}
        
    # Relabeling class
    def relabel(example):        
        example['class'] = label_map[example['class']]
        return example
    
    # Relabel categorical        
    dataset = dataset.map(lambda x : relabel(x))
    
    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets


def load_diabetes():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_Diabetes130US'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets


def load_credit():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_credit'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets


def load_california():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_california'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets


def load_heloc():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_heloc'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
                        
    return datasets

def load_diamonds():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_diamonds'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_diamonds():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_diamonds'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_house_sales():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_house_sales'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Cat vars
    cat_vars = ['waterfront']
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(dataset[var]))}
        
    # Relabeling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
                example[v] = lmap[example[v]]
             
        return example
    
    # Relabel categorical
    dataset = dataset.map(lambda x : relabel(x, label_maps))


    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_bike_sharing():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_Bike_Sharing_Demand'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_road_safety():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_cat_road-safety'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Log-scale engine capacity
    col = 'Engine_Capacity_(CC)'
    new = np.log10(dataset[col])
    dataset = (
        dataset
        .remove_columns(col)
        .add_column(col, new)
        .cast(dataset.features)
    )
    
    # Rescale OSRG
    cols = ['Location_Easting_OSGR', 'Location_Northing_OSGR']
    for col in cols:
        x = np.array(dataset[col]).reshape(-1, 1)
        y = MinMaxScaler().fit_transform(x).flatten().tolist()
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, y)
            .cast(dataset.features)
        )

    # Cat vars
    cat_vars = [
        'Vehicle_Reference_df_res', 'Vehicle_Type', 'Vehicle_Manoeuvre', 
        'Vehicle_Location-Restricted_Lane', 'Hit_Object_in_Carriageway', 
        'Hit_Object_off_Carriageway', 'Was_Vehicle_Left_Hand_Drive?', 
        'Propulsion_Code', 'Local_Authority_(District)', 'Urban_or_Rural_Area', 
        'Vehicle_Reference_df', 'Casualty_Reference', 'Sex_of_Casualty', 
        'Pedestrian_Location', 'Pedestrian_Movement', 'Casualty_Type', 
        'SexofDriver'
    ]  
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(dataset[var]))}
        
    # Relabeling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
                example[v] = lmap[example[v]]
             
        return example
    
    # Relabel categorical
    dataset = dataset.map(lambda x : relabel(x, label_maps))

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_brazilian_houses():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_Brazilian_houses'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Log-scale num vars
    num_vars = ['hoa_(BRL)', 'rent_amount_(BRL)', 'property_tax_(BRL)', 'fire_insurance_(BRL)', 'total_(BRL)']
    features = dataset.features
    for var in num_vars:        
        # Recast to float
        features[var]  = Value('float32')        
        # Log-scale engine capacity
        new = np.log1p(dataset[var])
        dataset = (
            dataset
            .remove_columns(var)
            .add_column(var, new)
            .cast(features)
        )
    
    # Cat vars
    cat_vars = ['area']  
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(dataset[var]))}
        
    # Relabeling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
                example[v] = lmap[example[v]]
             
        return example
    
    # Relabel categorical
    dataset = dataset.map(lambda x : relabel(x, label_maps))
        
    # Split dataset
    datasets = split_dataset(dataset)
        
    return datasets


def load_abalone():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_abalone'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets

def load_supreme():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_analcatdata_supreme'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_seattle_crime():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_seattlecrime6'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_superconduct():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_superconduct'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_wine_quality():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_wine_quality'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Rename columns    
    dataset = dataset.rename_column("fixed.acidity", "fixed_acidity")
    dataset = dataset.rename_column("volatile.acidity", "volatile_acidity")
    dataset = dataset.rename_column("citric.acid", "citric_acid")
    dataset = dataset.rename_column("residual.sugar", "residual_sugar")
    dataset = dataset.rename_column("free.sulfur.dioxide", "free_sulfur_dioxide")    
    dataset = dataset.rename_column("total.sulfur.dioxide", "total_sulfur_dioxide")    

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_medical_charges():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_medical_charges'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Log-scale num vars
    num_vars = ['Average_Covered_Charges', 'Average_Medicare_Payments']
    features = dataset.features
    for var in num_vars:        
        # Recast to float
        features[var]  = Value('float32')        
        # Log-scale engine capacity
        new = np.log10(dataset[var])
        dataset = (
            dataset
            .remove_columns(var)
            .add_column(var, new)
            .cast(features)
        )
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_jannis():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_jannis'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets


def load_defaults():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_cat_default-of-credit-card-clients'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Log scale 
    cols = ['x1', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23']
    features = dataset.features
    for col in cols:        
        # Recast to float
        features[col]  = Value('float32')        
        # Log-scale engine capacity
        new = np.log1p(dataset[col])
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, new)
            .cast(features)
        )
    
    # Min-max scale
    cols = ['x12', 'x13', 'x14', 'x15', 'x16', 'x17']
    features = dataset.features
    for col in cols:
        # Recast to float
        features[col]  = Value('float32')           
        x = np.array(dataset[col]).reshape(-1, 1)
        y = MinMaxScaler().fit_transform(x).flatten().tolist()
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, y)
            .cast(features)
        )    
        
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets
    
    
def load_bioresponse():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_Bioresponse'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_allstate():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_Allstate_Claims_Severity'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets        


def load_mercedes():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_Mercedes_Benz_Greener_Manufacturing'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets        


def load_sgemm():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_SGEMM_GPU_kernel_performance'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets   


def load_ailerons():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_Ailerons'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets  


def load_topo():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_topo_2_1'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_miami_housing():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_MiamiHousing2016'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_cpu():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_cpu_act'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Log scale 
    cols = [
        'lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 
        'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pgin', 'ppgin', 
        'pflt', 'vflt', 'runqsz', 'freemem', 'freeswap'
    ]    
    features = dataset.features
    for col in cols:        
        # Recast to float
        features[col]  = Value('float32')        
        # Log-scale engine capacity
        new = np.log1p(dataset[col])
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, new)
            .cast(features)
        )
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_elevators():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_elevators'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_house():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_house_16H'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Log scale 
    cols = ['P1']    
    features = dataset.features
    for col in cols:        
        # Recast to float
        features[col]  = Value('float32')        
        # Log-scale engine capacity
        new = np.log1p(dataset[col])
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, new)
            .cast(features)
        )
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_houses():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_houses'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Cols to log-transform
    cols =  ['total_rooms', 'total_bedrooms', 'population', 'households']
    features = dataset.features
    for col in cols:        
        # Recast to float
        features[col]  = Value('float32')        
        # Log-scale engine capacity
        new = np.log10(dataset[col])
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, new)
            .cast(features)
        ) 
           
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_pol():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_pol'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets          


def load_sulfur():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_sulfur'
        
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets  


def load_miniboone():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_MiniBooNE'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Cat vars
    cat_vars = ['signal']  
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(dataset[var]))}
        
    # Relabeling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
                example[v] = lmap[example[v]]
             
        return example
    
    # Relabel categorical
    dataset = dataset.map(lambda x : relabel(x, label_maps))
    
    # Rescale ParticleID_19
    cols = ['ParticleID_19']
    features = dataset.features
    for col in cols:
        # Recast to float
        features[col]  = Value('float32')           
        x = np.array(dataset[col]).reshape(-1, 1)
        y = MinMaxScaler().fit_transform(x).flatten().tolist()
        dataset = (
            dataset
            .remove_columns(col)
            .add_column(col, y)
            .cast(features)
        )        
    
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_bank_marketing():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_bank-marketing'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Cat vars
    cat_vars = ['Class']  
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(dataset[var]))}
        
    # Relabeling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
                example[v] = lmap[example[v]]
             
        return example
    
    # Relabel categorical
    dataset = dataset.map(lambda x : relabel(x, label_maps))

         
    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_yprop():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_num_yprop_4_1'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_ukair():
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_particulate-matter-ukair-2017'
            
    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
    
    # Rename columns    
    dataset = dataset.rename_column("Environment.Type", "Environment_Type")
    dataset = dataset.rename_column("Altitude..m.", "Altitude_m")
    dataset = dataset.rename_column("PM.sub.2.5..sub..particulate.matter..Hourly.measured.", "PM_sub_2_5_sub_particulate_matter_Hourly_measured")
    dataset = dataset.rename_column("PM.sub.10..sub..particulate.matter..Hourly.measured.", "PM_sub_10_sub_particulate_matter_Hourly_measured")

    # Split dataset
    datasets = split_dataset(dataset)
    
    return datasets 


def load_airlines():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_Airlines_DepDelay_1M'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Sanitize column names
    for n in dataset.column_names:
        if '.' in n:    
            nn = n.replace('.', '_')
            dataset = dataset.rename_column(n, nn)
    
    # Select features
    features = dataset.features
    
    # Relabel categorical variables
    cat_vars = ['Month', 'DayofMonth']
        
    # Select categoricals
    data_cat = np.array([dataset[v] for v in cat_vars])     
        
    # Encode   
    data_cat = OrdinalEncoder().fit_transform(data_cat.T)
        
    # Replace columns                     
    for i, v in enumerate(cat_vars):
        features[v]  = Value('int64')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_cat[:,i])
        )
                            
    # Rescale numerical variables   
    num_vars = ['CRSDepTime', 'CRSArrTime', 'Distance', 'DepDelay']     
                
    # Select numericals
    data_num = np.array([dataset[v] for v in num_vars])
                                    
    # Min-max scale
    data_num = MinMaxScaler().fit_transform(data_num.T)
           
    # Replace columns             
    for i, v in enumerate(num_vars):
        features[v]  = Value('float32')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_num[:,i])
        )
    
    # Recast features
    dataset = dataset.cast(features)    
    
    # Split
    datasets = split_dataset(dataset)
    
    return datasets


def load_albert():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_cat_albert'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Sanitize column names
    for n in dataset.column_names:
        if '.' in n:    
            nn = n.replace('.', '_')
            dataset = dataset.rename_column(n, nn)
    
    # Select features
    features = dataset.features
    
    # Relabel categorical variables
    cat_vars = [
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V13', 
        'V19', 'V22', 'V30', 'V33', 'V35', 'V36', 'V40', 'V41', 'V42', 'V43', 
        'V45', 'V47', 'V50', 'V51', 'V52', 'V59', 'V63', 'V72', 'V75', 'class'
    ]
        
    # Select categoricals
    data_cat = np.array([dataset[v] for v in cat_vars])     
        
    # Encode   
    data_cat = OrdinalEncoder().fit_transform(data_cat.T)
        
    # Replace columns                     
    for i, v in enumerate(cat_vars):
        features[v]  = Value('int64')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_cat[:,i])
        )
                                
    # Recast features
    dataset = dataset.cast(features)    
    
    # Split
    datasets = split_dataset(dataset)
    
    return datasets


def load_covertype():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_covertype'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Sanitize column names
    for n in dataset.column_names:
        if '.' in n:    
            nn = n.replace('.', '_')
            dataset = dataset.rename_column(n, nn)
    
    # Select features
    features = dataset.features
    
    # Relabel categorical variables
    cat_vars = ['Y']
        
    # Select categoricals
    data_cat = np.array([dataset[v] for v in cat_vars])     
        
    # Encode   
    data_cat = OrdinalEncoder().fit_transform(data_cat.T)
        
    # Replace columns                     
    for i, v in enumerate(cat_vars):
        features[v]  = Value('int64')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_cat[:,i])
        )
                            
    # Rescale numerical variables   
    num_vars = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
                
    # Select numericals
    data_num = np.array([dataset[v] for v in num_vars])
                                    
    # Min-max scale
    data_num = MinMaxScaler().fit_transform(data_num.T)
           
    # Replace columns             
    for i, v in enumerate(num_vars):
        features[v]  = Value('float32')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_num[:,i])
        )
    
    # Recast features
    dataset = dataset.cast(features)    
    
    # Split
    datasets = split_dataset(dataset)
    
    return datasets



def load_higgs():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'clf_num_Higgs'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Sanitize column names
    for n in dataset.column_names:
        if '.' in n:    
            nn = n.replace('.', '_')
            dataset = dataset.rename_column(n, nn)
    
    # Select features
    features = dataset.features
    
    # Relabel categorical variables
    cat_vars = ['target']
        
    # Select categoricals
    data_cat = np.array([dataset[v] for v in cat_vars])     
        
    # Encode   
    data_cat = OrdinalEncoder().fit_transform(data_cat.T)
        
    # Replace columns                     
    for i, v in enumerate(cat_vars):
        features[v]  = Value('int64')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_cat[:,i])
        )
                            
    # Rescale numerical variables   
    num_vars = [
        'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude', 
        'missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_2_pt', 
        'jet_2_eta', 'jet_2_phi', 'jet_3_pt', 'jet_3_eta', 'jet_3_phi', 
        'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'm_jj', 'm_jjj', 'm_lv', 
        'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
    ]    
                
    # Select numericals
    data_num = np.array([dataset[v] for v in num_vars])
                                    
    # Min-max scale
    data_num = MinMaxScaler().fit_transform(data_num.T)
           
    # Replace columns             
    for i, v in enumerate(num_vars):
        features[v]  = Value('float32')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_num[:,i])
        )
    
    # Recast features
    dataset = dataset.cast(features)    
    
    # Split
    datasets = split_dataset(dataset)
    
    return datasets


def load_nyc_taxi():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_nyc-taxi-green-dec-2016'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Sanitize column names
    for n in dataset.column_names:
        if '.' in n:    
            nn = n.replace('.', '_')
            dataset = dataset.rename_column(n, nn)
    
    # Select features
    features = dataset.features
    
    # Relabel categorical variables
    cat_vars = [
        'VendorID', 'store_and_fwd_flag', 'RatecodeID', 'extra', 'mta_tax', 
        'improvement_surcharge', 'trip_type', 'lpep_pickup_datetime_day', 
        'lpep_pickup_datetime_hour', 'lpep_pickup_datetime_minute', 
        'lpep_dropoff_datetime_day', 'lpep_dropoff_datetime_hour', 
        'lpep_dropoff_datetime_minute'
    ]
        
    # Select categoricals
    data_cat = np.array([dataset[v] for v in cat_vars])     
        
    # Encode   
    data_cat = OrdinalEncoder().fit_transform(data_cat.T)
        
    # Replace columns                     
    for i, v in enumerate(cat_vars):
        features[v]  = Value('int64')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_cat[:,i])
        )
                            
    # Rescale numerical variables   
    num_vars = ['passenger_count', 'tolls_amount', 'total_amount', 'tip_amount']
                
    # Select numericals
    data_num = np.array([dataset[v] for v in num_vars])
                                    
    # Min-max scale
    data_num = MinMaxScaler().fit_transform(data_num.T)
           
    # Replace columns             
    for i, v in enumerate(num_vars):
        features[v]  = Value('float32')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_num[:,i])
        )
    
    # Recast features
    dataset = dataset.cast(features)    
    
    # Split
    datasets = split_dataset(dataset)
    
    return datasets


def load_soil():
    
    # Dataset name
    name = 'inria-soda/tabular-benchmark'
    
    # Subset name
    subset = 'reg_cat_visualizing_soil'

    # Fetch dataset
    dataset = load_dataset(name, subset, split = 'train')
        
    # Sanitize column names
    for n in dataset.column_names:
        if '.' in n:    
            nn = n.replace('.', '_')
            dataset = dataset.rename_column(n, nn)
    
    # Select features
    features = dataset.features
    
    # Relabel categorical variables
    cat_vars = ['isns']
        
    # Select categoricals
    data_cat = np.array([dataset[v] for v in cat_vars])     
        
    # Encode   
    data_cat = OrdinalEncoder().fit_transform(data_cat.T)
        
    # Replace columns                     
    for i, v in enumerate(cat_vars):
        features[v]  = Value('int64')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_cat[:,i])
        )
                            
    # Rescale numerical variables   
    num_vars = ['northing', 'easting', 'resistivity', 'track']
                
    # Select numericals
    data_num = np.array([dataset[v] for v in num_vars])
                                    
    # Min-max scale
    data_num = MinMaxScaler().fit_transform(data_num.T)
           
    # Replace columns             
    for i, v in enumerate(num_vars):
        features[v]  = Value('float32')  
        dataset = (
            dataset
            .remove_columns(v)
            .add_column(v, data_num[:,i])
        )
    
    # Recast features
    dataset = dataset.cast(features)    
    
    # Split
    datasets = split_dataset(dataset)
    
    return datasets


# -----------------------
# Image-tabular datasets
# -----------------------

def load_streetview() -> dict:

    # HF dataset name
    name = 'stochastic/random_streetview_images_pano_v0.0.2'

    # Fetch datasets
    dataset = load_dataset(name, split = 'train')

    # Select numeric variables
    num_vars = ['latitude', 'longitude']

    # Set numericals to float
    features = dataset.features.copy()
    for var in num_vars:
        features[var]  = Value('float32')        
    dataset = dataset.cast(features)
        
    # Rescale numericals
    for var in num_vars:
        x = np.array(dataset[var]).reshape(-1, 1)
        y = MinMaxScaler().fit_transform(x).flatten().tolist()
        dataset.remove_columns(var).add_column(var, y).cast(dataset.features)

    # Create map for relabeling
    label_map = {l : i for i, l in enumerate(set(dataset['country_iso_alpha2']))}

    # Relabiling function
    def relabel(example):
        example['country_iso_alpha2'] = label_map[example['country_iso_alpha2']]
        return example

    # Relabel categorical    
    datasets = dataset.map(relabel)

    # Img transformation sequences
    transform = Compose([ToTensor(), Resize([64,64], antialias=True)])

    # Add image transformation    
    dataset = dataset.with_transform(lambda x: image_transform(x, transform))

    # Add split names
    datasets = split_dataset(dataset)

    return datasets


def load_skin_cancer() -> dict:

    # HF dataset name
    name = 'marmal88/skin_cancer'

    # Splits to use
    split = ['train', 'validation', 'test']

    # Fetch datasets
    datasets = load_dataset(name, split = split)

    # Add split names
    datasets = dict(zip(split, datasets))

    # Remove entries with missing age
    for split, dataset in datasets.items():
        datasets[split] = dataset.select(
            i for i, x in enumerate(dataset['age'])
            if x is not None
        )
                
    # Cat vars
    cat_vars = ['dx', 'dx_type', 'sex', 'localization']
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(datasets['train'][var]))}
        
    # Relabiling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
            example[v] = lmap[example[v]]
        return example
    
    # Relabel categorical
    for split, dataset in datasets.items():
        datasets[split] = dataset.map(lambda x : relabel(x, label_maps))

    # Img transformation sequences
    transform = Compose([ToTensor(), Resize([128,128], antialias=True)])

    # Add image transformation
    for split, dataset in datasets.items():
        datasets[split] = dataset.with_transform(lambda x: image_transform(x, transform))

    return datasets


# ----------------------
# Text-tabular datasets
# ----------------------

def load_kickstarter():
    
    # Dataset name
    name = 'james-burton/kick_starter_funding'
        
    # Splits to use
    split = ['train', 'validation', 'test']
    
    # Fetch datasets
    datasets = load_dataset(name, split = split)
    
    # Add split names
    datasets = dict(zip(split, datasets))
    
    # Numeric vars
    num_vars = ['goal', 'deadline', 'created_at']
    
    # Recast numericals to floats
    for split, dataset in datasets.items():
        features = dataset.features.copy()
        for var in num_vars:
            features[var]  = Value('float64')        
        datasets[split] = dataset.cast(features)    
            
    # Rescale numerical variables    
    for var in num_vars:
        
        # Init scaler        
        scaler = MinMaxScaler()
                
        for split in ['train', 'validation', 'test']:
            # Select dataset
            dataset = datasets[split]
                                                
            # Rescale 
            x = np.array(dataset[var]).reshape(-1, 1)
            
            # Convert to log if applicable
            if var == 'goal':
                x = np.log10(x)
            
            if split == 'train':
                y = scaler.fit_transform(x)
            else:
                y = scaler.transform(x)
                
            # Convert to list
            y = y.flatten().tolist()
            
            # Add to dataset
            datasets[split] = (
                dataset
                .remove_columns(var)
                .add_column(var, y)
                .cast(dataset.features)
            )
        
    # Text vars
    txt_vars = ['name', 'desc', 'keywords']
    for split, dataset in datasets.items():
        datasets[split] = dataset.map(lambda x: process_text(x, txt_vars))
        
    # Cat vars
    cat_vars = ['disable_communication', 'country', 'currency']
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(datasets['train'][var]))}
        
    # Relabiling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
            example[v] = lmap[example[v]]
        return example
    
    # Relabel categorical
    for split, dataset in datasets.items():
        datasets[split] = dataset.map(lambda x : relabel(x, label_maps))

    return datasets    


def load_clothing():
    
    name = 'Censius-AI/ECommerce-Women-Clothing-Reviews'
    
    # Fetch dataset
    dataset = load_dataset(name, split = 'train')
    
    # Text vars
    txt_vars = ['Title', 'Review Text']    
    dataset = dataset.map(lambda x: process_text(x, txt_vars))
        
    # Cat vars
    cat_vars = ['Recommended IND', 'Division Name', 'Department Name', 'Class Name', 'Rating']
        
    # Create maps for relabeling
    label_maps = {}
    for var in cat_vars:
        label_maps[var] = {l : i for i, l in enumerate(set(dataset[var]))}
        
    # Relabiling function
    def relabel(example, label_maps):
        for v, lmap in label_maps.items():
            example[v] = lmap[example[v]]
        return example
    
    # Relabel categorical
    dataset = dataset.map(lambda x : relabel(x, label_maps))
        
    # Add split names
    datasets = split_dataset(dataset)
                
    return datasets


def load_datasets(name: str) -> dict: 
    func_name = 'load_{}'.format(name)    
    try:
        func = globals()[func_name]
    except:
        'Missing loader for {} dataset!'.format(name)
    return func()


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, y=None, **fit_params):
        return np.array(X.todense())


class FeatureExtractor():
    def __init__(self, datasets: dict, datacard: DataCard, pca_dims = 10,  max_words = 1000):
        self.datasets = datasets
        self.datacard = datacard
        self.pca_dims = pca_dims
        self.max_words = max_words
        self.features = {}        
        self.targets = {}    
    
    def extract(self, dataset):

        X = {v:np.array(dataset[v]) for v in self.datacard.features}
                        
        if self.datacard.response is not None:                
            y = np.array(dataset[self.datacard.response])
        else:
            y = None
        
        print('features extracted')
        return X, y

    def extract_sequentially(self, dataset):
        
        X = {k:[] for k in self.datacard.features} 
           
        if self.datacard.response is not None:
            y = []
        else:
            y = None
            
        for sample in dataset:
            for v in self.datacard.img_vars:
                sample[v] = sample[v].flatten().tolist()
            
            for k in X.keys():
                X[k].append(sample[k])
            
            if y is not None:
                y.append(sample[self.datacard.response])

        X = {k:np.array(x) for k, x in X.items()}            
        
        if y is not None:            
            y = np.array(y)
                
        return X, y
    
    def transform_img_var(self, var):
        pca = PCA(n_components=self.pca_dims)
        
        pca.fit(self.features['train'][var])
        
        for split, features in self.features.items():
            self.features[split][var] = pca.transform(features[var])
        
    def transform_txt_var(self, var):
        print('traisnforming this txt var:{}'.format(var))
        pipe = Pipeline(
            [
                ('vectorizer', CountVectorizer()), 
                ('to_dense', DenseTransformer()),  
                ('pca', PCA(self.pca_dims))
            ]
        )
        
        pipe.fit(self.features['train'][var])
          
        for split, features in self.features.items():
            self.features[split][var] = pipe.transform(features[var])
    
    def transform_cat_var(self, var):
        enc = OneHotEncoder(sparse_output=False)
        enc.fit(self.features['train'][var].reshape(-1, 1))
        
        for split, features in self.features.items():
            self.features[split][var] = enc.transform(features[var].reshape(-1, 1))
    
    def transform_num_var(self, var):
        for split, features in self.features.items():
            self.features[split][var] = features[var].reshape(-1, 1)
            
    def transform_features(self):
        for v in self.datacard.features:
            
            if v in self.datacard.img_vars:
                self.transform_img_var(v)
                
            elif v in self.datacard.txt_vars:
                self.transform_txt_var(v)

            elif v in self.datacard.cat_vars:
                self.transform_cat_var(v)
            
            else:
                self.transform_num_var(v)
        
        for split, features in self.features.items():            
            self.features[split] = np.concatenate(
                [x for x in features.values()], 
                axis = 1
            )
                            
    def __call__(self):
        self.features = {}
        self.targets = {}
        if len(self.datacard.img_vars) > 0:
            print('sequential extraction')
            for split, dataset in self.datasets.items():
                self.features[split], self.targets[split] = self.extract_sequentially(dataset)
        else:
            print('simple extraction')
            for split, dataset in self.datasets.items():
                self.features[split], self.targets[split] = self.extract(dataset)

        self.transform_features()                        
        return self.features, self.targets


def make_features(datasets, datacard):
    
    features = {}
    targets = {}
    for split, dataset in datasets.items():
        
        # Variables to extract
        vars = datacard.cat_vars + datacard.num_vars
        
        # Container for data    
        df = {v : [] for v in vars}
                
        for x in dataset:        
            for v in vars:
                df[v].append(x[v])
       
        df = pd.DataFrame.from_dict(df)       
        
        # Select targets
        if datacard.response is not None:
            y = df[datacard.response].values
            X = df.drop(columns = datacard.response)
        else:
            y = None        
            X = df
            
        features[split] = X
        targets[split] = y
    
    # Select categorical to encode 
    cat_features = [v for v in datacard.features if v in datacard.cat_vars]
    
    # Onehot encode categoricals
    if len(cat_features) > 0:
    
        # Init transformer
        transformer = ColumnTransformer(
            [
                ('encode_cats', OneHotEncoder(sparse_output=False), cat_features),
            ],
            remainder='passthrough'
        )
        
        # Fit transformer
        transformer.fit(features['train'])
        
        # Transform features
        for split, x in features.items():
            features[split] = transformer.transform(x)
            
    else:
        
        # Transform features
        for split, x in features.items():
            features[split] = x.values
        
    
    return features, targets