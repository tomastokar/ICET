import time
import yaml
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier

from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

from utils.datautils import load_datasets, DataCard
from utils.testutils import ShallowTester

NUM_REPLICATES = 5

OUTPUT_COLUMNS = [
    'replicate',
    'kmeans_perf',
    'knn_perf',  
    'predictor_perf'
]

MAX_SAMPLES = int(2e+4)

def main():
    # Create datacard    
    with open('./src/datacards.yml', 'r') as f:
        datacards = yaml.load(f, Loader=yaml.FullLoader)      
        
    start = time.time()            
    for name, datacard in datacards.items():
                
        if name == 'xor_blobs':
            continue
                    
        # Init datacard
        datacard = DataCard(**datacard)
        
        if (len(datacard.img_vars) + len(datacard.txt_vars) == 0) and (datacard.response is not None):            
                            
            # Init results data frame
            results = pd.DataFrame([], columns = OUTPUT_COLUMNS)
                        
            for replicate in range(NUM_REPLICATES):
                                                                        
                # Load data
                datasets = load_datasets(name)
                
                # -------------------
                # Data preprocessing
                # -------------------
                   
                # Convert to features and reponses
                features = {}
                response = {}            
                for split, dataset in datasets.items():
                    dataset = pd.DataFrame(dataset)
                    features[split] = dataset[datacard.features]
                    response[split] = dataset[datacard.response]

                # Cateogrical features
                cat_features = [f for f in datacard.features if f in datacard.cat_vars]

                # Init data transformer
                transformer = ColumnTransformer(
                    [
                        (
                            '{}_encoder'.format(var), 
                            OneHotEncoder(sparse_output=False, handle_unknown = 'ignore'), 
                            [var]
                        ) for var in cat_features
                    ],
                    remainder='passthrough',
                    verbose_feature_names_out=False
                )   
                
                # Fit transformer
                transformer.fit(features['train'])
                
                # Output feature names
                columns = transformer.get_feature_names_out()
                
                # Transform features
                features_encoded = {}
                for split, X in features.items():
                    features_encoded[split] = pd.DataFrame(
                        transformer.transform(X), 
                        columns = columns
                    )
                    
                # Embeddings from features (numpy arrays)
                embeddings = {
                    split : X.values for split, X in features_encoded.items()
                }
                
                # -----------------------------------
                # Test supervised learning (XGBoost)
                # -----------------------------------                
                if datacard.response in datacard.cat_vars:            
                    xgb = XGBClassifier()
                    
                    # Fit XGB
                    xgb.fit(
                        embeddings['train'], 
                        response['train']
                    )
                    
                    # Make predictions
                    preds = xgb.predict(embeddings['test'])
                    preds = [round(pred) for pred in preds]
                    
                    # Evaluate predictions
                    predictor_perf = balanced_accuracy_score(response['test'], preds)
                    
                else:                    
                    xgb = XGBRegressor()
                    
                    # Fit XGB
                    xgb.fit(
                        embeddings['train'], 
                        response['train']
                    )
                    
                    # Make predictions
                    preds = xgb.predict(embeddings['test'])  
                
                    # Evaluate predictions
                    predictor_perf = mean_squared_error(response['test'], preds)     
                
                
                # ------------------------------------
                # Subsample training data if required
                # ------------------------------------
                n = len(response['train']) 
                if n > MAX_SAMPLES:
                    # Which samples to use
                    idx = np.random.permutation(n)[:MAX_SAMPLES]
                                        
                    # Sumbsample features
                    features_encoded['train'] = features_encoded['train'].iloc[idx,:]
                    features['train'] = features['train'].iloc[idx,:]
                    response['train'] = response['train'][idx]
                    embeddings['train'] = embeddings['train'][idx] 
                                    
                # ------------------------------------
                # Test clustering and KNN of features
                # ------------------------------------
                            
                # Get features
                tester = ShallowTester(
                    datacard=datacard,
                    embeddings = embeddings, 
                    targets = response
                )
                
                kmeans_perf, knn_perf = tester.test()
                                                            
                # --------------------------    
                # Assemble new results line
                # --------------------------
                result = [
                    replicate, kmeans_perf, knn_perf, predictor_perf
                ]
                                                                                    
                # Add to results
                results.loc[len(results)] = result

                # Report
                print('\n')
                print(results.iloc[-1])
                print('\n\tElapsed time: {:1.1f} seconds!\n'.format(time.time() - start))
                
                # Save to file
                results.to_csv('./results/{}_benchmark.csv'.format(name))
                               


if __name__ == '__main__':
    main()