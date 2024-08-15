import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .datautils import DataCard

class FeatureExtractor():
    def __init__(self, datasets: dict, datacard: DataCard):
        self.datasets = datasets
        self.datacard = datacard
      
    def _extract(self, dataset):

        X = {v : dataset[v] for v in self.datacard.features}
        X = pd.DataFrame.from_dict(X)
                                
        if self.datacard.response is not None:                
            y = np.array(dataset[self.datacard.response])
        else:
            y = None
                
        return X, y

    def _transform_features(self):
        # Select categorical to encode 
        cat_features = [v for v in self.datacard.features if v in self.datacard.cat_vars]
    
        # Onehot encode categoricals
        if len(cat_features) > 0:
        
            # Init transformer
            transformer = ColumnTransformer(
                [
                    (
                        'encode_cats', 
                        OneHotEncoder(sparse_output=False, handle_unknown = 'ignore'), 
                        cat_features
                    ),
                ],
                remainder='passthrough'
            )
            
            # Fit transformer
            transformer.fit(self.features['train'])
            
            # Transform features
            for split, X in self.features.items():
                self.features[split] = transformer.transform(X)
                
        else:
            
            # Transform features
            for split, x in self.features.items():
                self.features[split] = x.values
                                    
    def __call__(self):
        self.features = {}
        self.response = {}
            
        for split, dataset in self.datasets.items():
            X, y = self._extract(dataset)
            self.features[split] = X
            self.response[split] = y
            
        self._transform_features()                        
        return self.features, self.response

