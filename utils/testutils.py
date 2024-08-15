import time
import torch
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, silhouette_score
from .auxutils import acc_rate, mse_error


def get_xor_embeddings(model, data):    
    # Generate embeddings
    model.eval()
    with torch.no_grad():
        # Generate embeddings
        embs, joint, _ = model(data.data, calc_loss = False)    
    
        # Calc similarities
        sims = model.sim_func(joint, joint)
        
        # Outputs
        output = {}
        output['embs'] = {v:embs[i].cpu().numpy() for i, v in enumerate(data.vars)}
        output['sims'] = sims.cpu().numpy()
        output['labs'] = data.data[0].cpu().numpy()
    
    return output            


class ShallowTester():
    def __init__(self, datacard, embeddings: dict, targets: dict, max_clusters: int = 10, k_max: int = 10, max_samples: int = int(2e+4)):
        
        # Set datacard
        self.datacard = datacard
        
        # Set embeddings and targets
        self.embeddings = embeddings
        self.targets = targets
                
        # Maximum number for knn
        self.k_max = k_max
        
        # Maximum number of clusters for kmeans
        self.max_clusters = max_clusters
        
        # Maximum number of samples to use
        self.max_samples = max_samples
        
        # Subsample if required
        if len(self.embeddings['train']) > self.max_samples:
            self._subsample()        
                                                
    def _subsample(self):

        # Number of samples
        n = len(self.embeddings['train'])
        
        # Which samples to use
        idx = np.random.permutation(n)[:self.max_samples]
                            
        # Sumbsample features
        self.embeddings['train'] = self.embeddings['train'][idx,:]
        
        # Subsample targets
        if self.targets['train'] is not None:
            self.targets['train'] = self.targets['train'][idx]

                         
    def test_clustering(self): 
                       
        # Benchmark sillhouette score
        best_silhouette = 0.        
        
        print('\n\tFitting k-means:\n')
        for k in range(2, self.max_clusters + 1):
            
            # Init k-means clustering
            kmeans = KMeans(n_clusters=k) #, n_init = 'auto')
            
            # Fit kemans
            X_ = np.unique(self.embeddings['train'], axis = 0) # To prevent warnings
            kmeans.fit(X_)
                            
            # Get distances to nearest centroids
            dmat = kmeans.transform(self.embeddings['validation'])
                        
            # Get centroid labes
            labs = dmat.argmin(axis = 1)
                    
            # Eval silhouette
            try: 
                silhouette = silhouette_score(self.embeddings['validation'], labs)
            except:
                silhouette = 0.
            
            print('K={}; Eval. silhouette={:1.2f}'.format(k, silhouette))
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                self.kmeans = kmeans
        
        # Test set clustering                
        dmat = self.kmeans.transform(self.embeddings['test'])
        
        # Centroids
        labs = dmat.argmin(axis = 1)
        
        # Test score
        try: 
            silhouette = silhouette_score(self.embeddings['test'], labs)
        except: 
            silhouette = None # If only single cluster labels are predicted

        return silhouette
    
    def test_knn_classification(self):
                
        # Benchmark acuracy score
        best_acc = 0.        
        
        print('\n\tFitting KNN:\n')
        for k in range(2, self.k_max + 1):
        
            # Init KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            
            # Fit KNN
            knn.fit(self.embeddings['train'], self.targets['train'])
        
            # Get predicitons
            labs = knn.predict(self.embeddings['validation'])
        
            # Calc accuracy
            acc = balanced_accuracy_score(self.targets['validation'], labs)
            
            print('K={}; Eval. accuracy={:1.2f}'.format(k, acc))
            if acc > best_acc:
                best_acc = acc
                self.knn = knn
        
        # Test set clustering                
        labs = self.knn.predict(self.embeddings['test'])
                
        # Test accuracy
        accuracy = balanced_accuracy_score(self.targets['test'], labs)
                    
        return accuracy
                
    def test_knn_regression(self):
                
        # Benchmark acuracy score
        best_mse = np.Inf        
        
        print('\n\tFitting KNN:\n')
        for k in range(2, self.k_max + 1):
        
            # Init KNN classifier
            knn = KNeighborsRegressor(n_neighbors=k)
            
            # Fit KNN
            knn.fit(self.embeddings['train'], self.targets['train'])
        
            # Get predicitons
            labs = knn.predict(self.embeddings['validation'])
        
            # Calc accuracy
            mse = mean_squared_error(self.targets['validation'], labs)
            
            print('K={}; Eval. error={:1.2f}'.format(k, mse))
            if mse < best_mse:
                best_mse = mse
                self.knn = knn
        
        # Test set clustering                
        labs = self.knn.predict(self.embeddings['test'])
                
        # Test accuracy
        mse = mean_squared_error(self.targets['test'], labs)
                    
        return mse
                    
    def test(self):
        
        # Init timer
        start = time.time()
        
        # Test clustering
        clustering = self.test_clustering() 
        
        # Test supervised performance
        prediction = None
        if self.datacard.response is not None:
            if self.datacard.response in self.datacard.cat_vars:
                prediction = self.test_knn_classification()
            else:
                prediction = self.test_knn_regression()

        print('\n\t Shallow testing finished in {:1.2f} seconds!\n'.format(time.time() - start)) 
        return clustering, prediction


class TransferTester():
    def __init__(self, model, datacard, data_loaders, tokenizer = None, device = 'cpu'):        
        # Set model
        self.model = model        
        
        # Set device
        self.device = device
        
        # Put model to device
        self.model.to(self.device)
        
        # Set datacard
        self.datacard = datacard
                
        # Data loaders        
        self.data_loaders = data_loaders
                
        # Tokenizer
        self.tokenizer = tokenizer               

    def cat_eval(self, X):
        # Predict values of cat vars
        # and calculate accuracy rates
        accuracies = {}
        for query_var in self.datacard.cat_vars:
            
            if query_var != self.datacard.response:              
                                              
                # Get list of evidence inputs
                evidence = {v:X[v] for v in self.datacard.features if v != query_var}
                            
                candidates = X[query_var]
                
                # Run modality transfer
                x_hat = self.model.transfer(query_var, candidates, evidence)
                            
                # Calculate accuracy
                acc = acc_rate(X[query_var], x_hat)

                # Add to container
                accuracies[query_var] = acc.item()
                                    
        return accuracies

    def num_eval(self, X):
        # Predict values of num vars
        # and calculate MSE rates        
        errors = {}
        for query_var in self.datacard.num_vars:
            
            if query_var != self.datacard.response:              
                                              
                # Get list of evidence inputs
                evidence = {v:X[v] for v in self.datacard.features if v != query_var}
                
                # Query input (candiates)
                candidates = X[query_var]
                
                # Run modality transfer
                x_hat = self.model.transfer(query_var, candidates, evidence)
                
                # Calculate error
                err = mse_error(X[query_var], x_hat)

                # Add to container
                errors[query_var] = err.item()
            
        return errors      
    
    def tokenize(self, X):    
        for v in self.datacard.txt_vars:
            X[v] = self.tokenizer(
                X[v], 
                return_tensors='pt',
                padding=True
            )
            
                 
        return X   

    def step(self, X):
        
        # Tokenize
        if self.tokenizer is not None:
            X = self.tokenize(X)
                    
        # To device
        X = {v:x.to(self.device) for v, x in X.items()}
        
        # Eval transfer
        accs = self.cat_eval(X)                    
        errs = self.num_eval(X)                    
            
        # Return accuracy and error rates 
        return accs, errs
        
    def test(self):
        start = time.time()
        self.model.eval()                    
        with torch.no_grad():
                                        
            # Init container for accuracies and errors
            total_accs = {v : 0. for v in self.datacard.cat_vars if v != self.datacard.response}
            total_errs = {v : 0. for v in self.datacard.num_vars if v != self.datacard.response}
            
            # Iterate over the batches
            n = len(self.data_loaders['test'])
            for i, X in enumerate(self.data_loaders['test']):
                
                # Make step
                accs, errs = self.step(X)
                
                # Add to containers
                for v, acc in accs.items():
                    total_accs[v] += acc

                for v, err in errs.items():
                    total_errs[v] += err

                print('Testing batch {}/{}.'.format(i, n))
                                        
            # Calculate average
            for v, acc in accs.items():
                total_accs[v] /= n

            for v, err in errs.items():
                total_errs[v] /= n
                                       
            print('\n\t Finished in {:1.2f} seconds!\n'.format(time.time() - start))
        
        return accs, errs
        

