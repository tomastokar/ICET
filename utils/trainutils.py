import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam

from .auxutils import acc_rate, mse_error

class MMCRLTrainer():
    def __init__(self, 
                 model,   
                 datacard,
                 data_loaders,
                 learning_rate = 1e-4, 
                 decay = 1e-5,
                 tokenizer = None,
                 device = 'cpu',
                 verbosity = 10,
                 patience = 10,
                 checkpoint = './src/checkpoint.pt'):
        
        # Set models
        self.model = model
        
        # Set device
        self.device = device
        
        # Set model to device
        self.model.to(self.device)
        
        # Set datacard
        self.datacard = datacard
                        
        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr = learning_rate,
            weight_decay = decay
        ) 
        
        # Data loaders
        self.data_loaders = data_loaders
                            
        # Tokenizer
        self.tokenizer = tokenizer
        
        # Set verbosity 
        self.verbosity = verbosity
        
        # Early stopper
        if patience > 0:
            self.stopper = EarlyStop(
                checkpoint=checkpoint,
                patience=patience
            )
        else:
            self.stopper = None        
        
    def tokenize(self, X):    
        for v in self.datacard.txt_vars:
            X[v] = self.tokenizer(
                X[v], 
                return_tensors='pt',
                padding=True
            )                 
        return X 
              
    def _step(self, X):
        # Tokenize
        if self.tokenizer is not None:
            X = self.tokenize(X)
                
        # Set device
        X = {v : x.to(self.device) for v, x in X.items()}
        
        # Apply model        
        _, loss = self.model(X)                
                
        return loss
  
    def _eval(self):
        
        # Eval loss init
        eval_loss = 0.     
        
        self.model.eval()  
        with torch.no_grad():
            n = len(self.data_loaders['validation'])  
            for X in self.data_loaders['validation']:   
                
                # Run model                         
                loss = self._step(X)
                            
                # Increment loss and performance
                eval_loss += loss.item()
            
            eval_loss /= n
                    
        return eval_loss

    def _train_epoch(self):
        # Init loss
        loss_agg = 0.   
                    
        self.model.train()   
        n = len(self.data_loaders['train'])             
        for i, X in enumerate(self.data_loaders['train']):
            # Clean cache
            # torch.cuda.empty_cache()                
            
            # Apply model
            loss = self._step(X)
            
            # Backpropagate
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            
            # Loss and performance
            loss_agg += loss.item()
            
            # Printout                                
            if self.verbosity > 0:
                if i % self.verbosity == 0:                                            
                    print('Batch: {}/{}\t Loss: {:1.3f}'.format(i + 1, n, loss.item()))

        
        return loss_agg/n
    
    def _embed(self, X):
        # Tokenize
        if self.tokenizer is not None:
            X = self.tokenize(X)
                
        # Set device
        X = {v : x.to(self.device) for v, x in X.items()}
        
        # Apply model        
        z, _ = self.model(X, calc_loss = False)
                        
        return z
    
    def _make_embeddings(self, split):
        self.model.eval()
        with torch.no_grad():            
            
            # Initiate container for embeedings
            embeddings = []
            
            # Targets container
            if self.datacard.response is not None:
                targets = []
            else:
                targets = None
                            
            # Generate embeddings and associated targets
            for X in self.data_loaders[split]:
                
                # Get embeddings
                z = self._embed(X)
                
                # Release from GPU (if applicable)
                z = z.cpu() 
                
                # Capture embeddings                
                embeddings.append(z)                
                    
                # Capture targets                    
                if targets is not None:
                    targets.append(X[self.datacard.response].reshape(-1))
    
            # Concatenate embeddings
            embeddings = torch.cat(embeddings).numpy()
                        
            # Concatentate targets
            if targets is not None:
                targets = torch.cat(targets).reshape(-1).numpy()

        return embeddings, targets
        
    def embed_data(self):
        
        # Init containers
        embeddings = {}
        targets = {}
                
        # Make embedding for each split
        for split in self.data_loaders.keys():
            embeddings[split], targets[split] = self._make_embeddings(split)                                
                                    
        return embeddings, targets
        
    def train(self, epochs = 10):
        
        train_track = []
        valid_track = []
                
        start = time.time()                
        for epoch in range(epochs):
            
            # Train and validate
            train_loss = self._train_epoch()
            valid_loss = self._eval()    
            
            # Add to tracks
            train_track.append(train_loss)
            valid_track.append(valid_loss)
             
            end = time.time()
            print('\n\t Epoch: {}/{}, Avg. loss: {:1.3f}, Eval. loss: {:1.3f}'.format(epoch+1, epochs, train_track[-1], valid_track[-1]))
            print('\n\t Finished in {:1.2f} seconds!\n'.format(end - start))

            if self.stopper is not None:
                self.stopper(valid_loss, self.model)
                if self.stopper.stop:
                    print('\n\t !!! Early stopping !!!\n')
                    params = self.stopper.load_checkpoint()
                    self.model.load_state_dict(params)
                    self.stopper.counter = 0
                    self.stopper.stop = False
                    break
                                                                
        return train_track, valid_track    


class EarlyStop:
    def __init__(self, checkpoint, patience=5, delta=0):
        self.patience = patience
        self.counter = 0
        self.stop = False
        self.max_score = None
        self.min_loss = np.Inf
        self.delta = delta
        self.state = None
        self.checkpoint = checkpoint

    def __call__(self, loss, model):
        score = -loss
        if self.max_score is None:
            self.max_score = score
            self.save_checkpoint(loss, model)
        elif score < self.max_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.max_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        self.min_loss = loss
        torch.save(model.state_dict(), self.checkpoint)
        
    def load_checkpoint(self):
        return torch.load(self.checkpoint)


class PredictorTrainer():
    def __init__(self, 
                 model,
                 datacard,                 
                 data_loaders,
                 learning_rate = 1e-3, 
                 decay = 1e-5,
                 tokenizer = None,
                 device = 'cpu',
                 verbosity = 10,
                 patience = 5,
                 checkpoint = './src/checkpoint.pt'):

        # Set model
        self.model = model
                
        # Set device
        self.device = device
        
        # Add to device
        self.model.to(self.device)
            
        # Set datacard
        self.datacard = datacard
        
        # Parameters
        params = self.model.parameters()

        # Optimizer
        self.optimizer = Adam(
            params,
            lr = learning_rate,
            weight_decay = decay
        ) 
        
        # Data loader        
        self.data_loaders = data_loaders
            
        # Tokenizer
        self.tokenizer = tokenizer
        
        # Loss function
        if self.datacard.response in self.datacard.cat_vars:
            # Categorical loss
            self.loss_func = nn.CrossEntropyLoss()
            
            # Categorical eval func
            self.eval_func = acc_rate
            
        else:
            # Numeric loss
            self.loss_func = nn.MSELoss()
            
            # Numeric eval func
            self.eval_func = mse_error            

        # Verbosity
        self.verbosity = verbosity
        
        # Early stopper
        if patience > 0:
            self.stopper = EarlyStop(
                checkpoint=checkpoint,
                patience=patience
            )
        else:
            self.stopper = None
        
    def tokenize(self, X):    
        for v in self.datacard.txt_vars:
            X[v] = self.tokenizer(
                X[v], 
                return_tensors='pt',
                padding=True
            )                 
        return X 
              
    def _step(self, X):
        
        # Tokenize
        if self.tokenizer is not None:
            X = self.tokenize(X)
        
        # Set device
        X = {v:x.to(self.device) for v, x in X.items()}
                        
        # Get prediction
        y_hat = self.model(X)
        
        # Select reponse
        y = X[self.datacard.response]
        
        # Calc loss
        loss = self.loss_func(y_hat, y)
                
        return loss
  
    def _eval(self):
        
        # Eval loss init
        eval_loss = 0.     
        
        self.model.eval()
        with torch.no_grad():
            n = len(self.data_loaders['validation'])  
            for X in self.data_loaders['validation']:   
                
                # Get loss
                loss = self._step(X)               
                
                # Increment loss
                eval_loss += loss.item()
            
            # Average loss
            eval_loss /= n
                    
        return eval_loss
    
    def _train_epoch(self):
        # Init loss
        loss_agg = 0.   
                    
        self.model.train()  
        n = len(self.data_loaders['train'])              
        for i, X in enumerate(self.data_loaders['train']):
            # Clean cache
            # torch.cuda.empty_cache()                
            
            # Apply model
            loss = self._step(X)
            
            # Backpropagate
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            
            # Loss and performance
            loss_agg += loss.item()
            
            # Printout                                
            if self.verbosity > 0:
                if i % self.verbosity == 0:                                            
                    print('Batch: {}/{}\t Loss: {:1.3f}'.format(i + 1, n, loss.item()))
        
        return loss_agg/n
    
    def train(self, epochs = 10):
        
        # Train track container
        train_track = []        
        valid_track = []
        
        start = time.time()                
        for epoch in range(epochs):
            
            train_loss = self._train_epoch()
            valid_loss = self._eval()
                
            train_track.append(train_loss)
            valid_track.append(valid_loss)                
            
            end = time.time()
            print('\n\t Epoch: {}/{}, Avg. loss: {:1.3f}, Eval. loss: {:1.3f}'.format(epoch+1, epochs, train_track[-1], valid_track[-1]))
            print('\n\t Finished in {:1.2f} seconds!\n'.format(end - start))
            
            if self.stopper is not None:
                self.stopper(valid_loss, self.model)
                if self.stopper.stop:
                    print('\n\t !!! Early stopping !!!\n')
                    params = self.stopper.load_checkpoint()
                    self.model.load_state_dict(params)
                    self.stopper.counter = 0
                    self.stopper.stop = False
                    break
                                 
        return train_track, valid_track   
    
    def test(self):
        
        self.model.eval()
        with torch.no_grad():
            preds = []
            targets = []
            for X in self.data_loaders['test']:
                
                # Tokenize
                if self.tokenizer is not None:
                    X = self.tokenize(X)
                
                # Set device
                X = {v:x.to(self.device) for v, x in X.items()}
                                
                # Get prediction
                y_hat = self.model(X)
                
                # Select reponse
                y = X[self.datacard.response]
                
                # To cpu
                y_hat = y_hat.cpu()
                y = y.cpu()
                
                preds.append(y_hat)
                targets.append(y)
                
            preds = torch.cat(preds, dim = 0)
            targets = torch.cat(targets, dim = 0)
        
        # If logits returned convert 
        # to class preds or, else flatten
        if preds.size(1) > 1:
            preds = torch.argmax(preds, dim = 1)
        else:
            preds = preds.flatten()
                
        performance = self.eval_func(targets, preds)
        
        return performance
        
        
            
            
            

            
        
        
        
         
