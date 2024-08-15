import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel
from random import shuffle


class LinearPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearPredictor, self).__init__()
        self.W = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.W(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = []):
        super(MLP, self).__init__()        
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))            
        self.layers = nn.Sequential(*layers)    
        self.output_dim = output_dim
        
    def forward(self, x):
        y = self.layers(x)
        return y


class MLPNormed(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = []):
        super(MLPNormed, self).__init__()        
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))            
        self.layers = nn.Sequential(*layers)    

    def forward(self, x):
        y = self.layers(x)
        return y
        

class MLPSwished(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = []):
        super(MLPSwished, self).__init__()        
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.SiLU())
            # layers.append(nn.BatchNorm1d(dim))
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))            
        self.layers = nn.Sequential(*layers)    

    def forward(self, x):
        y = self.layers(x)
        return y


class ImageMapping(nn.Module):
    def __init__(self, output_dim):
        super(ImageMapping, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.projector = nn.Linear(1000, output_dim)        
        self.output_dim = output_dim
        
    def forward(self, x):
        y = self.encoder(x)
        y = F.relu(y)
        z = self.projector(y)
        return z


class TextMapping(nn.Module):
    def __init__(self, output_dim):
        super(TextMapping, self).__init__()              
        self.transformer = BertModel.from_pretrained('bert-base-uncased')  # RobertaModel.from_pretrained('roberta-base')    
        self.projector = nn.Linear(768, output_dim)
        self.output_dim = output_dim
        
    def forward(self, x):                
        y = self.transformer(**x)
        y = y['pooler_output']
        y = F.relu(y)
        z = self.projector(y)                   
        return z


class CatMapping(nn.Module):
    def __init__(self, emb_num, output_dim):
        super(CatMapping, self).__init__()
        self.embedding = nn.Embedding(emb_num, output_dim)        
        self.output_dim = output_dim
        
    def forward(self, x):        
        z = self.embedding(x)
        return z


# class CatMapping(nn.Module):
#     def __init__(self, emb_num, emb_dim, latent_dim):
#         super(CatMapping, self).__init__()
#         self.embedding = nn.Embedding(emb_num, emb_dim)
#         self.projector = nn.Linear(emb_dim, latent_dim)
        
#     def forward(self, x):
#         try:
#             y = self.embedding(x)
#         except:
#             print(self.embedding.num_embeddings)
#             print(x.max())
#         z = self.projector(y)                    
#         return z


class CLIPEmbedder(nn.Module):
    def __init__(self, mappings: dict, tau_init: float = 0.07):
        super(CLIPEmbedder, self).__init__()   
        
        # Base mappings     
        self.mappings = nn.ModuleDict(mappings)   
                
        # Temperature parameters  
        self.tau = nn.Parameter(torch.log(torch.tensor(tau_init)))
                        
    def sim_func(self, x, y):
        a = F.normalize(x, dim = 1)
        b = F.normalize(y, dim = 1)
        sim = torch.matmul(a, b.T)
        sim = sim / torch.exp(self.tau)
        return sim        
        
    def calc_loss(self, Z):
        loss = 0.
        for vi, zi in Z.items():                
            sims = sum([self.sim_func(zi, zj) for vj, zj in Z.items() if vj != vi])   
            labs = torch.arange(sims.shape[0], device=sims.device)        
            loss += F.cross_entropy(sims, labs, reduction='mean') 
        loss /= len(Z)                                                  
        return loss
    
    def transfer(self, query_var: str, candidates: torch.Tensor, evidence: dict):
        '''
        Modality transfer
        '''        
        
        # Get latent representations of evidence
        Z = [self.mappings[v](x) for v, x in evidence.items()]
        
        # Get latent representation of query
        z = self.mappings[query_var](candidates)
        
        # Calc pairwise similarity
        sim = sum([self.sim_func(z, zi) for zi in Z]) 
        
        # Select candidate value
        estimate = candidates[torch.argmax(sim, dim = 0)]
        
        return estimate     
    
    def forward(self, X: dict, calc_loss = True):
           
        # Apply mappings     
        Z = {v:mapping(X[v]) for v, mapping in self.mappings.items()}
                
        # Calculate similarities        
        if calc_loss:            
            loss = self.calc_loss(Z)
        else:
            loss = None
            
        # Embeddings list
        Z = [z for z in Z.values()]
        
        # Aggregate embeddings
        z = sum(Z) / len(Z)        
            
        return z, loss


class GMCEmbedder(nn.Module):
    def __init__(self, mappings: dict, aux_maps: dict, hidden_dim: int, latent_dim: int, tau_init = 0.1):
        super(GMCEmbedder, self).__init__()        
        
        # Base mappings
        self.mappings = nn.ModuleDict(mappings) 
        
        # Mapping to be used to get joint representions
        self.aux_maps = nn.ModuleDict(aux_maps) 
                
        # Additional layer to project to intermediate space
        self.multi_map = nn.Linear(len(self.aux_maps) * hidden_dim, hidden_dim)      
        
        # self.multi_map = MLPNormed(len(self.ordering) * input_dim, input_dim, hidden_dims=[input_dim,])
        
        # Projection from intermediate to latent space 
        # self.projector = MLPNormed(input_dim, latent_dim, hidden_dims=[input_dim,]) 
        self.phi = MLPSwished(hidden_dim, latent_dim, hidden_dims=[hidden_dim,]) 
                
        # Temperature parameters  
        self.tau = torch.tensor(tau_init, requires_grad=True)
                        
    def sim_func(self, x, y):
        a = F.normalize(x, dim = 1)
        b = F.normalize(y, dim = 1)
        sim = torch.matmul(a, b.T)
        sim = sim / self.tau
        return sim   
    
    def get_joint(self, X: dict):
        # Get joint representation
        H = [aux_map(X[v]) for v, aux_map in self.aux_maps.items()]
        h = torch.cat(H, dim = 1)  
        
        # Map to intermediary space      
        h = self.multi_map(h) 
        
        # Get joint latent representation
        a = self.phi(h)
        return a
                        
    def calc_loss(self, Z, a):      
                                  
        # Calculate joint-joint similarity
        joint_sim = torch.exp(self.sim_func(a, a))        
        
        # Iterate though modalities                                    
        loss = 0.
        for z in Z.values():
            
            # Positive
            pos_sim = torch.exp(self.sim_func(a, z))
            
            # Negative
            neg_sim = sum([torch.exp(self.sim_func(z, z)), pos_sim, joint_sim])
            
            # Mask out neg
            neg_sim = neg_sim * (1. - torch.eye(neg_sim.size(0), device = neg_sim.device))
        
            # Positive diagonal
            pos_sim = torch.diag(pos_sim, 0)
                                                
            # Increment loss
            loss += -torch.log(pos_sim / neg_sim.sum(dim = 1)).mean()
            
        loss /= len(Z)                    
        return loss
    
    def transfer(self, query_var: str, candidates: torch.Tensor, evidence: dict):
        '''
        Modality transfer
        '''        
        
        # Get intermediate representations of evidence
        H = [self.mappings[v](x) for v, x in evidence.items()]

        # Get intermediate representations of evidence
        Z = [self.phi(h) for h in H]
                
        # Get intermediate representation of query
        h = self.mappings[query_var](candidates)
        
        # Get latent representation of query
        z = self.phi(h)
        
        # Calc pairwise similarity
        sim = sum([self.sim_func(z, zi) for zi in Z])
        
        # Select candidate value
        estimate = candidates[torch.argmax(sim, dim = 0)]
                
        return estimate    
    
    def forward(self, X: dict, calc_loss = True):
                    
        # Get intermediate modality-specific representations
        H = {v:mapping(X[v]) for v, mapping in self.mappings.items()}
                
        # Get latent modality-specific representations
        Z  = {v:self.phi(h) for v, h in H.items()}        

        # Get joint representation   
        z = self.get_joint(X)                        
                               
        # Calculate loss
        if calc_loss:       
       
            # Calculate loss                   
            loss = self.calc_loss(Z, z)
        else:
            loss = None
                        
        return z, loss


class MMSLoss(nn.Module):
    def __init__(self):
        super(MMSLoss, self).__init__() 
        
    def forward(self, sims, margin = 0.001):
        # Calc delta
        deltas = torch.eye(sims.shape[0], device = sims.device) * margin
        
        # Adjust similarities
        sims = sims - deltas
        
        # Calc loss
        labs = torch.arange(sims.shape[0], device=sims.device)
        loss = F.nll_loss(F.log_softmax(sims, dim=1), labs)        
        return loss
       

class MCNEmbedder(nn.Module):
    def __init__(self, mappings: dict, no_anchors: int, latent_dim: int, margin: float = 0.001):
        super(MCNEmbedder, self).__init__()
        # Base mappings        
        self.mappings = nn.ModuleDict(mappings)                      
        
        # Centroids
        self.centroids = nn.Embedding(no_anchors, latent_dim)  
        
        # Margin
        self.margin = margin
        
        # Loss func
        self.loss_func = MMSLoss()
        
    # def sim_func(self, x, y):
    #     a = F.normalize(x, dim = 1)
    #     b = F.normalize(y, dim = 1)
    #     sim = torch.matmul(a, b.T)
    #     return sim  
    
    def sim_func(self, x, y):
        return torch.matmul(x, y.T)
                
    def contrastive_loss(self, Z: dict):
        loss = 0.
        for vi, zi in Z.items():
            # Paiwise similarities                
            sims = [self.sim_func(zi, zj) for vj, zj in Z.items() if vj != vi]
            loss += sum([self.loss_func(sim, self.margin) for sim in sims])    
        loss /= len(Z)        
        return loss

    def clustering_loss(self, Z, a):      
          
        # Get nearest centoids
        dx = torch.cdist(a, self.centroids.weight.data).pow(2)
        mu = self.centroids(torch.argmin(dx, dim = 1))
        
        # Calc loss across modalities
        loss = 0.
        for z in Z.values():
            # Calc similarities to nearest centroids
            sims = self.sim_func(z, mu)
            
            # Calc loss            
            loss += self.loss_func(sims, self.margin)
        loss /= len(Z)
        return loss
    
    def transfer(self, query_var: str, candidates: torch.Tensor, evidence: dict):
        '''
        Modality transfer
        '''        
        
        # Get latent representations of evidence
        Z = [self.mappings[v](x) for v, x in evidence.items()]
                
        # Get latent representation of query
        z = self.mappings[query_var](candidates)
        
        # Calc pairwise similarity
        sim = sum([self.sim_func(z, zi) for zi in Z])
        
        # Select candidate value
        estimate = candidates[torch.argmax(sim, dim = 0)]
                
        return estimate          
        
    def forward(self, X: dict, calc_loss = True):
        
        # Get latent representations
        Z = {v : mapping(X[v]) for v, mapping in self.mappings.items()}

        # Get joint latent representation
        z = sum(Z.values()) / len(Z)
                
        # Calculate similarities
        if calc_loss:            
            loss = self.contrastive_loss(Z)
            loss += self.clustering_loss(Z, z)        
        else:
            loss = None
                  
        return z, loss


class ICETEmbedder(nn.Module):
    def __init__(self, mappings: dict, phi_dims: list, hidden_dim: int, latent_dim: int, tau_init: float = 0.07):
        super(ICETEmbedder, self).__init__()
        # Base mappings                
        self.mappings = nn.ModuleDict(mappings)   
        
        # Phi function
        self.phi = MLP(hidden_dim, latent_dim, phi_dims)
        
        # Temperature
        self.tau = nn.Parameter(torch.log(torch.tensor(tau_init)))
        
    def sim_func(self, x, y):
        a = F.normalize(x, dim = 1)
        b = F.normalize(y, dim = 1)
        sim = torch.matmul(a, b.T)
        sim = sim / torch.exp(self.tau)
        return sim   
        
    def loss_func(self, sims):
        labs = torch.arange(sims.shape[0], device=sims.device)        
        loss = F.cross_entropy(sims, labs, reduction='mean') 
        return loss
    
    def calc_loss(self, H, h):
        # Sum latent representations
        H_sum = h * len(H)
        
        # Denominator
        k = len(H) - 1 
        
        # Calc loss
        loss = 0.        
        for hv in H.values():
            
            # Calc anchor
            mu = (H_sum - hv) / k  
            
            # Project by phi
            zv, z_mu = self.phi(hv), self.phi(mu)
                        
            # Calc similarities
            sims = self.sim_func(zv, z_mu)
            
            # Calc loss
            loss += self.loss_func(sims)
        
        return loss
        
        
    def transfer(self, query_var: str, candidates: torch.Tensor, evidence: dict):
        '''
        Modality transfer
        '''
                
        # Get latent representations of evidence
        H = [self.mappings[v](x) for v, x in evidence.items()]
        
        # Joint representation of evidence
        mu = sum(H) / len(H)

        # Get latent representation of query
        h = self.mappings[query_var](candidates)

        # Make projections
        z, z_mu = self.phi(h), self.phi(mu)
                
        # Calc similarity
        sim = self.sim_func(z, z_mu)
                        
        # Select candidate value
        estimate = candidates[torch.argmax(sim, dim = 0)]
        
        return estimate
        
    
    def forward(self, X: dict, calc_loss = True):
        # Get interemdiate representations
        H = {v:mapping(X[v]) for v, mapping in self.mappings.items()}
        
        # Get joint intermediate representation
        h = sum([v for v in H.values()])/len(H)

        # Get joint representation
        z = self.phi(h)
                                    
        # Calculate loss
        if calc_loss:  
            loss = self.calc_loss(H, h)
        else:                        
            loss = None
                                
        return z, loss


class SCARFEmbedder(nn.Module):
    def __init__(self, mappings: dict, latent_dim: int, corruption_rate: float = 0.5, tau: float = 1.0):
        super(SCARFEmbedder, self).__init__()

        # Base mappings                
        self.mappings = nn.ModuleDict(mappings)   
        
        # Names of the embedded vars
        self.vars = [v for v in self.mappings.keys()]
        
        # Number of inputs
        self.M = len(self.mappings)
        
        # Projection head function   
        self.projector = MLP(
            latent_dim * self.M, 
            latent_dim, 
            [latent_dim,]
        )
        
        # Number of inputs to corrupt
        self.Q = int(corruption_rate * self.M)
        
        # Temperature
        self.tau = torch.tensor(tau)
    
    def corrupt_view(self, H):    
        
        # Init corrupted instance
        H_ = {v : torch.clone(x) for v, x in H.items()}
        
        # Select variables to corrupt
        shuffle(self.vars)
        corrupt_vars = self.vars[:self.Q]
                                
        # Corrupt input
        for v in corrupt_vars:   
            n = H[v].size(0)         
            idx = torch.randint(n, (n,)) #torch.randperm(X[v].size(0))
            H_[v] = torch.clone(H[v][idx])
                
        return H_
        
    
    def mini_forward(self, H):

        # Concat
        h = torch.cat([v for v in H.values()], axis = 1)        
        
        # Project
        z = self.projector(h)
        
        return z
    
    def sim_func(self, x, y):
        a = F.normalize(x, dim = 1)
        b = F.normalize(y, dim = 1)
        sim = torch.matmul(a, b.T)
        sim = sim / self.tau
        return sim   
        
    def loss_func(self, sims):
        labs = torch.arange(sims.shape[0], device=sims.device)        
        loss = F.cross_entropy(sims, labs, reduction='mean') 
        return loss    

    def forward(self, X, calc_loss = True):
        
        # Get interemdiate representations
        H = {v:mapping(X[v]) for v, mapping in self.mappings.items()}
                
        # Genuine representation
        z  = self.mini_forward(H)
        
        # Calculate loss
        if calc_loss:  
            
            if self.training:          
                # Corrupt inputs
                H_ = self.corrupt_view(H)

                # Corrupted representation
                z_ = self.mini_forward(H_)
                
                # Compute loss
                sims = self.sim_func(z, z_)                        
                loss = self.loss_func(sims)    
            
            else:

                # Compute loss
                sims = self.sim_func(z, z)
                loss = self.loss_func(sims)
                            
        else:
            loss = None

        return z, loss 
        

class SUBTABEmbedder(nn.Module):
    def __init__(self, mappings: dict, latent_dim: int, num_blocks: int = 4, overlap: float = 0.75, num_skip: int = 1, tau: float = 1.0):
        super(SUBTABEmbedder, self).__init__()
        
        # Base mappings                
        self.mappings = nn.ModuleDict(mappings)   
    
        # Number of modalities
        self.M = len(self.mappings)    
        
        if num_blocks > self.M:
            num_blocks = self.M
            num_skip = 0
            overlap = 0
            
        # Number of blocks to use
        self.num_use = num_blocks - num_skip
        
        # Block size        
        n_cols = max(1, int(self.M / num_blocks))
        n_overlap = int(overlap * n_cols)
        self.block_size = n_cols + n_overlap 
                
        # Set boundaries
        self.boundaries = []
        for i in range(num_blocks):
            if i == 0:
                start = 0
                end = n_cols + n_overlap                
            else:
                start = i * n_cols - n_overlap
                end = (i + 1) * n_cols
            self.boundaries.append([start, end])
        
        # Projector
        self.projector = MLP(
            latent_dim * self.block_size, 
            latent_dim, 
            [latent_dim,]
        )
        
        # Temperature
        self.tau = tau
        
    def sim_func(self, x, y):
        a = F.normalize(x, dim = 1)
        b = F.normalize(y, dim = 1)
        sim = torch.matmul(a, b.T)
        sim = sim / self.tau
        return sim        
        
    def calc_contrastive_loss(self, Z):
        loss = 0.
        for i, zi in enumerate(Z):                
            sims = sum([self.sim_func(zi, zj) for j, zj in enumerate(Z) if j != i])   
            labs = torch.arange(sims.shape[0], device=sims.device)        
            loss += F.cross_entropy(sims, labs, reduction='mean') 
        loss /= len(Z)                                                  
        return loss     
    
    def calc_contrastive_loss(self, Z):
        loss = 0.
        for i, zi in enumerate(Z):                
            sims = sum([self.sim_func(zi, zj) for j, zj in enumerate(Z) if j != i])   
            labs = torch.arange(sims.shape[0], device=sims.device)        
            loss += F.cross_entropy(sims, labs, reduction='mean') 
        loss /= len(Z)                                                  
        return loss    
    
    def dist_func(self, a, b):
        return (a - b).pow(2).mean()
    
    def calc_distance_loss(self, Z):
        loss = 0.
        for i, zi in enumerate(Z):                
            loss += sum([self.dist_func(zi, zj) for j, zj in enumerate(Z) if j != i]) / self.M                        
        loss /= self.M
        return loss    

    def forward(self, X, calc_loss = True):
        
        # Get latent presentations representations
        Z = [mapping(X[v]) for v, mapping in self.mappings.items()]
                
        if calc_loss:            
            Z_blocks = []
            shuffle(self.boundaries)            
            for start, end in self.boundaries[:self.num_use]:
                block = torch.cat(Z[start:end], dim = 1)
                z_block = self.projector(block)
                Z_blocks.append(z_block)
        
            loss = self.calc_contrastive_loss(Z_blocks)
            loss += self.calc_distance_loss(Z_blocks)
        
        else:
            loss = None
        
        # Get joint representation
        z = sum(Z) / len(Z)
        
        return z, loss
    

class Predictor(nn.Module):
    def __init__(self, mappings: dict, latent_dim: int, output_dim: int, phi: nn.Module = None):
        super(Predictor, self).__init__()
        
        # Init mappings
        self.mappings = nn.ModuleDict(mappings)
        
        # Should pre-trained Phi function be used                    
        self.phi = phi if phi is not None else nn.Identity()
                
        # Init predictor            
        self.predictor = MLP(
            latent_dim * len(self.mappings), 
            output_dim, 
            hidden_dims = [latent_dim,]
        )
                    
    def forward(self, X):
        
        H = [mapping(X[v]) for v, mapping in self.mappings.items()]
        Z = [self.phi(h) for h in H]
        h = torch.cat(Z, dim = 1)
        return self.predictor(h)