from transformers import BertTokenizer

from .datautils import DataCard
from .modelutils import *



def make_mappings(datacard, output_dim, num_hidden_dims = []):
    # Mappings
    mappings = {}
        
    # Image mapping - reutilize single mapping    
    if len(datacard.img_vars) > 0:                    
        # Init mapping
        mapping = ImageMapping(output_dim = output_dim)
        
        # Add to mappings
        for v in datacard.img_vars:
            if v != datacard.response:
                mappings[v] = mapping
            
    # Txt mapping - reutilize single mapping    
    if len(datacard.txt_vars) > 0:        
        # Init mapping
        mapping = TextMapping(output_dim = output_dim)
        
        # Add to mappings
        for v in datacard.txt_vars:
            if v != datacard.response:
                mappings[v] = mapping
        
        # Init tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = None

    # Cat mappings            
    for v in datacard.cat_vars:
        if v != datacard.response:
            num = datacard.cat_levels[v]        
            mappings[v] = CatMapping(num, output_dim)
                
    # Num mappings
    for v in datacard.num_vars:              
        if v != datacard.response:
            mappings[v] = MLP(1, output_dim, num_hidden_dims)
        
    return mappings, tokenizer    


def make_embedder(model_type: str, 
                  datacard: DataCard,
                  latent_dim: int, 
                  hidden_dim: int,  
                  phi_dims: list, 
                  no_anchors: int = 10):
                        
    # Assemble the model
    if model_type == 'CLIP':
        
        # Initiate mappings, and tokenizer (if applicable)
        mappings, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=latent_dim,
            num_hidden_dims=[latent_dim,]
        )        
                                
        # Init CLIP model
        model = CLIPEmbedder(mappings = mappings)                
        
    elif model_type == 'GMC':
        
        # Initiate mappings, and tokenizer (if applicable)
        mappings, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=hidden_dim,
            num_hidden_dims=[hidden_dim,]
        )         
                        
        # Create list of aux. mappings
        aux_maps, _ = make_mappings(
            datacard= datacard, 
            output_dim = hidden_dim, 
            num_hidden_dims = [hidden_dim,]
        )
        
        # Init GMC model
        model = GMCEmbedder(
            mappings = mappings, 
            aux_maps = aux_maps, 
            hidden_dim = hidden_dim, 
            latent_dim = latent_dim
        )                
    
    elif model_type == 'MCN':    
        
        # Initiate mappings, and tokenizer (if applicable)
        mappings, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=latent_dim,
            num_hidden_dims=[latent_dim,]
        )               
        
        # Init MCN model
        model = MCNEmbedder(
            mappings = mappings,
            no_anchors = no_anchors,
            latent_dim = latent_dim
        )
        
    elif model_type == 'SCARF':
        
        # Initiate mappings, and tokenizer (if applicable)
        mappings, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=latent_dim,
            num_hidden_dims=[latent_dim,]
        )            
                        
        # Init ICEH model   
        model = SCARFEmbedder(
            mappings = mappings, 
            latent_dim = latent_dim
        )    
            
    elif model_type == 'SUBTAB':
        
        # Initiate mappings, and tokenizer (if applicable)
        mappings, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=latent_dim,
            num_hidden_dims=[latent_dim,]
        )        
                                
        # Init SUBTAB model
        model = SUBTABEmbedder(
            mappings = mappings,
            latent_dim = latent_dim
        )
                 
    elif model_type == 'ICEH':

        # Initiate mappings, and tokenizer (if applicable)
        mappings, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=hidden_dim,
            num_hidden_dims=[hidden_dim,]
        )            
                
        # Init ICEH model   
        model = ICETEmbedder(
            mappings = mappings, 
            phi_dims = phi_dims,
            hidden_dim = hidden_dim,
            latent_dim = latent_dim
        )         
        
    elif model_type == 'CONTROL':

        # Initiate mappings, and tokenizer (if applicable)
        _, tokenizer = make_mappings(
            datacard=datacard,
            output_dim=hidden_dim,
            num_hidden_dims=[hidden_dim,]
        )            
                
        # Return none
        model = None          
        
    else:
        print('Wrong model type: {}'.format(model_type))
        raise ValueError()
        
    if model is not None:        
        for param in model.parameters():
            param.requires_grad = True     
                
    return model, tokenizer


def make_predictor(embedder: nn.Module, datacard: DataCard, latent_dim: int):
    
    # Set output dim
    if datacard.response in datacard.cat_vars:
        output_dim = datacard.cat_levels[datacard.response]
    else:
        output_dim = 1
    
    if embedder is not None:
    
        # Set phi
        phi = embedder.phi if hasattr(embedder, 'phi') else None
        
        # Init model
        model = Predictor(
            mappings=embedder.mappings,   
            latent_dim = latent_dim,         
            output_dim = output_dim,        
            phi = phi
        )
        
    else:
        
        mappings, _ = make_mappings(
            datacard=datacard, 
            output_dim=latent_dim, 
            num_hidden_dims=[latent_dim,]
        ) 
        
        # Init model
        model = Predictor(
            mappings = mappings,   
            latent_dim = latent_dim,         
            output_dim = output_dim,        
            phi = None
        )           
    
    # Make sure all wigths are trainable
    if model is not None:        
        for param in model.parameters():
            param.requires_grad = True     
        
    return model
