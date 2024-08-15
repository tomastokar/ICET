import time
import yaml
import pandas as pd


from utils.auxutils import set_device, count_params, construct_parser, collate_func, make_loaders
from utils.makeutils import make_embedder, make_predictor
from utils.datautils import DataCard, load_datasets
from utils.trainutils import MMCRLTrainer, PredictorTrainer
from utils.testutils import TransferTester, ShallowTester


OUTPUT_COLUMNS = [
    'replicate',
    'batch_size',
    'learning_rate',          
    'latent_dim',    
    'model_type', 
    'embedder_params', 
    'embedder_train_loss', 
    'embedder_valid_loss', 
    'kmeans_perf',
    'knn_perf',  
    'predictor_params',  
    'predictor_train_loss', 
    'predictor_valid_loss',     
    'predictor_perf'
]

GRP_COLS = ['batch_size', 'learning_rate', 'latent_dim', 'model_type', 'embedder_params', 'predictor_params']
RES_COLS = ['kmeans_perf', 'knn_perf', 'predictor_perf']

def main(args):
    
    # Load config files
    with open('./src/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)      
        config = config[args.dataset]    
    
    # Create datacard    
    with open('./src/datacards.yml', 'r') as f:
        datacard = yaml.load(f, Loader=yaml.FullLoader)      
        datacard = datacard[args.dataset]    
        datacard = DataCard(**datacard)
    
    # Set device
    device = set_device(config)
                
    # Output names
    output_columns =  OUTPUT_COLUMNS.copy()
    output_columns += [v for v in datacard.cat_vars if v != datacard.response]
    output_columns += [v for v in datacard.num_vars if v != datacard.response]
        
    # Init results data frame
    results = pd.DataFrame([], columns = output_columns)
    
    # Browse hyper-params
    start = time.time()  
    
    for replicate in range(config['NO_REPLICATES']):
    
        print('\n\t===== Loading {} dataset =====\n'.format(args.dataset))
    
        # Load dataset
        datasets = load_datasets(args.dataset)

        # ------------
        # Experiments
        # ------------
        for batch_size in config['BATCH_SIZES']: 
                            
            # Collator
            collator = lambda x: collate_func(x, datacard)
            
            # Build data loaders        
            data_loaders = make_loaders(
                datasets=datasets, 
                collator=collator, 
                batch_size=batch_size,
                num_workers=config['N_WORKERS']
            )
            
            for learning_rate in config['LEARNING_RATES']:                                                           
                for latent_dim in config['LATENT_DIMS']:
                    for model_type in config['MODEL_TYPES']:
                                               
                        # Placeholder for missing results
                        embedder_params = None
                        embedder_train_loss = None
                        embedder_valid_loss = None     
                        kmeans_perf = None
                        knn_perf = None
                        accs = [None for v in datacard.cat_vars if v != datacard.response]
                        errs = [None for v in datacard.num_vars if v != datacard.response]
                        predictor_params = None
                        predictor_train_loss = None
                        predictor_valid_loss = None
                        predictor_perf = None

                        
                        # Init embedding model                         
                        embedder, tokenizer = make_embedder(
                            model_type = model_type,
                            datacard = datacard,
                            latent_dim = latent_dim, 
                            hidden_dim = config['HIDDEN_DIM'],
                            phi_dims = config['PHI_DIMS'],
                            no_anchors = max(2, batch_size // 10)
                        )

                        
                        if model_type != 'CONTROL':
                            
                            # ---------------
                            # Train embedder
                            # ---------------
                                                                                                                                
                            # Calc number of model params
                            embedder_params = count_params(embedder)
                                                                                                                            
                            # Init trainer
                            trainer = MMCRLTrainer(
                                model=embedder, 
                                datacard=datacard,
                                data_loaders=data_loaders,
                                learning_rate=learning_rate,                             
                                tokenizer=tokenizer,
                                device = device,
                                verbosity=config['VERBOSITY'],
                                patience=5,
                                checkpoint='./src/{}.pt'.format(args.dataset)
                            )        
                            
                            # Train model
                            train_track, valid_track = trainer.train(epochs = config['EPOCHS']) 
                            
                            # Add errors to results
                            embedder_train_loss = train_track[-1]
                            embedder_valid_loss = valid_track[-1]
                                                        
                            # ---------------------------
                            # Testing embeddings quality
                            # ---------------------------
                            
                            if datacard.response is not None:
                                
                                # Extract features and assoc. targets
                                features, targets = trainer.embed_data()

                                # Shallow tester
                                shallow_tester = ShallowTester(
                                    datacard,
                                    features,
                                    targets
                                )                        
                                
                                # Test clusterins
                                kmeans_perf, knn_perf = shallow_tester.test()
                                                        
                            # --------------------------                    
                            # Testing modality transfer
                            # --------------------------
                                                                                    
                            # Check if model supports modality transfer
                            transfer = getattr(embedder, "transfer", None)
                            
                            if callable(transfer):
                                # Init transfer tester
                                transfer_tester = TransferTester(
                                    embedder, 
                                    datacard=datacard,
                                    data_loaders=data_loaders,
                                    tokenizer=tokenizer,
                                    device = device 
                                )        
                        
                                # Test model 
                                accs, errs = transfer_tester.test()
                                
                                # To list
                                accs = [acc for acc in accs.values()]
                                errs = [err for err in errs.values()]
     
                        # --------------------------
                        # Transfer learning testing
                        # --------------------------
                        if datacard.response is not None:
                                                                                                                
                            # Init predictor
                            predictor = make_predictor(
                                embedder=embedder,
                                datacard=datacard, 
                                latent_dim=latent_dim
                            )
                            
                            # Count parameters
                            predictor_params = count_params(predictor)
                                                        
                            # Init trainer for predictor
                            trainer = PredictorTrainer(
                                model=predictor, 
                                datacard=datacard,
                                data_loaders=data_loaders,
                                learning_rate=learning_rate,
                                tokenizer=tokenizer,
                                device=device,
                                verbosity=config['VERBOSITY'],
                                checkpoint='./src/{}.pt'.format(args.dataset)
                            )
                            
                            # Train predictor
                            train_track, valid_track = trainer.train(epochs=config['EPOCHS']) # Fixed maximum number of epochs
                            
                            # Add errors to results
                            predictor_train_loss = train_track[-1]
                            predictor_valid_loss = valid_track[-1]
                                                        
                            # Test performance
                            predictor_perf = trainer.test()
                            

                        # Assemble new results line
                        result = [
                            replicate, batch_size, learning_rate, latent_dim, model_type, 
                            embedder_params, embedder_train_loss, embedder_valid_loss, 
                            kmeans_perf, knn_perf,  
                            predictor_params, predictor_train_loss, predictor_valid_loss, predictor_perf
                        ]
                        
                        result += accs
                        result += errs
                                                                                         
                        # Add to results
                        results.loc[len(results)] = result

                        # Report
                        print('\n')
                        print(results.iloc[-1])
                        print('\n\tElapsed time: {:1.1f} seconds!\n'.format(time.time() - start))
                        
                        # Save to file
                        results.to_csv('./results/{}_experiment.csv'.format(args.dataset))

    # ------------------------------------
    # Aggregate results across replicates 
    # ------------------------------------
    # Read the obtained results
    results = pd.read_csv('./results/{}_experiment.csv'.format(args.dataset), index_col = 0)
    
    # The columns of the imputed variables (following OUTPUT_COLUMNS)
    imputation_cols = results.columns[14:].tolist() 
    
    # Aggregate by mean
    results = (
        results
        .groupby(GRP_COLS, dropna = False)[RES_COLS + imputation_cols]
        .mean()
        .reset_index()
    )
    
    # Write to file
    results.to_csv('./results/{}_experiment.csv'.format(args.dataset))
    
if __name__ == '__main__':  
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
