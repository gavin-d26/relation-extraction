import argparse
import random
import numpy as np
import torch
import scripts.configs as configs
import scripts.datatools as datatools
import scripts.models as models
import scripts.train as train
import scripts.predict as predict

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_csv_path')
    parser.add_argument('test_csv_path')
    parser.add_argument('submission_csv_path')
    
    args = parser.parse_args()
    
    # parse script args
    train_csv_path=args.train_csv_path
    test_csv_path=args.test_csv_path
    submission_csv_path=args.submission_csv_path
    
    # get train and val dataframes
    train_df, val_df, ID_TO_NAME = datatools.preprocess_raw_training_file(train_csv_path)
    
    # create vectorizer and dataloaders
    vectorizer = datatools.make_vectorizer(train_csv_path)
    train_loader, val_loader = datatools.create_dataloaders(train_df, val_df, vectorizer, configs.batch_size)
    
    # initialize model
    model = models.RelationClassifierPro(train_loader.dataset[0][0].shape[0], dropout=configs.dropout)
    
    train.train_func(
        model,
        train_loader,
        val_loader,
        epochs=configs.epochs,
        run_name=configs.run_name,
        lr=configs.lr,
        optimizer=configs.optimizer,
        device=configs.device,
        damping_factor=configs.damping_factor,
        weight_decay=configs.weight_decay,
    )
    
    predict.make_submission_file(model,
                                 vectorizer,
                                 test_csv_path, 
                                 ID_TO_NAME, 
                                 save_submission_file_path=submission_csv_path, 
                                 threshold=0., 
                                 device=configs.device)
    
    
    predict.make_validation_file(model,
                                 vectorizer,
                                 val_df, 
                                 ID_TO_NAME, 
                                 threshold=0., 
                                 device=configs.device)
    
    
if __name__=="__main__":
    main()
    
    