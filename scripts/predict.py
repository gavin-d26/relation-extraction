import pandas as pd
import torch
from .datatools import utterances_to_tensors

torch.manual_seed(0)
# converts raw model predictions to a series of text names
def preds_array_to_series(preds, ids_to_names):
    preds= preds.tolist()
    series = pd.Series(preds)
    
    def convert_to_names(plist, ids_to_names):
        plist = [ids_to_names[i] for i,item in enumerate(plist) if item!=0]
        if "none" in plist or plist==[]:
            plist=['none']
        return " ".join(plist)
        
    series= series.apply(lambda x: convert_to_names(x, ids_to_names))
    return series


# func to create a .csv file for kaggle submission
def make_submission_file(model, vectorizer, hw_test_csv_path, ids_to_names, save_submission_file_path="submission.csv", threshold=0., device='cpu'):
    device=torch.device(device)
    model.to(device)
    df = pd.read_csv(hw_test_csv_path, index_col="ID")
    
    inputs = utterances_to_tensors(df['UTTERANCES'], vectorizer)
    
    model.eval()
    with torch.inference_mode():
        preds = model(inputs.to(device)).cpu()
    
    preds = (preds>threshold).long().numpy()
    df["Core Relations"] = preds_array_to_series(preds, ids_to_names)
    df=df.drop(columns=['UTTERANCES'])
    df.to_csv(save_submission_file_path)
    
    
def make_validation_file(model, vectorizer, val_df, ids_to_names, save_validation_file_path="validation.csv", threshold=0., device='cpu'):
    device=torch.device(device)
    model.to(device)
    
    # clean sentences
    val_df['UTTERANCES']=clean_utterance_text(val_df['UTTERANCES'])
    
    # convert sentences to tensors
    inputs = utterances_to_tensors(val_df['UTTERANCES'], vectorizer)
    
    model.eval()
    with torch.inference_mode():
        preds = model(inputs.to(device)).cpu()
    
    preds = (preds>threshold).long().numpy()
    val_df.to_csv("./data/val_df.csv")
    val_df.iloc[:, 1:] = preds
    val_df.to_csv(save_validation_file_path)
    