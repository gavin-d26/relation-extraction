import pandas as pd
import torch
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import spacy
import spacy_cleaner
from spacy_cleaner.processing import removers, replacers, mutators, transformers

torch.manual_seed(0)


# converts .csv file to train and val Dataframes. Also indexes the classes
def preprocess_raw_training_file(hw_csv_file):
    
    model = spacy.load("en_core_web_sm")
    pipeline = spacy_cleaner.Cleaner(
    model,
    removers.remove_number_token
    )
    
    df = pd.read_csv(hw_csv_file)
    df["UTTERANCES"]=pipeline.clean(df["UTTERANCES"].str.strip())
    df["UTTERANCES"]=df["UTTERANCES"].apply(lambda x: ''.join((item for item in x if not item.isdigit())))
    df["CORE RELATIONS"] = df["CORE RELATIONS"].str.split(" ")
    edf = df.explode('CORE RELATIONS')
    edf["values"]=str(1)
    pdf=edf.pivot(columns="CORE RELATIONS", index='ID', values='values').fillna(str(0))
    pdf['UTTERANCES'] = df['UTTERANCES']
    
    # reorder the columns
    pdf=pdf[['UTTERANCES', 'actor.gender', 'gr.amount', 'movie.country', 'movie.directed_by',
       'movie.estimated_budget', 'movie.genre', 'movie.gross_revenue',
       'movie.initial_release_date', 'movie.language', 'movie.locations',
       'movie.music', 'movie.produced_by', 'movie.production_companies',
       'movie.rating', 'movie.starring.actor', 'movie.starring.character',
       'movie.subjects', 'none', 'person.date_of_birth']]
    
    ID_TO_NAME = {k:v for k,v in enumerate(pdf.columns[1:])}
    X, Y = pdf.iloc[:, 0:1], pdf.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=0)
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_test, y_test], axis=1)
    return train_df, val_df, ID_TO_NAME


# fit a count vectorizer on text corpus
def make_vectorizer(hw_csv_file):
    df = pd.read_csv(hw_csv_file)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
    vectorizer.fit(df['UTTERANCES'].values)
    return vectorizer


# used to convert raw utterances to tensors for input to model (can be (B, embed_dim) or (B, S, embed_dim)!!)
def utterances_to_tensors(utterance_series, vectorizer):
    return torch.FloatTensor(vectorizer.transform(utterance_series.values).toarray())


# torch subclass for the dataset
class RelationExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, df, vectorizer):
        super().__init__()
        
        # preprocessing
        inputs = utterances_to_tensors(df['UTTERANCES'], vectorizer)
        targets = df.iloc[:, 1:].astype(int).values
        
        # convert to tensors
        self.inputs, self.targets = torch.FloatTensor(inputs), torch.FloatTensor(targets)
        
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    
    
    def __len__(self):
        return len(self.inputs)


# creates dataloaders for train and val datasets
def create_dataloaders(train_df, val_df, vectorizer, batch_size=32, ):
    train_dataset = RelationExtractionDataset(train_df, vectorizer)
    val_dataset = RelationExtractionDataset(val_df, vectorizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


        
        
        
        