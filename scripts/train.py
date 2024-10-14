from tqdm import tqdm
import torch
import torch.nn.functional as F
from .datatools import RelationExtractionDataset, create_dataloaders

torch.manual_seed(0)
# class to compute global/classwise multi-label accuracy
class MultilabelAccuracy():
    def __init__(self, classwise=False, threshold=0.):
        self.threshold=threshold
        self.preds=None
        self.targets=None
        self.classwise=classwise
    
    
    # append new batches    
    def update(self, preds, targets):
        self.preds = torch.concat((self.preds, preds.cpu()), dim=0) if self.preds is not None else preds.cpu()
        self.targets = torch.concat((self.targets, targets.cpu()), dim=0) if self.targets is not None else targets.cpu()
    
    
    # used to compute Multi-label Accuracy at the end of an epoch
    def compute(self):
        self.preds, self.targets = (self.preds>self.threshold).long(), self.targets.long()
        if self.classwise:
            result = (self.preds==self.targets).float().mean(dim=0).numpy()
        else:
            result = (self.preds==self.targets).all(dim=-1).float().mean()
        
        self.preds=self.targets=None
        return result


def balanced_loss_fn(preds, targets):
    num_targets = targets.sum(dim=0)
    discard_classes=num_targets!=0
    positive_score = len(targets)/(num_targets+1e-7)
    loss_classwise=F.binary_cross_entropy_with_logits(preds, targets, reduction='none', pos_weight=None).mean(dim=0)
    loss=loss_classwise[discard_classes].mean()
    return loss
    

# function to train the pytorch model        
def train_func(
    model,
    train_loader,
    val_loader,
    epochs,
    run_name="base-0_1",
    lr=3e-4,
    optimizer='adam',
    device='cpu'
               ):
    device = torch.device(device)
    model.to(device=device)
    
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    if optimizer=='adam':
        optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    
    train_acc = MultilabelAccuracy()
    val_acc = MultilabelAccuracy()
    val_acc_classwise = MultilabelAccuracy(classwise=True)
    
    print(f"Starting Training: {run_name}")
    
    train_acc_list=[]
    val_acc_list=[]
    
    for epoch in tqdm(range(epochs)):
        print(f"-------- Epoch {epoch} --------")
        
        train_loss=[]
        val_loss=[]
        
        # train on train set
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            preds = model(inputs)
            loss = balanced_loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_acc.update(preds.detach(), targets)
            train_loss.append(loss.detach().cpu())
        
        
        # evaluate on val set
        model.eval()
        with torch.inference_mode():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                preds = model(inputs)
                loss=loss_fn(preds, targets)
                val_loss.append(loss.cpu())
                val_acc.update(preds, targets)
                val_acc_classwise.update(preds, targets)
                
                
        metrics = {
            "train_loss":sum(train_loss)/len(train_loss),
            "train_acc":train_acc.compute(),
            "val_loss":sum(val_loss)/len(val_loss),
            "val_acc":val_acc.compute(),
            "val_acc_classwise":val_acc_classwise.compute(),
        }
        
        print(f'train_loss: {metrics["train_loss"]:.2f}   val_loss: {metrics["val_loss"]:.2f}   train_acc: {metrics["train_acc"]:.2f} \
                val_acc": {metrics["val_acc"]:.2f}')
        
        
        train_acc_list.append(metrics['train_acc'])
        val_acc_list.append(metrics['val_acc'])
    
    
    plot_accuracy(train_acc_list, val_acc_list, save_path='accuracy_plot.png')    
        
import matplotlib.pyplot as plt

def plot_accuracy(training_accuracy, validation_accuracy, save_path='accuracy_plot.png'):
    """
    Plots the training and validation accuracy per epoch and saves it as a PNG file.

    Parameters:
    - training_accuracy (list of float): List of training accuracies per epoch.
    - validation_accuracy (list of float): List of validation accuracies per epoch.
    - save_path (str): File path to save the plot. Default is 'accuracy_plot.png'.
    """
    if len(training_accuracy) != len(validation_accuracy):
        raise ValueError("Training and validation accuracy lists must have the same length.")

    epochs = range(0, len(training_accuracy))  # Epochs start from 1

    plt.figure(figsize=(8, 6))

    # Plotting both accuracies with distinct colors and markers
    plt.plot(epochs, training_accuracy, label='Training Accuracy', marker='o', color='blue')
    plt.plot(epochs, validation_accuracy, label='Validation Accuracy', marker='s', color='green')
    
    # Adding labels, title, and legend
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Baseline Model Training and Validation Accuracy per Epoch')
    plt.legend(loc='lower right')  # Legend placed at the bottom-right corner
    plt.grid(True)  # Enable grid for better visualization

    # Save the plot as a PNG file
    plt.savefig(save_path, format='png')
    plt.close()  # Close the plot to free up memory
    print(f"Plot saved as {save_path}")
        
        
        
        
            
            
            
        
            
            
            
    

