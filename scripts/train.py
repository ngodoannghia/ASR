import os
import wandb
import hydra
import torch
from omegaconf import DictConfig
from models.wav2vec_model import Wav2VecModel, load_model, save_model
from transformers import Wav2Vec2Processor
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW

def load_data(processed_data_path):
    dataset = load_dataset('csv', data_files={'train': os.path.join(processed_data_path, 'train.csv'),
                                              'val': os.path.join(processed_data_path, 'val.csv'),
                                              'test': os.path.join(processed_data_path, 'test.csv')})
    return dataset

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    # Check device (CPU hoặc GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # create and load model from checkpoint
    if cfg.model.checkpoint_path:
        model = load_model(cfg.model.checkpoint_path, cfg.model.pretrained_model_name)
    else:
        model = Wav2VecModel(cfg.model.pretrained_model_name)
    
    model.to(device)
    
    # Initialize wandb
    wandb.init(project=cfg.wandb.project_name, entity=cfg.wandb.entity)
    
    # Load dataset
    dataset = load_data(cfg.data.processed_data_path)
    
    # Prepare data loaders
    train_loader = DataLoader(dataset['train'], batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val'], batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(dataset['test'], batch_size=cfg.training.batch_size, shuffle=False)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training loop
    model.train()
    
    best_wer = float("inf")
    
    for epoch in range(cfg.training.epochs):
        for batch in train_loader:
            inputs = model.processor(batch['audio'], sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            labels = model.processor(batch['sentence'], return_tensors="pt", padding=True).to(device).input_ids
            outputs = model(input_values=inputs.input_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Logging to wandb```python
            wandb.log({"loss": loss.item()})
            
            # Evaluate WER on validation set after each epoch
            model.eval()
            all_predictions = []
            all_references = []
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = model.processor(batch['audio'], sampling_rate=16000, return_tensors="pt", padding=True)
                    labels = model.processor(batch['sentence'], return_tensors="pt", padding=True).input_ids
                    
                    predictions = model.transcribe(inputs)
                    references = model.processor.batch_decode(labels, group_tokens=False)
                    
                    all_predictions.extend(predictions)
                    all_references.extend(references)
                    
            # Compute the WER
            wer_score = model.compute_wer(all_predictions, all_references)
            print(f"Epoch {epoch + 1}, Word Error Rate (WER): {wer_score}")
            
            # Save the model if WER improves
            if wer_score < best_wer:
                best_wer = wer_score
                save_model(model, f"{cfg.training.model_checkpoint_path}/best_model.pt")
                print(f"New best model saved with WER: {wer_score}")
            
            if epoch >= cfg.training.epochs - 5 and epoch != cfg.training.epochs - 1:
                save_model(model, f"{cfg.training.model_checkpoint_path}/model_{epoch}.pt")
                
            del all_predictions, all_references
                
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}, Loss: {loss.item()}")
        
    # Final save of the model
    final_model_path = os.path.join(cfg.training.model_checkpoint_path, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    wandb.save(final_model_path)

    wandb.finish()

if __name__ == "__main__":
    train()