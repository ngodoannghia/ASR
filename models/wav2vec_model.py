import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import wer

class Wav2VecModel(nn.Module):
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base-960h"):
        super(Wav2VecModel, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_name)

    def forward(self, input_values, labels=None):
        # Forward pass through the model
        outputs = self.model(input_values, labels=labels)
        return outputs

    def transcribe(self, audio, sampling_rate):
        # Process the audio and make predictions
        input_values = self.processor(audio, return_tensors="pt", sampling_rate=sampling_rate).input_values
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription
    
    def transcribe(self, input_values):
        logits = self.model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        return transcription
    
    def compute_wer(self, predicted_texts, reference_texts):
        return wer(reference_texts, predicted_texts)

# Function to load the model
def load_model(checkpoint_path=None, pretrained_model_name="facebook/wav2vec2-base-960h"):
    model = Wav2VecModel(pretrained_model_name)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
    return model

# Function to save the model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)