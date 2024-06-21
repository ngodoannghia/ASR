import os
import hydra
from omegaconf import DictConfig
from utils import preprocess_audio, save_processed_audio, ensure_dir

@hydra.main(config_path="../configs", config_name="config")
def preprocess(cfg: DictConfig):
    raw_data_path = cfg.data.raw_data_path
    processed_data_path = cfg.data.processed_data_path

    ensure_dir(processed_data_path)

    for file_name in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file_name)
        audio, sample_rate = preprocess_audio(file_path)
        processed_file_path = os.path.join(processed_data_path, file_name)
        save_processed_audio(audio, sample_rate, processed_file_path)

if __name__ == "__main__":
    preprocess()