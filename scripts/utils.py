import os
import librosa
import soundfile as sf

def preprocess_audio(file_path, target_sample_rate=16000):
    """
    Preprocess an audio file by loading it and resampling to the target sample rate.

    Args:
        file_path (str): Path to the audio file.
        target_sample_rate (int): Target sample rate for the audio.

    Returns:
        tuple: Processed audio data and sample rate.
    """
    audio, sample_rate = librosa.load(file_path, sr=target_sample_rate)
    return audio, sample_rate

def save_processed_audio(audio, sample_rate, output_path):
    """
    Save the processed audio data to a file.

    Args:
        audio (np.ndarray): Audio data.
        sample_rate (int): Sample rate of the audio data.
        output_path (str): Path to save the processed audio file.
    """
    sf.write(output_path, audio, sample_rate)

def ensure_dir(directory):
    """
    Ensure that a directory exists. If it does not exist, create it.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)