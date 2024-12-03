import librosa
import numpy as np
import logging
import matplotlib.pyplot as plt
import librosa.display 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def extract_features(audio_path, sr=22050, duration=7, n_mfcc=40):
   
    try:
        logging.info(f"Loading audio file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        target_length = sr * duration
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        logging.info(f"Extracted MFCCs with shape: {mfccs.shape}")

        return mfccs
    except Exception as e:
        logging.error(f"Error processing audio file {audio_path}: {e}")
        raise

def visualize_features(mfccs, save_path=None):
   
    try:
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCCs')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Feature visualization saved to {save_path}")
        else:
            plt.show()
    except Exception as e:
        logging.error(f"Error visualizing features: {e}")

if __name__ == "__main__":
    test_audio_path = 'data/real/file21.wav' 
    
    try:
    
        mfcc_features = extract_features(test_audio_path, duration=7, n_mfcc=40)
    
        visualize_features(mfcc_features, save_path='mfcc_visualization.png')
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
