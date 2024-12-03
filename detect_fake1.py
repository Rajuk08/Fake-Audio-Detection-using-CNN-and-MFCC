from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from preprocess import extract_features
import os

def preprocess_and_expand(audio_path):
    mfccs = extract_features(audio_path) 
    mfccs = np.expand_dims(mfccs, axis=0) 
    mfccs = np.expand_dims(mfccs, axis=-1) 
    return mfccs

def visualize_confidence(real_prob, fake_prob, save_path="confidence_scores.png"):
    """
    Visualizes the confidence scores as a bar chart.
    """
    labels = ['Real', 'Fake']
    scores = [real_prob, fake_prob]
    colors = ['green', 'red']
    plt.bar(labels, scores, color=colors)
    plt.title('Confidence Scores')
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)
    plt.savefig(save_path)
    plt.close()
    print(f"Confidence visualization saved at {save_path}")

def detect_fake_audio(audio_path, model_path='fake_audio_detector.h5', visualize=False):
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    mfccs = preprocess_and_expand(audio_path)

    prediction = model.predict(mfccs)
    fake_prob = prediction[0][1] * 100 
    real_prob = prediction[0][0] * 100 

    print(f"Real: {real_prob:.2f}%")
    print(f"Fake: {fake_prob:.2f}%")
    result = 'Fake' if np.argmax(prediction) == 1 else 'Real'
    if visualize:
        visualize_confidence(real_prob, fake_prob)
    
    return result, real_prob, fake_prob

if __name__ == "__main__":
    audio_path = input("Please enter the path to the audio file you want to analyze: ")
    visualize = input("Would you like to visualize the confidence scores? (yes/no): ").strip().lower() == "yes"
    
    try:
        result, real_prob, fake_prob = detect_fake_audio(audio_path, visualize=visualize)
        print(f"The audio is classified as: {result}")
        print(f"Confidence - Real: {real_prob:.2f}%, Fake: {fake_prob:.2f}%")
    except Exception as e:
        print(f"Error processing the file: {e}")
