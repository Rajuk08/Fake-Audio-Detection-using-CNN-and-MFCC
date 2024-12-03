import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from preprocess import extract_features
import sounddevice as sd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_data(real_dir, fake_dir, duration=7):
    real_files = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.wav')]
    fake_files = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith('.wav')]
    
    X, y = [], []
    logging.info("Loading real audio files...")
    for file in real_files:
        mfccs = extract_features(file, duration=duration)
        X.append(mfccs)
        y.append(0)  
    
    logging.info("Loading fake audio files...")
    for file in fake_files:
        mfccs = extract_features(file, duration=duration)
        X.append(mfccs)
        y.append(1)  
    
    X = np.array(X)
    y = to_categorical(y, 2) 
    
    X = X / np.max(np.abs(X))
    return X, y

# CNN model
def build_model(input_shape):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# training history
def plot_training_history(history, save_path="training_history.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Training History")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Training history plot saved to {save_path}")

def record_audio(duration=5, sample_rate=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait() 
    print("Recording complete.")
    return audio, sample_rate


def detect_real_time(model_path='fake_audio_detector.h5'):
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Starting real-time detection...")
    duration = 5 
    audio, sample_rate = record_audio(duration=duration)

    temp_file = 'temp_audio.wav'
    sf.write(temp_file, audio, sample_rate)


    try:
        print("Processing audio...")
        features = extract_features(temp_file, duration=duration)
        features = features / np.max(np.abs(features)) 
        features = np.expand_dims(features, axis=-1)  # Channel dimension
        features = np.expand_dims(features, axis=0)   # Batch dimension

        print("Running prediction...")
        prediction = model.predict(features)
        real_confidence = prediction[0][0] * 100
        fake_confidence = prediction[0][1] * 100

        print("\n--- Detection Results ---")
        if real_confidence > fake_confidence:
            print(f"The audio is classified as **REAL** with {real_confidence:.2f}% confidence.")
        else:
            print(f"The audio is classified as **FAKE** with {fake_confidence:.2f}% confidence.")
        print("-------------------------\n")

    except Exception as e:
        print(f"Error during real-time detection: {e}")

# Main execution
if __name__ == '__main__':
    real_dir = 'data/real/' #put the path of  real file data 
    fake_dir = 'data/fake/' # put the path of  fake file data

    logging.info("Loading data...")
    X, y = load_data(real_dir, fake_dir, duration=7)

    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    logging.info("Building model...")
    model = build_model(input_shape)

    logging.info("Starting training...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=20, batch_size=32, callbacks=[early_stopping])

    logging.info("Training complete! Model Summary:")
    model.summary()

    logging.info("Saving model...")
    model.save('fake_audio_detector.h5')

    plot_training_history(history)
    logging.info("Training complete! Model and history saved.")

    logging.info("Evaluating the model on test data...")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    logging.info(f"Test Accuracy: {accuracy_score(y_test_classes, y_pred_classes):.2f}")
    print(classification_report(y_test_classes, y_pred_classes))

    print("\nWould you like to run real-time detection?")
    choice = input("Type 'yes' to continue or 'no' to exit: ").strip().lower()
    if choice == 'yes':
        detect_real_time(model_path='fake_audio_detector.h5')
