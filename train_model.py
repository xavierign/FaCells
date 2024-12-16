import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

def load_lstm_input(pickle_file):
    """
    Loads LSTM input data from a pickle file.
    
    Args:
        pickle_file (str): Path to the pickle file.
    
    Returns:
        tuple: (sequences, labels) - lists of sequences and corresponding labels.
    """
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    sequences, labels = zip(*data)
    return list(sequences), list(labels)

def analyze_sequence_lengths(pickle_file):
    """
    Analyzes and plots the distribution of sequence lengths.

    Args:
        pickle_file (str): Path to the pickle file containing the sequences and labels.
    """
    # Load the data from the pickle file
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the lengths of all sequences
    lengths = [len(sequence) for sequence, _ in data]

    # Print basic statistics
    print(f"Total number of sequences: {len(lengths)}")
    print(f"Minimum length: {np.min(lengths)}")
    print(f"Maximum length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")
    print(f"Standard deviation: {np.std(lengths):.2f}")

    # Plot the distribution of lengths
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def prepare_data(sequences, labels, max_length=8000):
    """
    Prepares sequences and labels for LSTM training.

    Args:
        sequences (list): List of sequences (variable-length lists of [x, y, l]).
        labels (list): List of labels (arrays or integers).
        max_length (int): Maximum allowed sequence length.

    Returns:
        tuple: (padded_sequences, mask_value, encoded_labels)
    """
    # Pad or truncate sequences to the specified max length
    mask_value = 0.0
    truncated_sequences = [seq[:max_length] for seq in sequences]
    padded_sequences = pad_sequences(truncated_sequences, padding='post', dtype='float32', value=mask_value)

    # Convert labels to a NumPy array (binary classification)
    label_array = np.array([1 if label == 1 else 0 for label in labels])

    return padded_sequences, mask_value, label_array


def build_lstm_model(input_shape, mask_value):
    """
    Builds an LSTM model for binary classification with variable-length input.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        mask_value (float): The value to mask during training.

    Returns:
        model: A compiled LSTM model.
    """
    #model = Sequential([
    #    Masking(mask_value=mask_value, input_shape=input_shape),
    #    LSTM(64),
    #    Dense(1, activation='sigmoid')  # Single output for binary classification
    #])
    
    model = Sequential([
        Masking(mask_value=mask_value, input_shape=input_shape),
        Bidirectional(LSTM(150, return_sequences=True)),
        Bidirectional(LSTM(150, return_sequences=True)),
        Bidirectional(LSTM(150)),
        Dense(1, activation='sigmoid')  # Binary classification
    ])


    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model
# Main script
if __name__ == "__main__":
    # Path to the pickle file
    pickle_file = "ml_data/lstm_input_data.pkl"

    #analyze_sequence_lengths(pickle_file)
    # Load the data
    sequences, labels = load_lstm_input(pickle_file)

    # Prepare the data
    padded_sequences, mask_value, label_matrix = prepare_data(sequences, labels)

    # Build the model
    
    input_shape = (padded_sequences.shape[1], padded_sequences.shape[2])  # (timesteps, features)
    model = build_lstm_model(input_shape, mask_value)

    # Print model summary
    model.summary()

    # Train the model
    history = model.fit(
        padded_sequences,
        label_matrix,
        epochs=10,
        batch_size=32,
        validation_split=0.2
    )

    # Save the trained model
    model.save("lstm_model.h5")
    print("Model saved to 'lstm_model.h5'")