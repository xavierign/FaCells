import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def prepare_lstm_input(np_dir, attributes_file, attribute = 'Male',output_pickle = "ml_data/lstm_input_data.pkl"):
    """
    Processes .npy files and attribute annotations to prepare LSTM input with variable-length sequences.
    
    Args:
        np_dir (str): Path to the directory containing .npy files with [x, y, l] sequences.
        attributes_file (str): Path to the attribute annotation file.

    Returns:
        list of tuples: Each tuple contains (sequence, label), where:
                        - sequence: A NumPy array of [x, y, l] points.
                        - label: A NumPy array of corresponding attributes.
    """
    # Load the attributes file
    attributes_df = pd.read_csv(attributes_file, sep=r'\s+', skiprows=1, header=0)
    
    # List to store sequences and labels
    data = []

    # Iterate through all .npy files in the directory
    for file in os.listdir(np_dir):
        if file.endswith(".npy"):
            file_path = os.path.join(np_dir, file)
            sequence = np.load(file_path, allow_pickle=True)
            
            att_index = file.split('_')[0]
            # Ensure the data is in the expected format

            if isinstance(sequence, np.ndarray):
                # Get the corresponding label from the attributes file
                base_name = os.path.splitext(file)[0]


                label = attributes_df.loc[att_index + '.jpg', attribute]
                data.append((np.array(sequence), label))

    # Save the data as a pickle file
    with open(output_pickle, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_pickle}")

    return data


# Example usage
if __name__ == "__main__":
    np_directory = "/Users/xaviergonzalez/Documents/repos/FaCells/CelebAMask-HQ/CelebA-HQ-np/"
    attributes_file = "/Users/xaviergonzalez/Documents/repos/FaCells/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"
    
    data = prepare_lstm_input(np_directory, attributes_file)
    
    # Print the first sequence and label
    for i, (sequence, label) in enumerate(data[:3]):
        print(f"Sample {i + 1}:")
        print(f"Sequence: {sequence}")
        print(f"Label: {label}\n")