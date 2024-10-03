from abc import ABC, abstractmethod
import chunk
import os
from pkgutil import ImpImporter
import numpy as np
from sklearn.model_selection import train_test_split
import cairosvg
from PIL import Image
import io
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from itertools import islice
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from collections import defaultdict

class CelebADrawingsClassifier(ABC):
    """Abstract class to classify drawings from the images of the CelebA dataset."""
    def __init__(self, drawings, attributes_file, attribute_for_classification, chunk_size=100, train_percentage=0.75):
        self._drawings = drawings
        self._attributes_file = attributes_file
        self._chunk_size = chunk_size
        self._train_percentage = train_percentage # maybe not necessary to store it 
        self._attribute_names = []
        self._attribute_for_classification_idx = self._set_attribute_for_classification(attribute_for_classification)
        self._drawing_attribute_value_data = [] # list of tuples (n, v) where n is name of drawing, v is attribute value for it
        self._drawing_attribute_value_dict = {} # keys are the indexes of the drawings and values are the attribute value
        self._populate_drawing_attribute_value()
        self.train_data = []
        self.test_data = []
        self._split_train_test()

    def _populate_drawing_attribute_value(self):
        with open(self._attributes_file, 'r') as f:
            lines = f.readlines()

            # skipping the first two rows because: 
            # the first row of the file is the number of images in the dataset and
            # the second row of the file has the names of the attributes
            for i, line in enumerate(lines[2:]):
                parts = line.strip().split()
                # the first element is the name of the image with extension
                name = os.path.splitext(parts[0])[0] 
                attribute_value = parts[self._attribute_for_classification_idx] # string that is either 1 or -1
                self._drawing_attribute_value_data += [(name, attribute_value)]
                self._drawing_attribute_value_dict[i] = attribute_value

    def _set_attribute_for_classification(self, name):
        # first populate attribute_names with the values in the attributes_file
        with open(self._attributes_file, 'r') as f:
            lines = f.readlines()
            self._attribute_names = lines[1].strip().split()

        if name not in self._attribute_names:
            raise ValueError(f"{name} not found in attributes list.")
        
        # adding 1 to index value because first element in the rows of the attributes file is the image name
        return self._attribute_names.index(name) + 1  

    def _svg_to_numpy_array(self, drawing_path):
        """Returns numpy array with shape (218, 178, 4)"""
        with open(drawing_path, 'r') as file:
            svg = file.read()
        png = cairosvg.svg2png(bytestring=svg)
        drawing = Image.open(io.BytesIO(png))

        return np.array(drawing)

    def _split_train_test(self):
        self.train_data, self.test_data = train_test_split(
            self._drawing_attribute_value_data, 
            train_size=self._train_percentage, 
            stratify=[value for _, value in self._drawing_attribute_value_data]
        )

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def show_results(self, pred_y, true_y):
        accuracy = accuracy_score(true_y, pred_y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(true_y, pred_y))
        print("\nClassification Report:")
        print(classification_report(true_y, pred_y))
        

class CelebAIncrementalClassifier(CelebADrawingsClassifier):
    def __init__(self, drawings_directory: str, attributes_file: str, attribute_for_classification: str, chunk_size: int = 100):
        super().__init__(drawings_directory, attributes_file, attribute_for_classification, chunk_size)
        self.model = SGDClassifier()

    def _chunks(self, data):
        iterator = iter(data)
        while True:
            chunk = list(islice(iterator, self._chunk_size))
            if not chunk:
                break
            yield chunk

    def _batch_data_generator(self, data):
        print("starting batch data generator")
        for chunk in self._chunks(data):
            batch_X = []
            batch_y = []

            for name, attribute_value in chunk:
                drawing_path = os.path.join(self._drawings, name + '_lines.svg')
                drawing_array = self._svg_to_numpy_array(drawing_path)
                batch_X.append(drawing_array)
                batch_y.append(int(attribute_value))
            
            batch_X = np.array(batch_X).reshape(len(batch_X), -1)
            batch_y = np.array(batch_y)
            yield batch_X, batch_y

    def train(self):
        first_batch = True
        i = 1
        for batch_X, batch_y in self._batch_data_generator(self.train_data):
            if first_batch: # for the first call, initialize partial_fit
                self.model.partial_fit(batch_X, batch_y, classes=np.array([-1, 1]))
                first_batch = False
                print("first batch done")
            else:
                self.model.partial_fit(batch_X, batch_y)
                if i % 100 == 0:
                    print(f"processed batch number {i}...")
                i += 1

    def evaluate(self):
        print("evaluating")
        total_true_y = []
        total_pred_y = []
        i = 0
        for batch_X, batch_y in self._batch_data_generator(self.test_data):
            pred_y = self.model.predict(batch_X)
            if i % 100 == 0:
                print(f"already predicted batch {i}")
            i += 1
            total_true_y.extend(batch_y)
            total_pred_y.extend(pred_y)
        
        self.show_results(total_pred_y, total_true_y)


class CelebALSTMClassifier(CelebADrawingsClassifier):
    def __init__(self, drawings_file: str, attributes_file: str, attribute_for_classification: str):
        super().__init__(drawings_file, attributes_file, attribute_for_classification)
        self._drawing_length_dict = defaultdict(list)
        self.model = self._build_model()
        self._generate_drawing_length_dict()

    def _build_model(self):
        model = Sequential([
            LSTM(64, input_shape=(None, 4), return_sequences=False), 
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _generate_drawing_length_dict(self):
        """Populates a dictionary with length of drawing (number of points) as keys and the values are the indexes 
        of the rows (drawings) that have that length."""
        with open(self._drawings, 'r') as file:
            for idx, drawing in enumerate(file):
                points = drawing.strip().split('),(')
                length = len(points)
                self._drawing_length_dict[length].append(idx)

    def _batch_data_generator(self, data):
        """For each of the lengths in the internal dictionary, generate a batch with the drawing of said length."""
        lengths = list(self._drawing_length_dict.keys())

        with open(self._drawings, 'r') as file:
            drawings = file.readlines()

        for length in lengths:
            idxs = self._drawing_length_dict[length]
            batch_X = []
            batch_y = []
            print(f"length {length}")

            for i in idxs:
                print(f"idx {i}")
                drawing_raw = drawings[i].strip()
                drawing = np.array(eval(f"[{drawing_raw}]"))
                batch_X.append(drawing)

                attribute_value = int(self._drawing_attribute_value_dict[i])
                batch_y.append(attribute_value)
            
            batch_X = np.array(batch_X).reshape(len(batch_X), -1)
            batch_y = np.array(batch_y)

            yield batch_X, batch_y

    def train(self, epochs=10):
        for epoch in range(epochs):
            print(f"epoch {epoch}")
            for batch_X, batch_y in self._batch_data_generator(self.train_data):
                print(f"fit")
                self.model.fit(batch_X, batch_y) 

    def evaluate(self):
        print("evaluating")
        total_true_y = []
        total_pred_y = []
        i = 0
        for batch_X, batch_y in self._batch_data_generator(self.test_data):
            pred_y = self.model.predict(batch_X)
            if i % 100 == 0:
                print(f"already predicted batch {i}")
            i += 1
            total_true_y.extend(batch_y)
            total_pred_y.extend(pred_y)
        
        self.show_results(total_pred_y, total_true_y)


def check_classes_balance(attributes_file):
    with open(attributes_file, 'r') as file:
        lines = file.readlines()

    attribute_names = lines[1].strip().split()
    classes_balances = []

    for attribute_index, attribute_name in enumerate(attribute_names):
        count_1 = 0
        count_minus_1 = 0

        for line in lines[2:]:
            parts = line.strip().split()
            attribute_value = int(parts[attribute_index + 1])  
            
            if attribute_value == 1:
                count_1 += 1
            elif attribute_value == -1:
                count_minus_1 += 1

        # Calculate total images and the balance ratio
        total = count_1 + count_minus_1
        balance_ratio = min(count_1, count_minus_1) / total

        classes_balances.append({
            "attribute": attribute_name, 
            "class_1_count": count_1,
            "class_minus_1_count": count_minus_1,
            "balance_ratio": balance_ratio,
            "balance_percentage": balance_ratio * 100
        })

    return classes_balances

if __name__ == "__main__":
    # Checking which are the most balanced classes
    # classes_balances = check_classes_balance('../CelebA/Anno/list_attr_celeba.txt')
    # for balance_info in classes_balances:
        # print(f"Attribute {balance_info['attribute']}: {balance_info['balance_percentage']:.2f} (class 1: {balance_info['class_1_count']}, class -1: {balance_info['class_minus_1_count']})")
    
    # classifier = CelebAIncrementalClassifier('../CelebADrawings/', '../CelebA/Anno/list_attr_celeba.txt', 'Smiling') 
    # classifier = CelebALSTMClassifier('../CelebADrawings/', '../CelebA/Anno/list_attr_celeba.txt', 'Smiling')

    # classifier = CelebALSTMClassifier('../test-convert-to-draw/all_drawings.txt', '../test-convert-to-draw/attributes.txt', 'Smiling')
    classifier = CelebAIncrementalClassifier('../test-convert-to-draw/drawings', '../test-convert-to-draw/attributes.txt', 'Smiling')

    classifier.train()
    classifier.evaluate()