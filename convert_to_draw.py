from email import utils
from typing import List, Tuple
import numpy as np
import cv2
import tools
from utils import Drawing, Line, PointEncoded
import tensorflow as tf
import statistics


def convert_sketch_to_encoded_drawing(sketch: List[List[Tuple[int, int]]]) -> Drawing:
	new_lines = [] # : List[Line]
	for line in sketch:
		line_tmp = Line([])
		for i, (point_x, point_y) in enumerate(line):
			point_tmp = PointEncoded(point_x, point_y)
			if i == 0:
				point_tmp.set_start_of_line()
			if i == len(line)-1:
				point_tmp.set_end_of_line()
			line_tmp.add_point(point_tmp)
		new_lines += [line_tmp]

	return Drawing(new_lines)

"""dataframe metadata fotos (id foto, file location, atributos (forest vs building)) 
	-> datos ordenados aleat en training y testing para usar en red neuronal
	formato:
	data = [
    [[[2598, 178, 1, 0], [2580, 194, 0, 0], ...]], : Drawing
    [[[2562, 226, 0, 0], [2562, 222, 0, 0], ...]], : Drawing
   	 ...
	]
	labels = [0, 1, ...]

	en principio que convierta a tf.data.Dataset
	si no, ver funcion text_dataset_from_directory
	  necesita directorio y archivos txt
	"""
def generate_dataset(drawings):
	ragged_drawings = fill_drawings_max(drawings)
	dataset = tf.data.Dataset.from_tensor_slices(ragged_drawings)
	return dataset

def fill_drawings_max(drawings):
	max_lines = max(len(drawing) for drawing in drawings)
	max_points = max(len(line) for drawing in drawings for line in drawing)
	rectangular_drawings = []
	for drawing in drawings:
		lines = []
		for line in drawing:	
			padded_line = line + [(0, 0)] * (max_points - len(line))
			lines.append(padded_line)
		padded_drawing = lines + [[(0, 0)] * max_points] * (max_lines - len(lines))
		rectangular_drawings.append(padded_drawing)
	ragged_drawings = tf.ragged.constant(rectangular_drawings)
	return ragged_drawings


def get_drawings():
	"""convert all photos into drawings -> iterate through directory and convert each to drawing
	this function will be called before generate_dataset"""
	pass

def len_lines(drawing):
	count = []
	for line in drawing:
		count += [len(line)]
	print(count)
	mean = statistics.mean(count)
	median = statistics.median(count)
	mode = statistics.mode(count)
	print(f"The mean is {mean}")
	print(f"The median is {median}")
	print(f"The mode is {mode}")


def convert_to_draw(inFile,
					input_dir = 'test_photos',
					output_path = 'test_draws',
					include_svg = True,
					include_annotated_img = True,
					line_width = 1,
					line_color = (10,10,256)):
	
	lines =  tools.sketch(input_dir + '/' + inFile, output_path + '/' + inFile.split('.')[0] + '_lines.svg')
	drawing = convert_sketch_to_encoded_drawing(lines)
	# drawing.pretty_printer()
	if include_annotated_img:
		#read image (again)
		im = cv2.imread(input_dir + '/' + inFile)
		#Make copy of the image
		imageLine = im.copy()

		for line in lines:
		    for i in range(1,len(line)):
		        
		        # Draw the image from point A to B
		        pointA = (line[i-1][0],line[i-1][1])
		        pointB = (line[i][0],line[i][1])
		        cv2.line(imageLine, pointA, pointB, line_color, thickness=1)
		# print(output_path + '/' + inFile.split('.')[0] + '_lines.png')
		cv2.imwrite(output_path + '/' + inFile.split('.')[0] + '_lines.png',imageLine)
	# print(lines)
	return lines

if __name__ == "__main__":
	# dataframe :
	draw = convert_to_draw("4.jpg", input_dir="../intel-image-building-forest/seg_train/buildings", output_path="test_draws/intel-image-building-forest")
	print(len_lines(draw))
	drawings = [draw]
	# drawings += [convert_to_draw("8.jpg", input_dir="../intel-image-building-forest/seg_train/forest", output_path="test_draws/intel-image-building-forest")]
	dataset = generate_dataset(drawings)
	
	num_elements = tf.data.experimental.cardinality(dataset).numpy()
	print("Number of elements in the dataset:", num_elements)

	# Print the shape of the first element in the dataset
	first_element = next(iter(dataset))
	first_shape = tf.shape(first_element)
	print("Shape of the first element:", first_shape)

	# Print the types of the elements in the dataset
	element_types = dataset.element_spec
	print("Types of the elements in the dataset:", element_types)