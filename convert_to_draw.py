from email import utils
from typing import List, Tuple
from filters import F_SobelX, appmask
import numpy as np
import cv2
import tools
from utils import Drawing, Line, PointEncoded
import tensorflow as tf
import statistics
from PIL import Image
from scipy.ndimage import sobel, gaussian_filter
import sys
import os


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
    # lines =  tools.sketch(input_dir + '/' + inFile, output_path + '/' + inFile.split('.')[0] + '_lines.json')
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
        # cv2.imwrite(output_path + '/' + inFile.split('.')[0] + '_lines.png',imageLine)
    # print(lines)
    return lines

def bulk_convert_to_draw(input_directory, output_directory):
    """Given an input directory and an output directory, bulk_convert_to_draw converts recursively in the directory"""
    # if output_directory does not exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # iterate over the first level of the input directory
    for entry in os.listdir(input_directory):
        entry_path = os.path.join(input_directory, entry)
        if os.path.isdir(entry_path):
            # if there are subdirectories, call bulk_convert_to_draw recursively
            has_subdirs = any(os.path.isdir(os.path.join(entry_path, subentry)) for subentry in os.listdir(entry_path))
            if has_subdirs:
                print(f"Converting {entry_path}...")
                bulk_convert_to_draw(entry_path, os.path.join(output_directory, entry))
            else:
                # no subdirectories so process all files 
                dest_directory = os.path.join(output_directory, entry)
                if not os.path.exists(dest_directory):
                    os.makedirs(dest_directory)
                for file in os.listdir(entry_path):
                    if file.lower().endswith('.jpg'):
                        input_file_path = os.path.join(entry_path, file)
                        output_file_path = os.path.join(dest_directory, file)
                        print(f"   Converting image {file} in {entry_path} to drawing in {dest_directory}...")
                        convert_to_draw(file, entry_path, dest_directory)
        else: # if it's not a directory and it's a jpg file, convert to draw
            if entry.lower().endswith('.jpg'):
                dest_directory = output_directory
                input_file_path = entry_path
                output_file_path = os.path.join(dest_directory, entry)
                print(f"   Converting image {input_directory} to drawing in {output_directory}...")
                convert_to_draw(entry, input_dir=input_directory, output_path=output_directory)

def get_sobel_results(image_path, output_path, blur_sigma=1.0):
    """Creates 2 images (one for each Sobel kernel applied) and saves them in output path with the name of the image
    and a suffix to indicate which coordinate it is."""
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    image_array = np.array(image)
    blurred_image = gaussian_filter(image_array, sigma=blur_sigma)
    # apply Sobel filter along both axes
    sobel_x = sobel(blurred_image, axis=0)
    sobel_y = sobel(blurred_image, axis=1)
    # normalize results
    sobel_x = ((sobel_x - sobel_x.min()) / (sobel_x.max() - sobel_x.min()) * 255).astype(np.uint8)
    sobel_y = ((sobel_y - sobel_y.min()) / (sobel_y.max() - sobel_y.min()) * 255).astype(np.uint8)

    sobel_x_image_path = os.path.join(output_path, f"sobel_x3_{os.path.basename(image_path)}")
    sobel_y_image_path = os.path.join(output_path, f"sobel_y3_{os.path.basename(image_path)}")
    # create images from arrays
    sobel_x_image = Image.fromarray(sobel_x)
    sobel_y_image = Image.fromarray(sobel_y)

    sobel_x_image.save(sobel_x_image_path)
    sobel_y_image.save(sobel_y_image_path)


if __name__ == "__main__":    
    # draw = convert_to_draw("4.jpg", input_dir="../intel-image-building-forest/seg_train/buildings", output_path="test_draws/intel-image-building-forest")
    # print("LINES SVG")
    # print(os.stat('test_draws/intel-image-building-forest/4_lines.svg').st_size)
    
    # took around 10 minutes to run
    # bulk_convert_to_draw('../intel-image-building-forest/', '../intel-image-building-forest-drawings/')

    # get_sobel_results("../IMG_4070.jpg", "../sobel-results/", blur_sigma=3.0)
    
    draw = convert_to_draw("4.jpg", input_dir="../intel-image-building-forest/seg_train/buildings", output_path="test_draws/intel-image-building-forest")
    encoded_drawing = convert_sketch_to_encoded_drawing(draw)
    encoded_drawing.pretty_printer()
    
    draw = convert_to_draw("23.jpg", input_dir="../intel-image-building-forest/seg_train/forest", output_path="test_draws/intel-image-building-forest")
    encoded_drawing = convert_sketch_to_encoded_drawing(draw)
    encoded_drawing.pretty_printer()
