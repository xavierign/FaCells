from email import utils
from itertools import count
from typing import Dict, List, Tuple
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
from collections import defaultdict


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

def convert_to_draw(inFile,
                    input_dir = 'test_photos',
                    output_path = 'test_draws',
                    include_svg = True,
                    include_annotated_img = True,
                    line_width = 1,
                    line_color = (10,10,256)):
    if include_svg:
        lines =  tools.sketch(input_dir + '/' + inFile, output_path + '/' + inFile.split('.')[0] + '_lines.svg')
    else:
        lines =  tools.sketch(input_dir + '/' + inFile)
    
    drawing = convert_sketch_to_encoded_drawing(lines)
    if include_annotated_img:
        # read image (again)
        im = cv2.imread(input_dir + '/' + inFile)
        # make copy of the image
        imageLine = im.copy()

        for line in lines:
            for i in range(1,len(line)):
                # draw the image from point A to B
                pointA = (line[i-1][0],line[i-1][1])
                pointB = (line[i][0],line[i][1])
                cv2.line(imageLine, pointA, pointB, line_color, thickness=1)

    return drawing

def save_drawing_txt(drawing: Drawing, file_name: str):
    """Appends drawing in txt file_name.
    
    A Drawing will be represented as a list of elements (x, y, s, e) where the first two represent the x and y 
    coordinates, s is 1 if point is the start of the line, otherwise 0 and same for e but if it's end of line.
    """
    with open(file_name, 'a') as file:
        lines = drawing.lines
        all_points = []
        for line in lines:
            for point in line.points:
                all_points.append(f"({point.x},{point.y},{point.start_of_line},{point.end_of_line})")
        
        # join all points for the drawing into a single row with commas separating them
        drawing_str = ','.join(all_points)
        file.write(drawing_str + '\n')

def update_len_longest_line(drawing: Drawing, len_longest_line: int):
    drawing.get_stats()
    drawing_len_longest_line = drawing.len_longest_line
    if drawing_len_longest_line == None:
        raise ValueError("len_longest_line is None.")
    
    return max(drawing_len_longest_line, len_longest_line)


def bulk_convert_to_draw(input_directory: str, output_directory: str, save_as_svg: bool = True) -> None:
    """Given an input directory and an output directory, bulk_convert_to_draw converts recursively in the directory"""
    # if output_directory does not exist, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not save_as_svg:
        txt_output_file = os.path.join(output_directory, 'all_drawings.txt')
        # if file exists, delete content of it
        if os.path.exists(txt_output_file):
            os.remove(txt_output_file)

    i = 0
    # iterate over the first level of the input directory
    for entry in os.listdir(input_directory):
        entry_path = os.path.join(input_directory, entry)
        if os.path.isdir(entry_path):
            # if there are subdirectories, call bulk_convert_to_draw recursively
            has_subdirs = any(os.path.isdir(os.path.join(entry_path, subentry)) for subentry in os.listdir(entry_path))
            if has_subdirs:
                print(f"Converting {entry_path}...")
                bulk_convert_to_draw(entry_path, os.path.join(output_directory, entry), save_as_svg=save_as_svg)
            else:
                # no subdirectories so process all files 
                dest_directory = os.path.join(output_directory, entry)
                if not os.path.exists(dest_directory):
                    os.makedirs(dest_directory)
                for file in os.listdir(entry_path):
                    if file.lower().endswith('.jpg'):
                        if i % 100 == 0:
                            print(f"   Processed {i}... converting image {file} in {entry_path} to drawing in {dest_directory}...")
                        i += 1
                        drawing = convert_to_draw(file, input_dir=entry_path, output_path=dest_directory, include_svg=save_as_svg, include_annotated_img=False)

                        if not save_as_svg:
                            save_drawing_txt(drawing, txt_output_file)

        else: # if it's not a directory and it's a jpg file, convert to draw
            if entry.lower().endswith('.jpg'):
                dest_directory = output_directory
                if i % 100 == 0:
                    print(f"   Processed {i}... converting image {entry} to drawing in {output_directory}...")
                i += 1
                drawing = convert_to_draw(entry, input_dir=input_directory, output_path=output_directory, include_svg=save_as_svg, include_annotated_img=False)

                if not save_as_svg:
                    save_drawing_txt(drawing, txt_output_file)


def pad_drawings_max(file_path):
    """Given a txt file path, modify file so that every drawing (row) has the same number of points (padding with (0,0,0,0))."""
    max_length = 0
    with open(file_path, 'r') as file:
        for line in file:
            points = line.strip().split('),(')
            num_points = len(points) + 1
            max_length = max(max_length, num_points)

    with open(file_path, 'r+') as file:
        lines = file.readlines()  
        file.seek(0)  # move pointer to the beginning to overwrite
        
        for line in lines:
            line = line.strip().strip('()')
            points = line.split('),(')
            
            num_tuples = len(points)
            
            # calculate how many (0,0,0,0) are needed
            if num_tuples < max_length:
                padding_needed = max_length - num_tuples
                padding = ['0,0,0,0'] * padding_needed
                points.extend(padding)
            
            # rebuild the line with padded tuples and write it
            drawing_str = '),('.join(points)
            file.write('(' + drawing_str + ')\n')


def count_points_per_drawing(file_path: str) -> Dict:
    """Given a file path, calculates how many points each drawing has.
    
    Expects file to have one drawing per row and to consist of tuples of 4 elements separated by comma.
    """
    points_count_dict = defaultdict(int)
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if line:
                if i % 100 == 0:
                    print(f"Processing line {i}...")
                points = line.split('),(')
                points_count_dict[len(points)] += 1

    sorted_points_count = dict(sorted(points_count_dict.items()))
    print("Number of points in drawing -> Number of drawings")
    for num_points, count in sorted_points_count.items():
        print(f"{num_points} -> {count}")


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
    
    # draw = convert_to_draw("4.jpg", input_dir="../intel-image-building-forest/seg_train/buildings", output_path="test_draws/intel-image-building-forest")
    # encoded_drawing = convert_sketch_to_encoded_drawing(draw)
    # encoded_drawing.pretty_printer()
    
    # bulk_convert_to_draw('../CelebA/Img/img_align_celeba', '../CelebADrawings/', save_as_svg=False)
    # bulk_convert_to_draw('../test-convert-to-draw', '../test-convert-to-draw', save_as_svg=False)
    # pad_drawings_max('../CelebADrawings/all_drawings.txt')
    # count_points_per_drawing('../CelebADrawings/all_drawings.txt')

    drawing = convert_to_draw("Ori.jpg", input_dir="../test", output_path="../test")
    drawing.pretty_printer()