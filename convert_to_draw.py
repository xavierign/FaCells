import numpy as np
import cv2
import tools

def convert_to_draw(inFile,
					input_dir = 'test_photos',
					output_path = 'test_draws',
					include_svg = True,
					include_annotated_img = True,
					line_width = 1,
					line_color = (150,150,150)):
	
	lines =  tools.sketch(input_dir + '/' + inFile, output_path + '/' + inFile.split('.')[0] + '_lines.svg')

	if include_annotated_img:
		#read image (again)
		im = cv2.imread(input_dir + '/' + inFile);
		#Make copy of the image
		imageLine = im.copy()

		for line in lines:
		    for i in range(1,len(line)):
		        
		        # Draw the image from point A to B
		        pointA = (line[i-1][0],line[i-1][1])
		        pointB = (line[i][0],line[i][1])
		        cv2.line(imageLine, pointA, pointB, line_color, thickness=1)
		print(output_path + '/' + inFile.split('.')[0] + '_lines.png')
		cv2.imwrite(output_path + '/' + inFile.split('.')[0] + '_lines.png',imageLine);

	return lines

convert_to_draw('rosario.jpg')
