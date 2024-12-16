import xml.etree.ElementTree as ET
import numpy as np
import os
from pathlib import Path
import pandas as pd

def parse_svg_to_labeled_coordinates(svg_file):
    """
    Parses an SVG file and extracts a single list of [x, y, l] coordinates from polyline elements.
    
    Args:
        svg_file (str): Path to the SVG file.
    
    Returns:
        list of list: A single list where each element is [x, y, l], with l indicating line start (-1), line end (1), or intermediate (0).
    """
    # Parse the SVG file
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # List to store all labeled points
    labeled_coordinates = []

    # Iterate through all polyline elements
    for polyline in root.findall('.//{*}polyline'):
        points = polyline.get('points')
        if points:
            # Split the points string by commas and spaces, and convert to integers
            nums = list(map(int, points.replace(',', ' ').split()))
            # Group the numbers into (x, y) pairs
            coords = [[nums[i], nums[i + 1]] for i in range(0, len(nums), 2)]

            # If the polyline has at least one point, add labels
            if coords:
                for idx, (x, y) in enumerate(coords):
                    if idx == 0:
                        labeled_coordinates.append([x, y, -1])  # Start of the line
                    elif idx == len(coords) - 1:
                        labeled_coordinates.append([x, y, 1])   # End of the line
                    else:
                        labeled_coordinates.append([x, y, 0])   # Intermediate points

    return labeled_coordinates

def process_all_svgs(input_dir, output_dir):
    """
    Processes all SVG files in the input directory, extracts coordinates, and saves them as .npy files.
    
    Args:
        input_dir (str): Path to the input directory containing SVG files.
        output_dir (str): Path to the output directory where .npy files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all SVG files in the input directory
    svg_files = list(Path(input_dir).glob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {input_dir}")
        return

    for svg_file in svg_files:
        # Create the corresponding .npy file path
        npy_file = Path(output_dir) / (svg_file.stem + ".npy")

        # Parse and save the coordinates
        coordinates = parse_svg_to_labeled_coordinates(svg_file)
        if coordinates:
            np.save(npy_file, coordinates)
            print(f"Processed and saved: {npy_file}")
        else:
            print(f"No coordinates found in {svg_file}")


def read_and_print_npy(npy_file):
    """
    Reads a .npy file containing lists of (x, y) coordinates and prints the contents.
    
    Args:
        npy_file (str): Path to the .npy file.
    """
    try:
        # Load the .npy file
        data = np.load(npy_file, allow_pickle=True)
        
        # Print each polyline
        print(data)
        print(f"Contents of {npy_file}:\n")
        for i, polyline in enumerate(data):
            print(f"Polyline {i + 1}:")
            for point in polyline:
                print(point)
            print()  # Add a blank line between polylines
    except Exception as e:
        print(f"Error reading {npy_file}: {e}")


def load_and_analyze_attributes(file_path):
    """
    Loads the attribute annotations file and analyzes the balance of -1 and 1 labels for each attribute.
    
    Args:
        file_path (str): Path to the attribute annotation file.
    """
    # Load the data, skipping the first row and using the second row as the header
    df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, header=0)
    
    # Drop the first column if it's an index or filename column
    if df.columns[0].lower() in ['index', 'filename']:
        df = df.drop(df.columns[0], axis=1)

    # Calculate the distribution of -1 and 1 for each attribute
    balance_info = {}
    for column in df.columns:
        counts = df[column].value_counts()
        count_neg1 = counts.get(-1, 0)
        count_pos1 = counts.get(1, 0)
        balance_info[column] = abs(count_neg1 - count_pos1)

    # Convert to a DataFrame and sort by balance (smallest difference is most balanced)
    balance_df = pd.DataFrame.from_dict(balance_info, orient='index', columns=['Balance Difference'])
    balance_df = balance_df.sort_values(by='Balance Difference')

    print("Attributes sorted by balance between -1 and 1 labels:")
    print(balance_df)

    return df


if __name__ == "__main__":
    


    #input_directory = "/Users/xaviergonzalez/Library/Mobile Documents/com~apple~CloudDocs/Documents/Art/AI experiments/deepFaceDraw/FaCells/CelebADrawings"
    #output_directory = "CelebAMask-HQ/CelebA-HQ-np/"
    
    #process_all_svgs(input_directory, output_directory)

    #npy_path = "CelebAMask-HQ/CelebA-HQ-np/2666_lines.npy"
    #read_and_print_npy(npy_path)

    # File path
    file_path = "/Users/xaviergonzalez/Documents/repos/FaCells/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt"

    # Load and display the DataFrame
    df = load_and_analyze_attributes(file_path)
    #print(df.head())

    # Optionally, save the DataFrame as a CSV file
    #output_path = "/Users/xaviergonzalez/Documents/repos/FaCells/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.csv"
    #df.to_csv(output_path, index=False)
    #print(f"DataFrame saved to {output_path}")