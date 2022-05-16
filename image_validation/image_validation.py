"""
Author: Artin

"""

import glob
import os.path
import hashlib
from PIL import Image, ImageStat
import numpy as np
import shutil


def validate_images(input_dir, output_dir, log_file, formatter=''):
    # Getting absolute path of input and output directory
    input_dir_abs = os.path.abspath(input_dir)
    output_dir_abs = os.path.abspath(output_dir)
    # Creating output directory only if it is not already exist
    if not os.path.exists(output_dir_abs):
        os.makedirs(output_dir_abs)
    # Get list of all files path inside input directory and its subdirectories
    files = sorted(glob.glob(os.path.join(input_dir_abs, "**", "*.*"), recursive=True))

    v_image_files = []
    v_image_hashes = []
    nv_files = []
    # Looping through found file list and checking conditions of a valid image
    for file in files:
        # Record file with invalid extension in appropriate array
        if not file.lower().endswith(('.jpg', '.jpeg')):
            nv_files.append(['1', file])
            continue

        # Record file with invalid size in appropriate array
        if os.path.getsize(file) > 250000:
            nv_files.append(['2', file])
            continue

        try:
            image_file = Image.open(file)
        except:
            # Record file if it is cannot be opened as an image file
            nv_files.append(['3', file])
            continue

        # Create hash of the image file
        image_file_hash = hashlib.sha256()
        image_file_hash.update(image_file.tobytes())
        image_file_hash = image_file_hash.digest()

        # Checking for criteria of rule number 4
        image_array = np.array(image_file)
        if image_array.shape[0] < 96 \
                or image_array.shape[1] < 96 \
                or len(image_array.shape) < 3 \
                or image_array.shape[2] != 3 \
                or image_file.mode != 'RGB':
            nv_files.append(['4', file])
            continue

        # Checking for image data variance - rule number 5
        image_file_var = ImageStat.Stat(image_file).var
        if image_file_var[0] <= 0 or image_file_var[1] <= 0 or image_file_var[2] <= 0:
            nv_files.append(['5', file])
            continue
        # Record file as and invalid one if it is repeated by considering its hash
        if image_file_hash in v_image_hashes:
            nv_files.append(['6', file])
            continue
        v_image_hashes.append(image_file_hash)
        v_image_files.append(file)
    # Copying valid files and creation of output directory if it is already not exist
    for index, valid_file in enumerate(v_image_files):
        shutil.copy(valid_file, os.path.join(output_dir_abs, f"{index:{formatter}}" + '.jpg'))
    # Creating a log file and appending lines per each invalid file
    log_file_object = open(log_file, 'a+', encoding='utf-8')
    for invalid_file in nv_files:
        log_file_object.write(f"{os.path.basename(invalid_file[1])};{invalid_file[0]}\n")
    log_file_object.close()
    # Returning valid images count
    return len(v_image_files)
