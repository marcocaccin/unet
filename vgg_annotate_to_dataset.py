import glob
import os

import pandas as pd
import numpy as np

from ast import literal_eval
from PIL import Image, ImageDraw
from tqdm import tqdm


SIZE = 512

pic_extension_pattern = '*.jpg'
IN_FOLDER = '../imgs/'
ANNOTATION_FILE = '../imgs/via_region_data.csv'
OUT_FOLDER_PICS = '../unet/data/origs'
OUT_FOLDER_LABELS = '../unet/data/labels'


annotations = pd.read_csv(
    ANNOTATION_FILE,
    usecols=['#filename', 'region_shape_attributes'],
    index_col='#filename'
).to_dict()['region_shape_attributes']


n_corr = 0
for jpeg_infile_path in tqdm(glob.glob(os.path.join(IN_FOLDER, pic_extension_pattern))):
    # compiling file names

    img_name = os.path.split(jpeg_infile_path)[-1]

    img_outfile = img_name[:-4] + '_pic.npy'
    img_outfile = os.path.join(OUT_FOLDER_PICS, img_outfile)
    lab_outfile = img_name[:-4] + '_label.npy'
    lab_outfile = os.path.join(OUT_FOLDER_LABELS, lab_outfile)

    try:
        # Convert original jpeg to B/W
        img_orig = Image.open(jpeg_infile_path)
        img_bw = img_orig.convert('L')

        # Load polygon lines
        polygon = literal_eval(annotations[img_name])
        polygon = np.array([polygon['all_points_x'], polygon['all_points_y']])
        polygon = tuple([tuple(u) for u in polygon.T[:-1]])

        # Paint polygon on new image from its contour
        img_label = Image.new('L', img_bw.size)
        draw = ImageDraw.Draw(img_label)
        draw.polygon(polygon, fill='white', outline='white')

        # Get 512x512 arrays from the images
        label_final = np.array(img_label.resize((SIZE, SIZE)), dtype=np.uint8)
        img_final = np.array(img_bw.resize((SIZE, SIZE)), dtype=np.uint8)

        # Save to files
        np.save(img_outfile, img_final)
        np.save(lab_outfile, label_final)

    except Exception as err:
        print(err)
        print('Corrupted file: {}'.format(jpeg_infile_path))
        n_corr += 1

print("Found {:d} corrupted files".format(n_corr))
