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

    img_outfile = img_name[:-4] + '_pic_np'
    img_outfile = os.path.join(OUT_FOLDER_PICS, img_outfile)
    lab_outfile = img_name[:-4] + '_label_np'
    lab_outfile = os.path.join(OUT_FOLDER_LABELS, lab_outfile)

    try:
        # original jpeg
        img_proc = Image.open(jpeg_infile_path)
        img_proc_greyscale = img_proc.convert('L')
        img_final = np.array(img_proc_greyscale.resize((SIZE,SIZE)), dtype=np.uint8)

        # Load polygon lines
        polygon = literal_eval(annotations[img_name])
        polygon = np.array([polygon['all_points_x'], polygon['all_points_y']])
        polygon = tuple([tuple(u) for u in polygon.T[:-1]])

        label_img = Image.new('L', (SIZE, SIZE))
        draw = ImageDraw.Draw(label_img)
        draw.polygon(polygon, fill='white', outline='white')

        lab_final = np.array(label_img)

        # save to files
        np.savetxt(img_outfile, img_final, fmt='%d')
        np.savetxt(lab_outfile, lab_final, fmt='%d')

    except Exception as err:
        print(err)
        print('Corrupted file: %s' %(jpeg_infile_path,))
        n_corr += 1
