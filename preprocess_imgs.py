import glob
import os

import numpy as np
from PIL import Image


def make_inputs_from_imgs(size, in_folder='./', out_folder='./data/', extension='png'):
   
    IN_FOLDER_IMGS = os.path.join(in_folder, 'Images/')
    IN_FOLDER_MASKS = os.path.join(in_folder, 'Annotations/')
    OUT_FOLDER_PICS = os.path.join(out_folder, 'origs/')
    OUT_FOLDER_LABELS = os.path.join(out_folder, 'labels/')

    n_corr = 0
    for img_name in glob.glob(os.path.join(IN_FOLDER_IMGS, '*.' + extension)):
        # compiling file names
        basename = os.path.basename(img_name)
        mask_name = os.path.join(IN_FOLDER_MASKS, basename)
        
        basename = basename[:-4]  # remove extension (dot and 3 characters)
        
        img_outfile = os.path.join(OUT_FOLDER_PICS, basename + '_pic.npy')
        lab_outfile = os.path.join(OUT_FOLDER_LABELS, basename + '_label.npy')
        
        try:
            # Convert original jpeg to B/W
            img_orig = Image.open(img_name)
            img_label = Image.open(mask_name)
            img_bw = img_orig.convert('L')
            
            # Get 512x512 arrays from the images
            label_final = np.array(img_label.resize((size, size)), dtype=np.uint8)
            img_final = np.array(img_bw.resize((size, size)), dtype=np.uint8)
            
            # Save to files
            np.save(img_outfile, img_final)
            np.save(lab_outfile, label_final)
        
        except Exception as err:
            print(err)
            print('Corrupted file: {}'.format(img_name))
            n_corr += 1
    
    print("Found {:d} corrupted files".format(n_corr))
