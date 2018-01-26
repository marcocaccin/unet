import glob
import os

import numpy as np
from PIL import Image


if __name__ == '__main':
    SIZE = 384
    
    pic_extension_pattern = '*.png'
    IN_FOLDER_IMGS = '../imgs/'
    IN_FOLDER_MASKS = '../masks/'
    OUT_FOLDER_PICS = '../unet/data/origs'
    OUT_FOLDER_LABELS = '../unet/data/labels'

    n_corr = 0
    for img_name in glob.glob(os.path.join(IN_FOLDER_IMGS, pic_extension_pattern)):
        # compiling file names
        basename = os.path.basename(img_name)
        mask_name = os.path.join(IN_FOLDER_MASKS, basename)
        
        basename = basename[:-4]
        
        img_outfile = os.path.join(OUT_FOLDER_PICS, basename + '_pic.npy')
        lab_outfile = os.path.join(OUT_FOLDER_LABELS, basename + '_label.npy')
        
        try:
            # Convert original jpeg to B/W
            img_orig = Image.open(img_name)
            img_label = Image.open(mask_name)
            img_bw = img_orig.convert('L')
            
            # Get 512x512 arrays from the images
            label_final = np.array(img_label.resize((SIZE, SIZE)), dtype=np.uint8)
            img_final = np.array(img_bw.resize((SIZE, SIZE)), dtype=np.uint8)
            
            # Save to files
            np.save(img_outfile, img_final)
            np.save(lab_outfile, label_final)
        
        except Exception as err:
            print(err)
            print('Corrupted file: {}'.format(img_name))
            n_corr += 1
    
    print("Found {:d} corrupted files".format(n_corr))
