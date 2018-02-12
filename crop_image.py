import glob
import sys

import numpy as np
import os
from PIL import Image
from keras.models import load_model


def crop_image(imgname, model, threshold=200, out_size=None):
    if isinstance(model, str):
        model = load_model(model)
        
    model_size = model.input_shape[1:-1]
    
    img0 = Image.open(imgname)
    x_size, y_size = img0.size

    # Convert input image into square BW image
    # with luminance values in range [-1, 1]
    img = img0.resize(model_size)
    img = np.array(img.convert("L"))
    img = 2 * ((img / 255.) - 0.5)
    
    # Predict pixel-wise probabilities and convert into luminance
    pred = model.predict(img[None, :, :, None])[0, :, :, 0]
    pred = (255 * pred).astype(np.uint8)
    
    # Resize image to initial size
    pred = np.array(Image.fromarray(pred).resize(img0.size))
    
    pos = np.array(np.where(pred > threshold))
    
    medians = np.median(pos, axis=1).astype(int)
    std = 2 * int(max(np.std(pos, axis=1))) + int(0.1 * min(x_size, y_size))
    
    y_min, x_min = medians - std
    y_max, x_max = medians + std
    
    cropped = img0.crop([
        max(0, x_min),  # left
        max(0, y_min),  # upper
        min(x_size, x_max),  # right
        min(y_size, y_max)  # lower
    ])

    mask = pred > threshold
    masked = np.array(img0) * mask[:, :, None]
    masked = Image.fromarray(masked)
    
    if out_size:
        cropped = cropped.resize(out_size)
        masked = masked.resize(out_size)

    return cropped, masked


if __name__ == "__main__":
    modelname = sys.argv[1].strip()
    
    model = load_model(modelname)
    
    for imgname in glob.iglob('../openfriday_nn/Test/*.png'):
        try:
            cropped, mask = crop_image(imgname=imgname, model=model)
            
            u = os.path.split(modelname)[-1]
            v = os.path.split(imgname)[-1]
            # cropped.save("predictions/pred-{}-{}.jpg".format(u, v))
            mask.save("predictions/pred-{}-{}.jpg".format(u, v))
        except Exception as err:
            print(err)
