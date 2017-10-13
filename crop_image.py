import os
import sys

import numpy as np

from PIL import Image

from keras.models import load_model


def crop_image(imgname, model=None):   

    if isinstance(model, str):
        model = load_model(model)

    img0 = Image.open(imgname)
    xsize, ysize = img0.size

    img = img0.resize((512, 512))
    img = np.array(img.convert('L'))
    img = 2 * ((img / 255.) - 0.5)
    
    pred = model.predict(img[None,:,:,None])[0,:,:,0]
    pred = (255 * pred).astype(np.uint8)

    pred = np.array(Image.fromarray(pred).resize(img0.size))

    pos = np.array(np.where(pred > 200))

    medians = np.median(pos, axis=1).astype(int)
    std = 2 * int(max(np.std(pos, axis=1))) + int(0.1 * min(img0.size))

    ymin, xmin = medians - std
    ymax, xmax = medians + std
              
    out = img0.crop([
        max(0, xmin), # left
        max(0, ymin), # upper
        min(xsize, xmax), # right
        min(ysize, ymax) # lower
    ])

    return out.resize((224,224))


if __name__ == '__main__':
    modelname = sys.argv[1].strip()
    imgname = sys.argv[2].strip()

    out = crop_image(imgname, model=modelname)

    u = os.path.split(modelname)[-1]
    v = os.path.split(imgname)[-1]
    out.save('predictions/pred-{}-{}.png'.format(u, v))

