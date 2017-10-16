"""
Extension of keras ImageDataGenerator for image segmentation data,
where input and label are arrays of the same shape and need to be
distorted in the same way.
"""
import keras.preprocessing.image as _image
import numpy as _np
from builtins import super


class ImageDataGenerator(_image.ImageDataGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def random_transform_covariant(self, x, y, seed=None):
        """
        Randomly augment a single image tensor and its label.

        # Arguments
            x: 3D tensor, single image.
            y: 3D tensor, single image label.
            seed: random seed.

        # Returns
            A randomly transformed version of the inputs (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if seed is not None:
            _np.random.seed(seed)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if self.rotation_range:
            theta = _np.pi / 180 * _np.random.uniform(
                -self.rotation_range,
                self.rotation_range
            )
        else:
            theta = 0

        if self.height_shift_range:
            tx = _np.random.uniform(
                -self.height_shift_range,
                self.height_shift_range
            ) * x.shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            ty = _np.random.uniform(
                -self.width_shift_range,
                self.width_shift_range
            ) * x.shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = _np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = _np.random.uniform(
                self.zoom_range[0],
                self.zoom_range[1],
                2
            )

        # Initialise transformation matrix with identity matrix (no transformation)
        transform_matrix = _np.eye(3)
        
        if theta != 0:
            rotation_matrix = _np.array([
                [_np.cos(theta), -_np.sin(theta), 0],
                [_np.sin(theta), _np.cos(theta), 0],
                [0, 0, 1]
            ])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = _np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
            
            transform_matrix = _np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = _np.array([
                [1, -_np.sin(shear), 0],
                [0, _np.cos(shear), 0],
                [0, 0, 1]
            ])
            transform_matrix = _np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = _np.array([
                [zx, 0, 0],
                [0, zy, 0],
                [0, 0, 1]
            ])
            transform_matrix = _np.dot(transform_matrix, zoom_matrix)

        height, width = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = _image.transform_matrix_offset_center(
            transform_matrix,
            height,
            width
        )
        
        x = _image.apply_transform(x, transform_matrix, img_channel_axis,
                                   fill_mode=self.fill_mode, cval=self.cval)
        y = _image.apply_transform(y, transform_matrix, img_channel_axis,
                                   fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip and _np.random.random() < 0.5:
            x = _image.flip_axis(x, img_col_axis)
            y = _image.flip_axis(y, img_col_axis)

        return x, y
