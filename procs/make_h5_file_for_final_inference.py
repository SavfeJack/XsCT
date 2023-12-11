

import h5py
import numpy as np
import SimpleITK as sitk
from PIL import Image


if __name__=='__main__':
    root_dir = 'C:\\Users\\user\\Desktop\\temp\\'

    empty = np.zeros(shape=(256, 256, 256))
    xray1 = Image.open(root_dir + 'resized_01.bmp')
    xray2 = Image.open(root_dir + 'resized_02.bmp')

    f = h5py.File(root_dir + 'ct_xray_data.h5', 'w')
    f['ct'] = empty
    f['ori_size'] = np.int64(320)
    f['spacing'] = [1.0, 1.0, 1.0]
    f['xray1'] = xray1
    f['xray2'] = xray2
    f.close()