import SimpleITK as sitk
import skimage
from skimage import transform
from skimage import io
import os
import h5py
from PIL import Image
import numpy as np
import scipy.ndimage


def shrink_image_to_128(source_file_name, target_file_name):
    image = io.imread(source_file_name)
    resized_volume = transform.resize(image, (128, 128))
    resized_volume = skimage.img_as_ubyte(resized_volume)
    io.imsave(target_file_name, resized_volume)


def shrink_volume_to_128(source_file_name, target_file_name):
    image = sitk.ReadImage(source_file_name)
    volume = sitk.GetArrayFromImage(image)
    resized_volume = scipy.ndimage.zoom(volume, (128/volume.shape[0], 128/volume.shape[1], 128/volume.shape[2]))
    # resized_volume = transform.resize(volume, (128, 128, 128))
    # resized_int_volume = skimage.img_as_int(resized_volume)
    resized_image = sitk.GetImageFromArray(resized_volume)
    sitk.WriteImage(resized_image, target_file_name)


if __name__ == '__main__':
    raw_data_root_folder = r'C:\XsCT\XsCT\data\COLONOgraphy\mesh_data_256'
    target_data_root_folder = r'C:\XsCT\XsCT\data\mesh_data_128'
    sub_folders = next(os.walk(raw_data_root_folder))[1]
    for sub_folder in sub_folders:
        if not os.path.exists(target_data_root_folder + '\\' + sub_folder):
            os.mkdir(target_data_root_folder + '\\' + sub_folder)
        shrink_image_to_128(raw_data_root_folder + '\\' + sub_folder + '\\resized_drr01.bmp',
                            target_data_root_folder + '\\' + sub_folder + '\\resized_drr01.bmp')
        shrink_image_to_128(raw_data_root_folder + '\\' + sub_folder + '\\resized_drr02.bmp',
                            target_data_root_folder + '\\' + sub_folder + '\\resized_drr02.bmp')
        shrink_volume_to_128(raw_data_root_folder + '\\' + sub_folder + '\\result_img.nii',
                             target_data_root_folder + '\\' + sub_folder + '\\result_img.nii')

        drr_1 = Image.open(target_data_root_folder + '\\' + sub_folder + '\\resized_drr01.bmp');
        drr_2 = Image.open(target_data_root_folder + '\\' + sub_folder + '\\resized_drr02.bmp');
        ct_image = sitk.ReadImage(target_data_root_folder + '\\' + sub_folder + '\\result_img.nii')
        ct_volume = sitk.GetArrayFromImage(ct_image)
        f = h5py.File(target_data_root_folder + '\\' + sub_folder + '\\' + 'ct_xray_data.h5', 'w')
        f['ct'] = ct_volume
        f['ori_size'] = np.int64(320)
        f['spacing'] = [1.0, 1.0, 1.0]
        f['xray1'] = drr_1
        f['xray2'] = drr_2
        f.close()

    pass
