

import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import h5py
from PIL import Image


def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

def img_respacing(input_img, target_spacing=[1.0, 1.0, 1.0], resamplemethod=sitk.sitkLinear): # sitk.sitkNearestNeighbor
    origin_size = input_img.GetSize()
    origin_spacing = input_img.GetSpacing()

    new_size = [
        round(origin_size[0] * (origin_spacing[0] / target_spacing[0])),
        round(origin_size[1] * (origin_spacing[1] / target_spacing[1])),
        round(origin_size[2] * (origin_spacing[2] / target_spacing[2]))
    ]
    print('new size: ', new_size)
    print('new spacing: ', target_spacing)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(input_img.GetOrigin())
    resampler.SetOutputDirection(input_img.GetDirection())
    resampler.SetOutputSpacing(target_spacing)

    # Set different type according to the need to resample the image
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(input_img.GetPixelID())
    else:
        resampler.SetOutputPixelType(input_img.GetPixelID())# Linear interpolation is used for PET/CT/MRI and the like

    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    output_img = resampler.Execute(input_img)
    return output_img


def img_pad(input_img, target_size=[258, 258, 258], constant=-1024):
    # input_size = input_img.GetSize()
    # bound = [
    #     int((target_size[0] - input_size[0]) / 2),
    #     int((target_size[1] - input_size[1]) / 2),
    #     int((target_size[2] - input_size[2]) / 2)
    # ]
    #
    # for idx in range(3):
    #     if bound[idx] < 0:
    #         bound[idx] = 0
    #
    # padder = sitk.ConstantPadImageFilter()
    # padder.SetConstant(constant)
    # padder.SetPadLowerBound(bound)
    # padder.SetPadUpperBound(bound)
    # output_img = padder.Execute(input_img)
    # return output_img

    #3D, Z-slice first
    img = sitk.GetArrayFromImage(input_img)
    while img.shape[0] < 256:
        img = np.pad(
            img, [(1, 1), (0, 0), (0, 0)], mode='constant', constant_values=constant)

    img = sitk.GetImageFromArray(img)

    #2D, XY then
    bound = [
        int((target_size[0] - img.GetSize()[0]) / 2),
        int((target_size[1] - img.GetSize()[1]) / 2),
        int((target_size[2] - img.GetSize()[2]) / 2)
    ]

    for idx in range(3):
        if bound[idx] < 0:
            bound[idx] = 0

    padder = sitk.ConstantPadImageFilter()
    padder.SetConstant(constant)
    padder.SetPadLowerBound(bound)
    padder.SetPadUpperBound(bound)
    output_img = padder.Execute(img)

    return output_img


if __name__ == "__main__":
    ROOT_dir = 'C:\\Users\\user\\Desktop\\temp\\'

    # # #--------------------------------------------------------------------------------------
    # # # read and process
    img = sitk.ReadImage(ROOT_dir + "\\4580220\\4580220.mhd", sitk.sitkFloat32)
    # image_array = sitk.GetImageFromArray(img)
    size = np.array(list(reversed(img.GetSize())))
    spacing = np.array(list(reversed(img.GetSpacing())))
    print('size: ', size)
    print('spacing: ', spacing)
    # image_array.SetOrigin(origin)
    # spacing.reverse()
    # image_array.SetSpacing(spacing)

    # hist_match = sitk.HistogramMatchingImageFilter()
    # hist_match.SetThresholdAtMeanIntensity(True)
    # img = hist_match.Execute(img, img)
    img = img_respacing(img)
    img = img_pad(img)
    sitk.WriteImage(img, ROOT_dir + 'temp.nii')

    # # #--------------------------------------------------------------------------------------
    # # # # crop here, crop range based on temp.nii
    # arr = sitk.GetArrayFromImage(sitk.ReadImage(ROOT_dir + 'temp.nii'))
    # result_img = arr[0:256, 0:256, 0:256] # dir:ZYX
    # result_img = sitk.GetImageFromArray(result_img)
    #
    # # hist_match2 = sitk.HistogramMatchingImageFilter()
    # # hist_match2.SetThresholdAtMeanIntensity(True)
    # # result_img = hist_match2.Execute(result_img, result_img)
    # sitk.WriteImage(result_img, ROOT_dir + 'result_img.nii')

    # # #--------------------------------------------------------------------------------------
    # # # # resize to 256*256
    # origin_xray1 = Image.open(ROOT_dir + 'drr01.bmp')
    # resized_xray1 = origin_xray1.resize([256, 256])
    # resized_xray1.save(ROOT_dir + 'resized_drr01.bmp')
    # origin_xray2 = Image.open(ROOT_dir + 'drr02.bmp')
    # resized_xray2 = origin_xray2.resize([256, 256])
    # resized_xray2.save(ROOT_dir + 'resized_drr02.bmp')
    #
    # # # # concatenate from here
    # result_img = sitk.ReadImage(ROOT_dir + 'result_img.nii')
    # np_result_img = sitk.GetArrayFromImage(result_img)
    # xray1 = Image.open(ROOT_dir + 'resized_drr01.bmp')
    # xray2 = Image.open(ROOT_dir + 'resized_drr02.bmp')
    #
    # f = h5py.File(ROOT_dir + 'ct_xray_data.h5', 'w')
    # f['ct'] = np_result_img
    # f['ori_size'] = np.int64(320)
    # f['spacing'] = [1.0, 1.0, 1.0]
    # f['xray1'] = xray1
    # f['xray2'] = xray2
    # f.close()