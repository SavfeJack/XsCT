

import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import h5py
from PIL import Image
import os
import glob
import csv

def crop_ROI(input_img, seg_img):
    volume = sitk.GetArrayFromImage(seg_img)
    for i in range(0, volume.shape[0]):
        result = volume[i, :, :]
        if result.any():
            min_x = i
            # print("axis" + str(0) + " min:" + str(i))
            break

    for i in range(volume.shape[0] - 1, 0, -1):
        result = volume[i, :, :]
        if result.any():
            max_x = i
            # print("axis" + str(0) + " max:" + str(i))
            break

    for i in range(0, volume.shape[1]):
        result = volume[:, i, :]
        if result.any():
            min_y = i
            # print("axis" + str(1) + " min:" + str(i))
            break

    for i in range(volume.shape[1] - 1, 0, -1):
        result = volume[:, i, :]
        if result.any():
            max_y = i
            # print("axis" + str(1) + " max:" + str(i))
            break

    for i in range(0, volume.shape[2]):
        result = volume[:, :, i]
        if result.any():
            min_z = i
            # print("axis" + str(2) + " min:" + str(i))
            break

    for i in range(volume.shape[2] - 1, 0, -1):
        result = volume[:, :, i]
        if result.any():
            max_z = i
            # print("axis" + str(2) + " max:" + str(i))
            break

    print("axisX: " + str(min_x) + ", " + str(max_x))
    print("axisY: " + str(min_y) + ", " + str(max_y))
    print("axisZ: " + str(min_z) + ", " + str(max_z))
    avg_y = round((min_y + max_y) / 2)
    avg_z = round((min_z + max_z) / 2)
    avg_x = round((min_x + max_x) / 2)
    image_origin = sitk.GetArrayFromImage(input_img)
    result = image_origin[max(0, (avg_x - 200)):min(image_origin.shape[0]-1, (avg_x + 200)), max(0, (avg_y - 200)):min(image_origin.shape[1]-1, (avg_y + 200)), max(0, (avg_z - 200)):min(image_origin.shape[2]-1, (avg_z + 200))]
    result = sitk.GetImageFromArray(result)
    return result
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

def img_resizeing(input_img, target_size=[256.0, 256.0, 256.0], resamplemethod=sitk.sitkLinear): # sitk.sitkNearestNeighbor
    origin_size = input_img.GetSize()
    origin_spacing = input_img.GetSpacing()

    new_spacing = [
        round((origin_size[0] * origin_spacing[0]) / target_size[0]),
        round((origin_size[1] * origin_spacing[1]) / target_size[1]),
        round((origin_size[2] * origin_spacing[2]) / target_size[2])
    ]

    new_size = target_size
    print('new size: ', new_size)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(input_img.GetOrigin())
    resampler.SetOutputDirection(input_img.GetDirection())
    resampler.SetOutputSpacing(new_spacing)

    # Set different type according to the need to resample the image
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(input_img.GetPixelID())
    else:
        resampler.SetOutputPixelType(input_img.GetPixelID())# Linear interpolation is used for PET/CT/MRI and the like

    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    output_img = resampler.Execute(input_img)
    return output_img

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
    # # ROOT_dir = 'C:\\Users\\user\\Desktop\\temp\\'
    # Save_dir = 'C:\\XsCT\\XsCT\\data\\COLONOgraphy\\crop_data\\'
    # ROOT_dir = 'C:\\XsCT\\XsCT\\data\\COLONOgraphy\\manifest-sFI3R7DS3069120899390652954\\CT COLONOGRAPHY\\'
    # seg_dir = glob.glob(os.path.join('C:\\XsCT\\XsCT\\data\\COLONOgraphy\\CTSpine1K\\trainset_all', "*"))
    # num = 1
    #
    # # 開啟 CSV 檔案
    # with open('C:\\XsCT\\XsCT\\data\\COLONOgraphy\\manifest-sFI3R7DS3069120899390652954\\Path.csv', newline='') as csvfile:
    #
    #     # 讀取 CSV 檔案內容
    #     rows = csv.reader(csvfile)
    #
    #     # 以迴圈輸出每一列
    #     for row in rows:
    #         if num == 5:
    #             num += 1
    #             continue
    #         print(row)
    #         data_directory = ROOT_dir + row[0]
    #         series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    #         if not series_IDs:
    #             print("ERROR: given directory \"" + data_directory + "\" does not contain a DICOM series.")
    #
    #         series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])
    #
    #         series_reader = sitk.ImageSeriesReader()
    #         series_reader.SetFileNames(series_file_names)
    #
    #         # Configure the reader to load all of the DICOM tags (public+private):
    #         # By default tags are not loaded (saves time).
    #         # By default if tags are loaded, the private tags are not loaded.
    #         # We explicitly configure the reader to load tags, including the
    #         # private ones.
    #         series_reader.MetaDataDictionaryArrayUpdateOn()
    #         series_reader.LoadPrivateTagsOn()
    #         image3D = series_reader.Execute()
    #         size = np.array(list(reversed(image3D.GetSize())))
    #         spacing = np.array(list(reversed(image3D.GetSpacing())))
    #         print('size: ', size)
    #         print('spacing: ', spacing)
    #
    #         seg_img = sitk.ReadImage(seg_dir[num-1])
    #
    #         img = img_respacing(image3D)
    #         seg_img = img_respacing(seg_img)
    #         img = img_pad(img)
    #         seg_img = img_pad(seg_img)
    #         img = crop_ROI(img, seg_img)
    #
    #         arr = sitk.GetArrayFromImage(img)
    #         result_img = arr[int((arr.shape[0]/2)-128):int((arr.shape[0]/2)+128), int((arr.shape[1]/2)-128):int((arr.shape[1]/2)+128), int((arr.shape[1]/2)-128):int((arr.shape[1]/2)+128)]
    #         result_img = sitk.GetImageFromArray(result_img)
    #
    #         path = Save_dir + 'mesh_data_%03d\\' % num
    #         if not os.path.exists(path):
    #             os.mkdir(path)
    #         # sitk.WriteImage(img, path + 'temp.nii')
    #         sitk.WriteImage(result_img, path + 'result_img.nii')
    #         num += 1

    # # #--------------------------------------------------------------------------------------
    # # # # read and process
    # img = sitk.ReadImage(ROOT_dir + "\\4580220\\4580220.mhd", sitk.sitkFloat32)
    # # image_array = sitk.GetImageFromArray(img)
    # size = np.array(list(reversed(img.GetSize())))
    # spacing = np.array(list(reversed(img.GetSpacing())))
    # print('size: ', size)
    # print('spacing: ', spacing)
    # # image_array.SetOrigin(origin)
    # # spacing.reverse()
    # # image_array.SetSpacing(spacing)
    #
    # # hist_match = sitk.HistogramMatchingImageFilter()
    # # hist_match.SetThresholdAtMeanIntensity(True)
    # # img = hist_match.Execute(img, img)
    # img = img_respacing(img)
    # img = img_pad(img)
    # sitk.WriteImage(img, ROOT_dir + 'temp.nii')

    # # #--------------------------------------------------------------------------------------
    # # # # crop here, crop range based on temp.nii
    # arr = sitk.GetArrayFromImage(sitk.ReadImage('C:\\XsCT\\XsCT\\data\\COLONOgraphy\\crop_data\\' + 'temp.nii'))
    # result_img = arr[0:256, 0:256, 0:256] # dir:ZYX
    # result_img = sitk.GetImageFromArray(result_img)
    #
    # # hist_match2 = sitk.HistogramMatchingImageFilter()
    # # hist_match2.SetThresholdAtMeanIntensity(True)
    # # result_img = hist_match2.Execute(result_img, result_img)
    # sitk.WriteImage(result_img, 'C:\\XsCT\\XsCT\\data\\COLONOgraphy\\crop_data\\' + 'result_img.nii')

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
    ROOT_dir = 'C:\\XsCT\\XsCT\\data\\COLONOgraphy\\mesh_data_256'
    Path = glob.glob(os.path.join(ROOT_dir, '*'))
    for num in range(924):
        ROOT_dir = Path[num]
        result_img = sitk.ReadImage(ROOT_dir + '\\result_img.nii')
        np_result_img = sitk.GetArrayFromImage(result_img)
        xray1 = Image.open(ROOT_dir + '\\resized_drr01.bmp')
        xray2 = Image.open(ROOT_dir + '\\resized_drr02.bmp')

        f = h5py.File(ROOT_dir + '\\ct_xray_data.h5', 'w')
        f['ct'] = np_result_img
        f['ori_size'] = np.int64(320)
        f['spacing'] = [1.0, 1.0, 1.0]
        f['xray1'] = xray1
        f['xray2'] = xray2
        f.close()