import math
import sys
import numpy as np
import itk
import itkwidgets
from itkwidgets import view
from ipywidgets import interactive
import ipywidgets as widgets
import matplotlib.pyplot as plt
import glob
import os
from skimage import io

# 加载原始体数据并显示
ROOT_dir = 'C:\\XsCT\\XsCT\\data\\mesh_data_256' # 'C:\\XsCT\\XsCT\\data\\mesh_data_256'
Path = glob.glob(os.path.join(ROOT_dir, '*'))
# data_APtrans = [5,9,15,25,28,31,32,38,40,41,45,46,47,48,51,55,60,61,66,82,83,84,86,88,89,91,95,96,98,101,102,103,104,117,118,119,120,125,127,142,145,161,162,165,167,173,178,179,181,183,187,188,191,192,205,216,220,227,232,234,235,240,243,247,250,251,252,255,258,259,264,265,266,268,274,275,280,284,300,323,325,327,334,336,338,339,340,343,363,372,373,374,375,380,383,387,388,389,390,392,393,394,395,413,414,415,417,419,422,427,434,436,437,445,452,454,455,456,458,463,464,469,470,472,474,475,476,490,491,492,499,512,513,515,516,518,521,528,529,535,536,538,541,542,543,545,547,549,550,552,559,560,561,562,563,566,567,586,587,589,591,593,597,599,601,603,608,609,610,630,632,636,639,641,643,646,647,649,651,658,659,689,690,698,718,722,724,726,727,729,730,731,736,738,747,751,752,756,758,762,764,766,767,768,770,776,777,778,780,783]

def DigitallyReconstructedRadiograph(
        ray_source_distance=5000,
        camera_tx=0.,
        camera_ty=0.,
        camera_tz=-15.,
        rotation_x=-90.,
        rotation_y=0.,
        rotation_z=90.,
        projection_normal_p_x=0.,
        projection_normal_p_y=0.,
        rotation_center_rt_volume_center_x=0.,
        rotation_center_rt_volume_center_y=0.,
        rotation_center_rt_volume_center_z=0.,
        threshold=0.,
):
    """
    Parameters description:

    ray_source_distance = 400                              # <-sid float>            Distance of ray source (focal point) focal point 400mm
    camera_translation_parameter = [0., 0., 0.]            # <-t float float float>  Translation parameter of the camera
    rotation_around_xyz = [0., 0., 0.]                     # <-rx float>             Rotation around x,y,z axis in degrees
    projection_normal_position = [0, 0]                    # <-normal float float>   The 2D projection normal position [default: 0x0mm]
    rotation_center_relative_to_volume_center = [0, 0, 0]  # <-cor float float float> The centre of rotation relative to centre of volume
    threshold = 10                                          # <-threshold float>      Threshold [default: 0]
    """

    dgree_to_radius_coef = 1. / 180. * math.pi
    camera_translation_parameter = [camera_tx, camera_ty, camera_tz]
    rotation_around_xyz = [rotation_x * dgree_to_radius_coef, rotation_y * dgree_to_radius_coef,
                           rotation_z * dgree_to_radius_coef]
    projection_normal_position = [projection_normal_p_x, projection_normal_p_y]
    rotation_center_relative_to_volume_center = [
        rotation_center_rt_volume_center_x,
        rotation_center_rt_volume_center_y,
        rotation_center_rt_volume_center_z
    ]

    imageOrigin = volume_lung.GetOrigin()
    imageSpacing = volume_lung.GetSpacing()
    imageRegion = volume_lung.GetBufferedRegion()
    imageSize = imageRegion.GetSize()
    imageCenter = [imageOrigin[i] + imageSpacing[i] * imageSize[i] / 2.0 for i in range(3)]

    transform.SetTranslation(camera_translation_parameter)
    transform.SetRotation(rotation_around_xyz[0], rotation_around_xyz[1], rotation_around_xyz[2])

    center = [c + imageCenter[i] for i, c in enumerate(rotation_center_relative_to_volume_center)]
    transform.SetCenter(center)

    interpolator.SetTransform(transform)
    interpolator.SetThreshold(threshold)
    focalPoint = [imageCenter[0], imageCenter[1], imageCenter[2] - ray_source_distance / 2.0]
    interpolator.SetFocalPoint(focalPoint)

    filter.SetInterpolator(interpolator)
    filter.SetTransform(transform)

    origin = [
        imageCenter[0] + projection_normal_position[0] - output_image_pixel_spacing[0] * (
                    output_image_size[0] - 1) / 2.,
        imageCenter[1] + projection_normal_position[1] - output_image_pixel_spacing[1] * (
                    output_image_size[1] - 1) / 2.,
        imageCenter[2] + imageSpacing[2] * imageSize[2]
    ]

    filter.SetOutputOrigin(origin)
    filter.Update()

    # plt.imshow(np.squeeze(filter.GetOutput()))
    # plt.show()

    # print informations
    print("Volume image informations:")
    print("\tvolume image origin : ", imageOrigin)
    print("\tvolume image size   : ", imageSize)
    print("\tvolume image spacing: ", imageSpacing)
    print("\tvolume image center : ", imageCenter)
    print("Transform informations:")
    print("\ttranslation         : ", camera_translation_parameter)
    print("\trotation            : ", rotation_around_xyz)
    print("\tcenter               : ", center)
    print("Interpolator informations: ")
    print("\tthreshold           : ", threshold)
    print("\tfocalPoint          : ", focalPoint)
    print("Filter informations:")
    print("\toutput origin        : ", origin)
    return filter.GetOutput()

# for path in Path:
for num in range(1):
    save = 'C:\\XsCT\\XsCT\\data\\sawbone\\sawbone_data_%04d' % (num+1)
    input_name = save + '\\result_img.nii'
    save_path = save + "\\resized_drr02.bmp"
    # save_ct = save + '\\result_img.nii'

    volume_lung = itk.imread(input_name, itk.ctype('float'))  # 读取影像文件，并将数据格式转换为float
    print(volume_lung.GetLargestPossibleRegion().GetSize())
    print(volume_lung.GetBufferedRegion().GetSize())
    print(volume_lung.GetSpacing())
    print(volume_lung.GetOrigin())
    # view(volume_lung, gradient_opacity=0.5, cmp=itkwidgets.cm.bone)

    output_image_pixel_spacing = [1, 1, 1]
    output_image_size = [256, 256, 1]  # [501, 501, 1]

    # volume_lung = itk.GetArrayFromImage(volume_lung)
    # volume_lung = volume_lung[:,::-1,:]
    # volume_lung = itk.GetImageFromArray(volume_lung)
    InputImageType = type(volume_lung)
    FilterType = itk.ResampleImageFilter[InputImageType, InputImageType]
    filter = FilterType.New()
    filter.SetInput(volume_lung)
    filter.SetDefaultPixelValue(0)
    filter.SetSize(output_image_size)
    filter.SetOutputSpacing(output_image_pixel_spacing)

    TransformType = itk.CenteredEuler3DTransform[itk.D]
    transform = TransformType.New()
    transform.SetComputeZYX(True)

    InterpolatorType = itk.RayCastInterpolateImageFunction[InputImageType, itk.D]
    interpolator = InterpolatorType.New()

    viewer = None
    img = DigitallyReconstructedRadiograph()
    img = itk.rescale_intensity_image_filter(img, output_minimum=0, output_maximum=255)
    img = itk.GetArrayFromImage(img)
    img = np.squeeze(img)
    img = img.astype(np.uint8)
    # plt.imshow(img)
    # plt.show()
    if not os.path.exists(save):
        os.makedirs(save)
    io.imsave(save_path, img)
    # itk.imwrite(volume_lung, save_ct)

