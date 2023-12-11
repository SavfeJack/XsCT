

import SimpleITK as sitk
import vtk
from vtkmodules.util import numpy_support


def numpyToVTK(data):
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetDimensions(shape[0], shape[1], shape[2])
    return img


if __name__ == "__main__":
    ROOT_dir = 'C:\\Users\\user\\Desktop\\temp\\'

    originCT = sitk.ReadImage(ROOT_dir + "real_ct.mha")
    generatedCT = sitk.ReadImage(ROOT_dir + "fake_ct.mha")

    viewer1 = sitk.ImageViewer()
    viewer1.SetTitle('originCT')
    viewer1.SetApplication(r'C:\Program Files\ITK-SNAP 3.6\bin\ITK-SNAP.exe')
    viewer1.Execute(originCT)

    viewer2 = sitk.ImageViewer()
    viewer2.SetTitle('generatedCT')
    viewer2.SetApplication(r'C:\Program Files\ITK-SNAP 3.6\bin\ITK-SNAP.exe')
    viewer2.Execute(generatedCT)

    # viewer1 = sitk.ImageViewer()
    # viewer1.SetTitle('originCT')
    # viewer1.SetApplication(r'C:\Users\user\AppData\Local\NA-MIC\Slicer 5.2.2\Slicer.exe')
    # viewer1.Execute(originCT)
    #
    # viewer2 = sitk.ImageViewer()
    # viewer2.SetTitle('generatedCT')
    # viewer2.SetApplication(r'C:\Users\user\AppData\Local\NA-MIC\Slicer 5.2.2\Slicer.exe')
    # viewer2.Execute(generatedCT)