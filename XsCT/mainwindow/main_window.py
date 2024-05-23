import sys
import os
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.vtkImageImportFromArray import *
from PyQt5.QtWidgets import QFileDialog
from vtkmodules.util import numpy_support
import numpy as np

class Ui_MainWindow(QtWidgets.QWidget):
    sign_one = QtCore.pyqtSignal(str)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # x-ray 視窗
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(15)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.horizontalLayout.addLayout(self.verticalLayout_3)

        # VTK 視窗
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(15)
        self.verticalLayout.setObjectName("verticalLayout")
        self.xray_widget = QtWidgets.QWidget(self.centralwidget)
        self.pred_widget = QtWidgets.QWidget(self.centralwidget)
        self.ren_xray, self.frame1, self.vtkWidget_xray, self.iren_xray, self.render_window1, self.actor_xray = self.setup()
        self.ren_pred, self.frame2, self.vtkWidget_pred, self.iren_pred, self.render_window2, self.actor_pred = self.setup()
        self.verticalLayout.addWidget(self.vtkWidget_xray)
        self.verticalLayout.addWidget(self.vtkWidget_pred)
        self.horizontalLayout.addLayout(self.verticalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_file = QtWidgets.QAction(MainWindow)
        self.actionLoad_file.setObjectName("actionLoad_file")
        self.menuFile.addAction(self.actionLoad_file)
        self.menubar.addAction(self.menuFile.menuAction())
        self.actionLoad_file.triggered.connect(self.slot_btn_chooseDir)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setup(self):
        """
        Create and setup the base vtk and Qt objects for the application
        """
        renderer = vtk.vtkRenderer()
        frame = QtWidgets.QFrame()
        vtk_widget = QVTKRenderWindowInteractor()
        interactor = vtk_widget.GetRenderWindow().GetInteractor()
        render_window = vtk_widget.GetRenderWindow()

        frame.setAutoFillBackground(True)
        vtk_widget.GetRenderWindow().AddRenderer(renderer)
        render_window.AddRenderer(renderer)
        interactor.SetRenderWindow(render_window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        actor = vtk.vtkActor()
        renderer.AddActor(actor)

        # required to enable overlapping actors with opacity < 1.0
        # this is causing some issues with flashing objects
        # render_window.SetAlphaBitPlanes(1)
        # render_window.SetMultiSamples(0)
        # renderer.UseDepthPeelingOn()
        # renderer.SetMaximumNumberOfPeels(2)

        return renderer, frame, vtk_widget, interactor, render_window, actor

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_3.setText(_translate("MainWindow", "TextLabel"))
        self.label_4.setText(_translate("MainWindow", "TextLabel"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_file.setText(_translate("MainWindow", "Load file"))

    def add_ct(self, ct_list):

        # numpy_data is a 3D numpy array
        ct_gt = self.npToVTK(np.squeeze(ct_list[0]))
        ct_pred = self.npToVTK(np.squeeze(ct_list[1]))

        gt_smooth = self.smoother(ct_gt)
        pred_smooth = self.smoother(ct_pred)
        gt_mapper = self.mapper(gt_smooth)
        pred_mapper = self.mapper(pred_smooth)

        self.actor_xray.SetMapper(gt_mapper)
        self.actor_pred.SetMapper(pred_mapper)
        # self.actor_xray.GetProperty().SetDiffuseColor(1, .94, .25)
        # self.actor_pred.GetProperty().SetDiffuseColor(1, .94, .25)

        # self.show()
        self.iren_xray.Initialize()
        self.iren_pred.Initialize()

    def mapper(self, smoother):
        # lut = make_colors(n)

        # mapper = vtk.vtkImageMapper3D()
        # mapper.SetInputData(smoother.GetOutput())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(smoother.GetOutputPort())
        # mapper.SetLookupTable(lut)
        # mapper.SetScalarRange(0, lut.GetNumberOfColors())
        return mapper

    def smoother(self, vtk_data):
        n = 20
        discrete = vtk.vtkFlyingEdges3D()
        # discrete.SetInputData(vtk_data.GetOutputPort())
        discrete.SetInputConnection(vtk_data.GetOutputPort())
        discrete.ComputeScalarsOff()
        discrete.ComputeGradientsOff()
        discrete.ComputeNormalsOn()
        discrete.SetValue(0, 500)

        # marchingCubes = vtk.vtkMarchingCubes()
        # marchingCubes.SetInputConnection(vtk_data.GetOutputPort())
        # marchingCubes.SetValue(0, 100)

        # smoothing_iterations = 15
        # pass_band = 0.001
        # feature_angle = 120.0
        #
        # smoother = vtk.vtkWindowedSincPolyDataFilter()
        # smoother.SetInputConnection(discrete.GetOutputPort())
        # smoother.SetNumberOfIterations(smoothing_iterations)
        # smoother.BoundarySmoothingOff()
        # smoother.FeatureEdgeSmoothingOff()
        # smoother.SetFeatureAngle(feature_angle)
        # smoother.SetPassBand(pass_band)
        # smoother.NonManifoldSmoothingOn()
        # smoother.NormalizeCoordinatesOn()
        # smoother.Update()
        return discrete

    def npToVTK(self, np_data):
        # shape = np_data.shape[::-1]
        # vtk_data = numpy_support.numpy_to_vtk(np_data.flatten(), 1, vtk.VTK_SHORT)
        #
        # vtk_image_data = vtk.vtkImageData()
        # vtk_image_data.SetDimensions(shape)
        # vtk_image_data.SetSpacing([1, 1, 1])
        # vtk_image_data.SetOrigin([0, 0, 0])
        # vtk_image_data.GetPointData().SetScalars(vtk_data)
        min_val = np.min(np_data)
        max_val = np.max(np_data)
        np_data = ((np_data - min_val) / (max_val - min_val)) * 1000
        vtk_image_data = vtkImageImportFromArray()
        vtk_image_data.SetArray(np_data)
        spacing = [1, 1, 1]
        vtk_image_data.SetDataSpacing(spacing)  # 设置spacing
        origin = (0, 0, 0)
        vtk_image_data.SetDataOrigin(origin)  # 设置vtk数据的坐标系原点
        vtk_image_data.Update()
        return vtk_image_data

    def add_source(self, ct_list):
        # Create source
        source = vtk.vtkConeSource()
        source.SetCenter(0, 0, 0)
        source.SetRadius(0.1)

        source1 = vtk.vtkSphereSource()
        source1.SetCenter(0, 0, 0)
        source1.SetRadius(0.3)

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        mapper1 = vtk.vtkPolyDataMapper()
        mapper1.SetInputConnection(source1.GetOutputPort())

        # Create an actor
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        #
        # actor1 = vtk.vtkActor()
        # actor1.SetMapper(mapper1)
        #
        # self.ren_xray.AddActor(actor)
        # self.ren_pred.AddActor(actor1)
        #
        # self.ren_xray.ResetCamera()
        # self.ren_pred.ResetCamera()

        # self.frame.setLayout(self.vl)
        # self.setCentralWidget(self.frame)

        self.actor_xray.SetMapper(mapper)
        self.actor_pred.SetMapper(mapper1)

        # self.show()
        # self.iren_xray.Initialize()
        # self.iren_pred.Initialize()

    def slot_btn_chooseDir(self):
        dir_choose = QFileDialog.getExistingDirectory(self.centralwidget, "选取文件夹", os.getcwd())  # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件夹为:")
        print(dir_choose)
        self.sign_one.emit(dir_choose)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    a = []
    ui.add_source(a)
    MainWindow.show()
    sys.exit(app.exec())