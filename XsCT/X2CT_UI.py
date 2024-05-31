
import argparse
from libs.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from libs.model.factory import get_model
from libs.utils.visualizer import tensor_back_to_unnormalization, save_images, tensor_back_to_unMinMax
import copy
import torch
import numpy as np
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from mainwindow.main_window import Ui_MainWindow
import h5py
from libs.dataset.data_augmentation import CT_XRAY_Data_Augmentation_Multi

import sys
module_path = os.path.abspath(os.getcwd() + '\\..')
if module_path not in sys.path:
    sys.path.append(module_path)
x2ct_input = []

def parse_args():
  parse = argparse.ArgumentParser(description='X2CT_UI')
  parse.add_argument('--data', type=str, default='mesh_data_128_sawbone_v2', dest='data',
                     help='input data ')
  parse.add_argument('--tag', type=str, default='multiview', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--gpu', type=str, default='0', dest='gpuid',
                     help='')
  parse.add_argument('--model_class', type=str, default='MultiViewCTGAN', dest='model_class',
                     help='')
  parse.add_argument('--check_point', type=str, default=100, dest='check_point',
                     help='checkpoint model to load')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  args = parse.parse_args()
  return args

class X2CT(QtWidgets.QWidget):
  show_ct = QtCore.pyqtSignal(list)
  def __init__(self):
    super(X2CT, self).__init__()
    args = parse_args()
    # check gpu
    if torch.cuda.is_available():
      split_gpu = str(args.gpuid).split(',')
      args.gpu_ids = [int(i) for i in split_gpu]
    else:
      print('no gpu')
      exit(0)

    # check point
    if args.check_point is None:
      args.epoch_count = 1
    else:
      args.epoch_count = int(args.check_point)

    # merge config with yaml
    ymlpath = "C:\\XsCT\\XsCT\\view_mode\\multiview.yml"
    cfg_from_yaml(ymlpath)

    # merge config with argparse
    self.opt = copy.deepcopy(cfg)
    self.opt = merge_dict_and_yaml(args.__dict__, self.opt)
    print_easy_dict(self.opt)

    self.opt.serial_batches = True

    # get model
    self.gan_model = get_model(self.opt.model_class)()
    print('taking --{}-- model'.format(self.gan_model.name))

    # set to test
    self.gan_model.eval()

    self.gan_model.init_process(self.opt)
    total_steps, epoch_count = self.gan_model.setup(self.opt)

    # must set to test mode again, due to omission of assigning mode to network layers
    # model.training is test, but BN.training is training
    if self.opt.verbose:
      print('model mode: {}'.format('training' if self.gan_model.training else 'testing'))
      for i, v in self.gan_model.named_modules():
        print(i, v.training)

    if 'batch' in self.opt.norm_G:
      self.gan_model.eval()
    elif 'instance' in self.opt.norm_G:
      self.gan_model.eval()
      # instance norm in training mode is better
      for name, m in self.gan_model.named_modules():
        if m.__class__.__name__.startswith('InstanceNorm'):
          m.train()
    else:
      raise NotImplementedError()

    if self.opt.verbose:
      print('change to model mode: {}'.format('training' if self.gan_model.training else 'testing'))
      for i, v in self.gan_model.named_modules():
        print(i, v.training)

    self.data_augmentation = CT_XRAY_Data_Augmentation_Multi(self.opt)

  def get_ct(self, input):
    with torch.no_grad():
      # model predicted CT from x-ray
      self.gan_model.set_input(input)
      self.gan_model.test()

      visuals = self.gan_model.get_current_visuals()
      img_path = self.gan_model.get_image_paths()

      # CT Source
      #
      generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
      real_CT = visuals['G_real'].data.clone().cpu().numpy()
      # To NDHW
      generate_CT_transpose = generate_CT
      real_CT_transpose = real_CT
      # Inveser Deepth
      generate_CT_transpose = generate_CT_transpose[:, ::-1, :, :]
      real_CT_transpose = real_CT_transpose[:, :, :, :]
      # To [0, 1]
      generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, self.opt.CT_MEAN_STD[0],
                                                             self.opt.CT_MEAN_STD[1])
      real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1])
      # Clip generate_CT
      generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

      # To HU coordinate
      generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, self.opt.CT_MIN_MAX[0],
                                                      self.opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
      real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, self.opt.CT_MIN_MAX[0], self.opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
      return generate_CT_transpose, real_CT_transpose

  def load_file(self, file_path):
    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])
    x_ray1 = np.asarray(hdf5['xray1'])
    x_ray2 = np.asarray(hdf5['xray2'])
    x_ray1 = np.expand_dims(x_ray1, 0)
    x_ray2 = np.expand_dims(x_ray2, 0)
    hdf5.close()
    return ct_data, x_ray1, x_ray2

  def pull_item(self, ct_data, x_ray1, x_ray2, file_path):
    # Data Augmentation
    ct, xray1, xray2 = self.data_augmentation.__call__([ct_data, x_ray1, x_ray2])

    return torch.stack([ct]), [torch.stack([xray1]), torch.stack([xray2])], file_path

  def getImgfromDir_AP(self, dirname):
    file_path = os.path.join(dirname, 'ct_xray_data.h5')
    hdf5 = h5py.File(file_path, 'r')
    self.ct_data = np.asarray(hdf5['ct'])
    self.x_ray1 = np.asarray(hdf5['xray1'])
    self.x_ray2 = np.asarray(hdf5['xray2'])
    self.x_ray1 = np.expand_dims(self.x_ray1, 0)
    self.x_ray2 = np.expand_dims(self.x_ray2, 0)
    hdf5.close()
    input = self.pull_item(self.ct_data, self.x_ray1, self.x_ray2, file_path)
    generate_CT_transpose, real_CT_transpose = self.get_ct(input)
    self.show_ct.emit([generate_CT_transpose, real_CT_transpose])

    


if __name__ == '__main__':
  app = QtWidgets.QApplication([])
  MainWindow = QtWidgets.QMainWindow()
  ui = Ui_MainWindow()
  ui.setupUi(MainWindow)
  model = X2CT()
  ui.sign_AP.connect(model.getImgfromDir)
  ui.sign_LAT.connect(model.getImgfromDir)
  model.show_ct.connect(ui.add_ct)
  MainWindow.show()

  sys.exit(app.exec())
