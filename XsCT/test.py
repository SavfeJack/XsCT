

import argparse
from libs.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from libs.dataset.factory import get_dataset
from libs.model.factory import get_model
from libs.utils.visualizer import tensor_back_to_unnormalization, tensor_back_to_unMinMax
from libs.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity, Peak_Signal_to_Noise_Rate_3D
from libs.utils import html
from libs.utils import ct as CT
import copy
import tqdm
import torch
import numpy as np
import os

import sys
module_path = os.path.abspath(os.getcwd() + '\\..')
if module_path not in sys.path:
    sys.path.append(module_path)


def parse_args():
  parse = argparse.ArgumentParser(description='XsCT')
  parse.add_argument('--data', type=str, default='mesh_data', dest='data',
                     help='input data')
  parse.add_argument('--tag', type=str, default='multiview', dest='tag',
                     help='distinct from other try')
  parse.add_argument('--dataroot', type=str, default='./data/mesh_data', dest='dataroot',
                     help='input data root')
  parse.add_argument('--dataset', type=str, default='test', dest='dataset',
                     help='')
  parse.add_argument('--datasetfile', type=str, default='./data/test.txt', dest='datasetfile',
                     help='')
  parse.add_argument('--ymlpath', type=str, default='./view_mode/multiview.yml', dest='ymlpath',
                     help='')
  parse.add_argument('--gpu', type=str, default='0', dest='gpuid',
                     help='')
  parse.add_argument('--dataset_class', type=str, default='align_ct_xray_views_std', dest='dataset_class',
                     help='')
  parse.add_argument('--model_class', type=str, default='MultiViewCTGAN', dest='model_class',
                     help='')
  parse.add_argument('--check_point', type=str, default=100, dest='check_point',
                     help='checkpoint model to load')
  parse.add_argument('--latest', action='store_true', dest='latest',
                     help='set to use latest cached model')
  parse.add_argument('--verbose', action='store_true', dest='verbose',
                     help='if specified, print more debugging information')
  parse.add_argument('--load_path', type=str, default=None, dest='load_path',
                     help='if load_path is not None, model will load from load_path')
  parse.add_argument('--how_many', type=int, dest='how_many', default=None,
                     help='if specified, only run this number of test samples for visualization')
  parse.add_argument('--resultdir', type=str, default='./result/multiview', dest='resultdir',
                     help='dir to save result')
  args = parse.parse_args()
  return args


def evaluate(args):
  # check gpu
  if args.gpuid == '':
    args.gpu_ids = []
  else:
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
  if args.ymlpath is not None:
    cfg_from_yaml(args.ymlpath)
  # merge config with argparse
  opt = copy.deepcopy(cfg)
  opt = merge_dict_and_yaml(args.__dict__, opt)
  print_easy_dict(opt)

  opt.serial_batches = True

  # add data_augmentation
  datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
  opt.data_augmentation = dataTestClass

  # get dataset
  dataset = datasetClass(opt)
  print('dataset {}'.format(dataset.name))
  dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=int(opt.nThreads),
    collate_fn=collateClass)

  dataset_size = len(dataloader)
  print('test images = %d' % dataset_size)

  # get model
  gan_model = get_model(opt.model_class)()
  print('taking --{}-- model'.format(gan_model.name))

  # set to test
  gan_model.eval()

  gan_model.init_process(opt)
  total_steps, epoch_count = gan_model.setup(opt)

  # must set to test mode again, due to omission of assigning mode to network layers
  # model.training is test, but BN.training is training
  if opt.verbose:
    print('model mode: {}'.format('training' if gan_model.training else 'testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  if 'batch' in opt.norm_G:
    gan_model.eval()
  elif 'instance' in opt.norm_G:
    gan_model.eval()
    # instance norm in training mode is better
    for name, m in gan_model.named_modules():
      if m.__class__.__name__.startswith('InstanceNorm'):
        m.train()
  else:
    raise NotImplementedError()

  if opt.verbose:
    print('change to model mode: {}'.format('training' if gan_model.training else 'testing'))
    for i, v in gan_model.named_modules():
      print(i, v.training)

  result_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  avg_dict = dict()
  for epoch_i, data in tqdm.tqdm(enumerate(dataloader)):

    gan_model.set_input(data)
    gan_model.test()

    visuals = gan_model.get_current_visuals()
    img_path = gan_model.get_image_paths()

    #
    # Evaluate Part
    #
    generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
    real_CT = visuals['G_real'].data.clone().cpu().numpy()
    # To [0, 1]
    # To NDHW
    if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
      generate_CT_transpose = generate_CT
      real_CT_transpose = real_CT
    else:
      generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
      real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
    generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                           opt.CT_MEAN_STD[1])
    real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0], opt.CT_MEAN_STD[1])
    # clip generate_CT
    generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

    # CT range 0-1
    mae0 = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse0 = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
    cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
    ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
    # CT range 0-4096
    generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
      np.int32)
    real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
      np.int32)
    psnr_3d = Peak_Signal_to_Noise_Rate_3D(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
    psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=4095)
    mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
    mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)

    name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
    name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
    name = name2 + '_' + name1
    print(cosinesimilarity, name)
    if cosinesimilarity is np.nan or cosinesimilarity > 1:
      print(os.path.splitext(os.path.basename(gan_model.get_image_paths()[0][0]))[0])
      continue

    metrics_list = [('MAE0', mae0), ('MSE0', mse0), ('MAE', mae), ('MSE', mse), ('CosineSimilarity', cosinesimilarity),
                    ('psnr-3d', psnr_3d), ('PSNR-1', psnr[0]),
                    ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]),
                    ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3])]

    for key, value in metrics_list:
      if avg_dict.get(key) is None:
        avg_dict[key] = [] + value.tolist()
      else:
        avg_dict[key].extend(value.tolist())

    # Save
    web_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
    # webpage = html.HTML(web_dir, 'experiment = %s, phase = %s, epoch = %s' % (opt.data, opt.dataset, opt.check_point))
    ctVisual = CT.CTVisual()
    image_root = os.path.join(web_dir, 'CT', name)
    if not os.path.exists(image_root):
      os.makedirs(image_root)
    save_path = os.path.join(image_root, 'fake_ct.mha')
    ctVisual.save(generate_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), path=save_path)
    save_path = os.path.join(image_root, 'real_ct.mha')
    ctVisual.save(real_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), path=save_path)

    del visuals, img_path

  for key, value in avg_dict.items():
    print('### --{}-- total: {}; avg: {} '.format(key, len(value), np.round(np.mean(value), 7)))
    avg_dict[key] = np.mean(value)

  return avg_dict


if __name__ == '__main__':
  args = parse_args()
  evaluate(args)
