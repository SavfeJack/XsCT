# XsCT
**a generative(GAN) model which is capable to generate sCT data from X-ray image** 

### Requirements:
1. pytorch=1.11.0
2. python=3.7
4. CUDA, cudnn=1.11.3
5. other python dependencies descripted under requirements.txt


### Structure:
```
XsCT/:
   |-data/: the folder include preprocessed train/valid/test data
   |    |-mesh_data/: the .h5 file preprocessed data
   |    |train.txt: train data list
   |    |valid.txt: valid data list
   |    |test.txt: test data list
   |
   |-view_mode/:
   |    |multiView.yml: multiview .yml file
   |    |singleView.yml: singleview .yml file
   |
   |-libs/:
   |    |-config/: includes the config file
   |    |-dataset/: includes the source code to process the dataset
   |    |-model/: includes the network definitions
   |    |-utils/: utility functions
   |
   |-save_models/:
   |    |-multiView_XsCT/: biplanar X-ray to CT model
   |    |-singleView_XsCT/: single view X-ray to CT model
   |
   |test.py: test script that demonstrates the inference workflow and outputs the metric results
   |train.py: train script
   |visual.py: same as test.py but viualize the output without calculating the statistics
   |
procs/:
   |ct_metadata_read_preproc.py: CT .mhd data reading, resampling, cropping and wrapping to .h5 file
   |make_h5_file_for_final_inference.py: make empty .h5 file for inference final result
   |read_mha.py: read CT .mha inference result
   |readXA.m: read and process C-arm DICOM XA image (.m)
   |resize_drr.m: resize calculated drr image (.m)
   |itk_calculate_drr.cpp: calculate drr image (.cpp)
   |
requirements.txt: python dependency libraries
README.md
```

### Tips:
1. **prepare dataset:**  
    .h5 file contains: CT data and drr image: prepare CT data, calculate drr image and wrap together (modules under ./procs folder)
2. **train model** (train.py script)  
3. **model inference** (test.py/ visual.py script)  
4. **generated .mha sCT** (can be viewed with ITK-SNAP program (./procs/read_mha.py script)  

### Train:
the default -args is set to multi-view mode under train script, since the model trained with multi-view X-ray image produce better sCT generate capability and performance
1. **single-view：**  
```
python train.py --ymlpath=./view_mode/singleview.yml --dataroot=./data/mesh_data --dataset=train --tag=singleview --data=mesh_data --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/train.txt --valid_datasetfile=./data/test.txt --valid_dataset=test
```
2. **multi-view：**  
```
python train.py --ymlpath=./view_mode/multiview.yml --dataroot=./data/mesh_data --dataset=train --tag=multiview --data=mesh_data --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/train.txt --valid_datasetfile=./data/test.txt --valid_dataset=test
```

### Test/ Visual:
test.py script make forward calculation of model, generate sCT and print out statics of image validation index  
visual.py script make forward calculation of model, generate sCT and save .png images for visualization
1. **single-view：**  
```
python test.py --ymlpath=./view_mode/singleview.yml --dataroot=./data/mesh_data --dataset=test --tag=singleview --data=mesh_data --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/test.txt --resultdir=./result/singleview --check_point=100  
```
```
python visual.py --ymlpath=./view_mode/singleview.yml --dataroot=./data/mesh_data --dataset=test --tag=singleview --data=mesh_data --dataset_class=align_ct_xray_std --model_class=SingleViewCTGAN --datasetfile=./data/test.txt --resultdir=./result/singleview --check_point=100 --how_many=50   
```
2. **multi-view：**  
```
python test.py --ymlpath=./view_mode/multiview.yml --dataroot=./data/mesh_data --dataset=test --tag=multiview --data=mesh_data --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/test.txt --resultdir=./result/multiview --check_point=100
```
```
python visual.py --ymlpath=./view_mode/multiview.yml --dataroot=./data/mesh_data --dataset=test --tag=multiview --data=mesh_data --dataset_class=align_ct_xray_views_std --model_class=MultiViewCTGAN --datasetfile=./data/test.txt --resultdir=./result/multiview --check_point=100 --how_many=50
```