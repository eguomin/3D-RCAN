# Copyright 2020 DRVision Technologies LLC.
# SPDX-License-Identifier: CC-BY-NC-4.0


from rcan.utils import apply, get_model_path, normalize, load_model
import argparse
from tensorflow import keras
import numpy as np
import tifffile
import os
from scipy.ndimage import zoom
from scipy.ndimage import rotate

#parser = argparse.ArgumentParser()
#parser.add_argument('-m', '--model_dir', type=str, required=True)
#parser.add_argument('-i', '--input', type=str, required=True)
#parser.add_argument('-o', '--output', type=str, required=True)
#parser.add_argument('-g', '--ground_truth', type=str)
#parser.add_argument('-b', '--bpp', type=int, choices=[8, 16, 32], default=32)
#args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
block_shape = (16,128,128)
#block_overlap_shape = (32,64,64) #None
block_overlap_shape = None

model_dir = 'Y:\\Yicong\\LineConfocal\\Ivans_cell_DL_1\\H2B-GFP\\Input_RCAN_DL_Decon_ZY_GaussainBlurTraining\\Model_64x64x64'
#model_dir =  'Y:\\Yicong\LineConfocal\\Ivans_cell_DL_1\\H2B-GFP\Input_RCAN_DL_Decon_ZY_GaussainBlurTraining\\test4P100'
predition_dir = 'Y:\\Yicong\\LineConfocal\\Ivans_cell_DL_1\\H2B-GFP_EMTB_mCherry_TimeLapse\\Cell12\\488_2x_RCAN_DL_Decon_ZY_GaussainBlurTraining'
test_folder = 'Test'
output_folder = test_folder + '_RCAN_DL_MGPU'
try:
    DL_path = predition_dir + '\\' + output_folder
    if not os.path.exists(DL_path):
        os.makedirs(DL_path)
except OSError:
    print ("Creation of the directory %s failed" % DL_path)
else:
    print ("Successfully created the directory %s " % DL_path)

input_labels=os.listdir(predition_dir + '\\' + test_folder)
maxlen = len(input_labels)
angle = np.linspace(-90,90, 7)

model_path = get_model_path(model_dir)
model = load_model(str(model_path))

if block_overlap_shape is None:
    overlap_shape = [
        max(1, x // 8) if x > 2 else 0
        for x in model.input.shape.as_list()[1:-1]]
else:
    overlap_shape = block_overlap_shape

for i in range(0,maxlen):
    Predition_File = predition_dir + '\\' + test_folder + '\\'+ input_labels[i]
    print('Loading raw image from', Predition_File)
    input_data = normalize(tifffile.imread(Predition_File))
    print('Applying model')

    ndim = input_data.ndim
    if ndim == 3:
        (nz, ny, nx) = input_data.shape
        nx1 = int(np.sqrt(nx*nx + ny*ny))
        input_data_pad = np.zeros((nz,nx1,nx1))
        print(input_data_pad.shape)
        input_data_pad[:,round(nx1/2)-round(ny/2):round(nx1/2)-round(ny/2)+ny,round(nx1/2)-round(nx/2):round(nx1/2)-round(nx/2)+nx] = input_data
    else:
        (ny, nx) = input_data.shape
        nx1 = int(np.sqrt(nx*nx + ny*ny))
        input_data_pad = np.zeros((nx1,nx1))
        print(input_data_pad.shape)
        input_data_pad[round(nx1/2)-round(ny/2):round(nx1/2)-round(ny/2)+ny,round(nx1/2)-round(nx/2):round(nx1/2)-round(nx/2)+nx] = input_data       
    
    for k in range(len(angle)-1):
        if ndim == 3:
            input_data1 = rotate(input_data_pad, -1*angle[k], axes=(1,2), reshape=False)
            result = apply(model, input_data1, overlap_shape=overlap_shape, verbose=True)
            denoised_result = rotate(result, angle[k], axes=(1,2), reshape=False)
            final_result = denoised_result[:,round(nx1/2)-round(ny/2):round(nx1/2)-round(ny/2)+ny,round(nx1/2)-round(nx/2):round(nx1/2)-round(nx/2)+nx]
        else:
            input_data1 = rotate(input_data_pad, -1*angle[k], reshape=False)
            result = apply(model, input_data1, overlap_shape = overlap_shape, verbose=True)
            denoised_result = rotate(result, angle[k], reshape=False)
            #denoised_result = result
            final_result = denoised_result[round(nx1/2)-round(ny/2):round(nx1/2)-round(ny/2)+ny,round(nx1/2)-round(nx/2):round(nx1/2)-round(nx/2)+nx]

        inmin, inmax = final_result .min(), final_result .max()
        final_result = (final_result - inmin) / (inmax - inmin) * 65535
        tifffile.imsave(predition_dir + '\\' + output_folder + '\\DL_' + str(int(angle[k])) + '_' + input_labels[i], final_result.astype('uint16'))

   
        print('Saving output image', input_labels[i])
        #    result.append(normalize(tifffile.imread(args.ground_truth)))

        #result = np.stack(result)
        #if result.ndim == 4:
        #    result = np.transpose(result, (1, 0, 2, 3))


