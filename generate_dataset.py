import os
import numpy as np
import SimpleITK as sitk
import torch

def load_scans(dcm_path):
    reader = sitk.ImageSeriesReader()
    name = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(name)
    sitk_img = reader.Execute()
    return sitk_img

def __normalization__(vol_np, win_level, win_width):
    win = [win_level - win_width / 2, win_level + win_width / 2]
    vol = vol_np.astype('float32')
    vol = np.clip(vol, win[0], win[1])
    vol -= win[0]
    vol /= win_width
    return vol

def get_boundaries_from_mask(mask):
    
	mask = torch.from_numpy(mask)
	mask_voxel_coords = torch.nonzero(mask)
	zmin = int(torch.min(mask_voxel_coords[:, 0]))
	zmax = int(torch.max(mask_voxel_coords[:, 0])) + 1
	ymin = int(torch.min(mask_voxel_coords[:, 1]))
	ymax = int(torch.max(mask_voxel_coords[:, 1])) + 1
	xmin = int(torch.min(mask_voxel_coords[:, 2]))
	xmax = int(torch.max(mask_voxel_coords[:, 2])) + 1
	out_bbox = {'zmin': zmin,
				'zmax': zmax,
				'ymin': ymin,
				'ymax': ymax,
				'xmin': xmin,
				'xmax': xmax}
	return out_bbox

dcm_path = r"/media/tx-deepocean/Data/data/pulmonary_vessels/train/dcm"
seg_path = r"/media/tx-deepocean/Data/data/pulmonary_vessels/train/seg"
save_path = r"/media/tx-deepocean/Data/code/TransUNet/data/Synapse/train_npz_3"

for sub in os.listdir(dcm_path):
    dcm_itk = load_scans(os.path.join(dcm_path, sub))
    seg_itk = sitk.ReadImage(os.path.join(seg_path, sub + '-seg.nii.gz'))
    dcm_arr = sitk.GetArrayFromImage(dcm_itk)
    dcm_arr = __normalization__(dcm_arr, -600, 1500)
    seg_arr = sitk.GetArrayFromImage(seg_itk)
    bbox = get_boundaries_from_mask(seg_arr)
    zmin = max(1, bbox['zmin'] - 15)
    zmax = min(dcm_arr.shape[0]-1, bbox['zmax'] + 15)
    for slice in range(zmin, zmax):
        np.savez_compressed(os.path.join(save_path, sub + "_" + str(slice) + '.npz'), 
                            image = dcm_arr[slice-1:slice+2],
                            label = seg_arr[slice])
    
    