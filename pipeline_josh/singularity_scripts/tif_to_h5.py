import sys, getopt
import os
import re
import glob
import h5py
import skimage.io
from tqdm import tqdm
import numpy as np 


def tif_slices_to_h5_volume(input_dir, output_dir):
    '''
    Write tif slices into hdf5 file, with chunk size 100x100x100
    sample data:
    /nrs/dickson/lillvis/temp/ExM/pMP8/20180917/slice-tiff-s0/ch0 ***
    /nrs/dickson/lillvis/temp/ExM/opticlobe/mi1/20180622/images/stitch/slice-tiff-s0/ch1 *
    /nrs/dickson/lillvis/temp/ExM/opticlobe/L2/20180504/images/stitch/slice-tiff-s0/ch0 **
    Args:
    input_dir: input directory
    output_dir: output directory
    '''

    assert os.path.exists(input_dir), \
        "ERROR: Input directory does not exist!"
    slice_paths = glob.glob(input_dir+'/*.tif')
    assert len(slice_paths), \
        "ERROR: tif images do not exist!"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    regex = re.compile(r'\d+')
    slice_paths = sorted(slice_paths, key=lambda f: int(regex.findall(os.path.basename(f))[0]))
    first_slice = skimage.io.imread(slice_paths[0])
    volume_shape = (len(slice_paths), *first_slice.shape)

    with h5py.File(output_dir+'/slices_to_volume.h5', 'w') as f:
        dset = f.create_dataset('volume', shape=volume_shape, chunks=(100,100,100))
        for z, tif_file in enumerate(tqdm(slice_paths)):
            tif_img = skimage.io.imread(tif_file)
            dset[z,:,:] = tif_img

    print("DONE!")
    return None


def main(argv):

    input_dir = None
    output_dir = None

    try:
        options, remainder = getopt.getopt(argv, "i:o:", ["input_dir=","output_dir="])
    except getopt.GetoptError:
        print("ERROR!") 
        print("Usage: tif_to_h5.py -i <input_directory> -o <output_directory>")
        sys.exit(1)
    
    for opt, arg in options:
        if opt in ('-i', '--input_dir'):
            input_dir = arg
        elif opt in ('-o', '--output_dir'):
            output_dir = arg
    
    if output_dir is None:
        output_dir = input_dir

    tif_slices_to_h5_volume(input_dir, output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])