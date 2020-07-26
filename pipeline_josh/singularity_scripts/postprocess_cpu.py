import sys, getopt
import numpy as np 
import h5py
from random import randint
import skimage.io
import matlab.engine
from skimage.measure import label, regionprops
import time
import csv
import os


def tif_read(file_name):
    """
    read tif image in (rows,cols,slices) shape
    """
    im = skimage.io.imread(file_name)
    im_array = np.zeros((im.shape[1],im.shape[2],im.shape[0]), dtype=im.dtype)
    for i in range(im.shape[0]):
        im_array[:,:,i] = im[i]
    return im_array


def tif_write(im_array, file_name):
    """
    write an array with (rows,cols,slices) shape into a tif image
    """
    im = np.zeros((im_array.shape[2],im_array.shape[0],im_array.shape[1]), dtype=im_array.dtype)
    for i in range(im_array.shape[2]):
        im[i] = im_array[:,:,i]
    skimage.io.imsave(file_name,im)
    return None


def hdf5_read(file_name, location):
    """
    read part of the hdf5 image in (rows,cols,slices) shape
    Args:
    file_name: hdf5 file name
    location: a tuple of (min_row, min_col, min_vol, max_row, max_col, max_vol) indicating what area to read
    """
    read_img = True
    while read_img:
        try:
            with h5py.File(file_name, 'r') as f:
                im = f['volume'][location[2]:location[5], location[0]:location[3], location[1]:location[4]]
            read_img = False
        except OSError:  # If other process is accessing the image, wait 5 seconds to try again
            time.sleep(randint(1,5))
    im_array = np.zeros((im.shape[1],im.shape[2],im.shape[0]), dtype=im.dtype)
    for i in range(im.shape[0]):
        im_array[:,:,i] = im[i]
    return im_array


def hdf5_write(im_array, file_name, location):
    """
    write an image array into part of the hdf5 image file
    Args:
    im_array: an image array
    file_name: an existing hdf5 file to partly write in
    location: a tuple of (min_row, min_col, min_vol, max_row, max_col, max_vol) indicating what area to write
    """
    assert os.path.exists(file_name), \
        print("ERROR: hdf5 file does not exist!")        
    im = np.zeros((im_array.shape[2],im_array.shape[0],im_array.shape[1]), dtype=im_array.dtype)
    for i in range(im_array.shape[2]):
        im[i] = im_array[:,:,i]
    write_img = True
    while write_img:
        try:
            with h5py.File(file_name, 'r+') as f:
                f['volume'][location[2]:location[5], location[0]:location[3], location[1]:location[4]] = im
            write_img = False
        except OSError: # If other process is accessing the image, wait 5 seconds to try again
            time.sleep(randint(1,5))
    return None


def remove_small_piece(out_hdf5_file, img_file_name, location, mask=None, threshold=10, percentage=1.0):
    """
    remove blobs that have less than N voxels
    write final result to output hdf5 file, output a .csv file indicating the location and size of each synapses
    Args:
    out_hdf5_file: output hdf5 file
    img_file_name: tif image file for processing
    mask: mask image
    location: a tuple of (min_row, min_col, min_vol, max_row, max_col, max_vol) indicating img location on the hdf5 file
    threshold: threshold to remove small blobs (default=10)
    percentage: threshold to remove the object if it falls in the mask less than a percentage. If percentage is 1, criteria will be whether the centroid falls within the mask
    """

    print("Removing small blobs and save results to disk...")
    img = tif_read(img_file_name)
    img[img!=0] = 1
    label_img = label(img, neighbors=8)
    regionprop_img = regionprops(label_img)
    idx = 0

    for props in regionprop_img:
        num_voxel = props.area
        curr_obj = np.zeros(img.shape, dtype=img.dtype)
        curr_obj[label_img==props.label] = 1
        center_row, center_col, center_vol = props.centroid

        if mask is not None:
            assert mask.shape == img.shape, \
                "Mask and image shapes do not match!"
        else:
            mask = np.ones(img.shape, dtype=img.dtype)
        curr_obj = curr_obj * mask

        exclude = False
        if num_voxel < threshold:
            exclude = True
        if percentage < 1:
            if np.count_nonzero(curr_obj) < num_voxel*percentage:
                exclude = True
        else:
            if mask[int(center_row), int(center_col), int(center_vol)] == 0:
                exclude = True

        if exclude:  
            img[label_img==props.label] = 0
        else:
            if idx == 0:
                out_path = os.path.dirname(out_hdf5_file)
                csv_name = 'stats_r'+str(location[0])+'_'+str(location[3]-1)+'_c'+str(location[1])+'_'+str(location[4]-1)+'_v'+str(location[2])+'_'+str(location[5]-1)+'.csv'
                with open(out_path+'/'+csv_name, 'w') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([' ID ', ' Num vxl ', ' centroid ', ' bbox row ', ' bbox col ', ' bbox vol '])
            idx += 1
            min_row, min_col, min_vol, max_row, max_col, max_vol = props.bbox
            bbox_row = (int(min_row+location[0]), int(max_row+location[0]))
            bbox_col = (int(min_col+location[1]), int(max_col+location[1]))
            bbox_vol = (int(min_vol+location[2]), int(max_vol+location[2]))
            
            center = (int(center_row+location[0]), int(center_col+location[1]), int(center_vol+location[2]))
            
            csv_row = [str(idx), str(num_voxel), str(center), str(bbox_row), str(bbox_col), str(bbox_vol)]
            with open(out_path+'/'+csv_name, 'a') as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(csv_row)    
        
    img[img!=0] = 255
    hdf5_write(img, out_hdf5_file, location)
    return None


def main(argv):
    """
    Main function
    """
    hdf5_file = None
    location = []
    mask_file = None
    threshold = 10
    percentage = 1.0
    try:
        options, remainder = getopt.getopt(argv, "i:l:m:t:p:", ["input_file=","location=","mask_file=","threshold=","percentage="])
    except:
        print("ERROR:", sys.exc_info()[0]) 
        print("Usage: postprocess_cpu.py -i <input_hdf5_file> -l <location> -m <mask_file> -t <threshold> -p <percentage>")
        sys.exit(1)
    
    # Get input arguments
    for opt, arg in options:
        if opt in ('-i', '--input_file'):
            hdf5_file = arg
        elif opt in ('-l', '--location'):
            location.append(arg.split(","))
            location = tuple(map(int, location[0]))
        elif opt in ('-m', '--mask_file'):
            mask_file = arg
        elif opt in ('-t', '--threshold'):
            threshold = int(arg)
        elif opt in ('-p', '--percentage'):
            percentage = float(arg)
    
    # Read part of the hdf5 image file based upon location
    if len(location):
        img = hdf5_read(hdf5_file, location)
        img_path = os.path.dirname(hdf5_file)
    else:
        print("ERROR: location need to be provided!")
        sys.exit(1)
    
    # Read part of the hdf5 mask image based upon location
    if mask_file is not None:
        mask = hdf5_read(mask_file, location)
        if np.count_nonzero(mask) == 0:  # if the mask has all 0s, write out the result directly
            hdf5_write(mask, hdf5_file, location)
            print("DONE! Location of the mask has all 0s.")
            sys.exit(0)
    else:
        mask = None

    start = time.time()
    print('#############################')
    out_img_name = img_path+'/r'+str(location[0])+'_'+str(location[3])+'_c'+str(location[1])+'_'+str(location[4])+'_v'+str(location[2])+'_'+str(location[5])+'.tif'
    tif_write(img, out_img_name)
    eng = matlab.engine.start_matlab()
    flag = eng.closing_watershed(out_img_name)
    eng.quit()
    remove_small_piece(out_hdf5_file=hdf5_file, img_file_name=out_img_name, location=location, mask=mask, threshold=threshold, percentage=percentage)
    if os.path.exists(out_img_name):
        os.remove(out_img_name)
    end = time.time()
    print("DONE! Running time is {} seconds".format(end-start))
    

if __name__ == "__main__":
    main(sys.argv[1:])