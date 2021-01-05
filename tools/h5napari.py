"""View a hdf5 file with napari
"""
#%% Imports
import numpy
import h5py
from dask import delayed
import dask.array as da
import napari 

#%% Open the hdf5 file
filename = 'D:/Janelia/UnetTraining/20201201_MWS/Q1_mws.h5'
infile = h5py.File(filename)
print(list(infile['t0'].keys()))
# %%
data = infile['t0/test1']
# %%
data.shape
# %%
ddata = da.from_array(data)
# %% Start napari
with napari.gui_qt():
    napari.view_image(ddata, contrast_limits=[0,1], multiscale=False)
# %%
