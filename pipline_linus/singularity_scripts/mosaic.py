"""  
A collection of tools to tile images for the use with a fully convolutional network.
Linus Meienberg
June 2020
"""

# NOTE this is a COPY, please work on the original file and superseed this one


import numpy as np

class UnetTiler3D():

    def __init__(self, image=None, image_shape=None, mask=None, output_shape=(132,132,132), input_shape=(220,220,220)):
        """
        Uses a pretrained 3D unet to segment an arbitrary input volume.

        A rectangular tiling of the image volume with the unet output is spatially extended to get a collection of input regions. 
        The class reassembles the unet output to a segmentation mask with the same shape as the input image.

        Parameters
        ----------
        image : image tensor
            the input tensor
        output_shape : tuple
            shape of the segmentation output of the unet
        input_shape : tuple
            shape of the image input of the unet
        """
        assert not image is None or not image_shape is None,  'specify at least an image shape for tiling'
        self.image = image
        self.image_shape = image_shape

        # Automatically infer image shape if not specified
        if image_shape is None:
            self.image_shape = image.shape

        # Allocate an array to assemble the mask in if not specified
        if mask is None:
            self.mask = np.zeros_like(image) # Allocate a tensor where the segmentation mask is stored
        else:
            assert self.image_shape == mask.shape, 'The mask and image array need to be of the same shape'
            self.mask = mask

        # Store output and input shape of the unet and check for correct number of dimensions
        self.output_shape = output_shape
        self.input_shape = input_shape
        assert len(self.image_shape) == 3, 'Specify a single channel 3D image with format (x,y,z)'
        assert len(self.output_shape) == 3, 'Specify the extent of the output shape as (x,y,z)'
        assert len(self.input_shape) == 3, 'Specify the extent of the input shape as (x,y,z)'
        
        # Calculate the coordinate mesh of the tiling
        # Each list goes up to the last multiple of tile_shape smaller than image_shape => endpoint excluded
        self.x = list(range(0,self.image_shape[0],self.output_shape[0]))
        self.y = list(range(0,self.image_shape[1],self.output_shape[1]))
        self.z = list(range(0,self.image_shape[2],self.output_shape[2]))

        # Expose the shape of the tiling
        self.shape = (len(self.x),len(self.y),len(self.z))
    
    def __len__(self):
        return np.prod(self.shape)

    def _indexToCoordinates(self, i):
        """Convert a tile index to tiling coordinates

        Parameters
        ----------
        i : int
            tile index

        Returns
        -------
        x,y,z : int
            the coordinates of the tile in the tiling grid
        """     
         # Sanity check
        assert i >=0, 'index out of bounds'
        assert i < len(self), 'index out of bounds'
        # Convert index to the coordinates of the tile
        x = i // (self.shape[1]*self.shape[2]) # number of elements that you skip by moving one position in dim 0
        i = i % (self.shape[1]*self.shape[2])
        y = i // self.shape[2]
        z = i % self.shape[2]
        return x,y,z

    def _coordinatesToIndex(self,x,y,z):
        """Converts the coordinates of a tile in the tiling grid to it's index
        """
        assert (x,y,z) < self.shape and (x,y,z) >= (0,0,0), 'Coordinates out of bounds'
        i = x*self.shape[1]*self.shape[2]
        i += y*self.shape[2]
        i += z
        return i

    def _getOutputTile(self, i):
        """Returns an axis aligned boundary box defining the ith tile. 
        The input image is tiled with the output of the unet to get a segmentation mask of the same size.

        Parameters
        ----------
        i : int
            index of the output tile

        Returns
        -------
        tuple
            aabb coordinate tuple (x0,y0,z0,x1,y1,z1) (diagonal oposite corners that define a rectangular volume)
        """
        x,y,z = self._indexToCoordinates(i)
        # assemble the coordinates of the target chunk
        x0 = self.x[x]
        y0 = self.y[y]
        z0 = self.z[z]
        x1 = x0 + self.output_shape[0]
        y1 = y0 + self.output_shape[1]
        z1 = z0 + self.output_shape[2]
        return (x0,y0,z0,x1,y1,z1)

    def _getInputTile(self, i):
        aabb = self._getOutputTile(i) # get the aabb of the corresponding input tile 
        delta = np.subtract(self.input_shape,self.output_shape) // 2 # symmetric expansion in each direction
        # we have to subtract delta from the inital coords and add it to the stop coords
        delta = np.concatenate((-delta, delta))
        aabb = np.add(aabb,delta) # element wise addition
        return tuple(aabb)

    def _cropAndPadAABB(self, volume ,aabb):
        assert len(volume.shape)==3, 'Need a 3D volume'
        # clip the aabb if it protrudes from the image volume
        start = [ np.max([0, d]) for d in aabb[:3] ] # origo is at (0,0,0)
        stop = [ np.min([volume.shape[i], aabb[i+3]]) for i in range(3) ]
        # calculate the padding in each direction
        pre_pad = [ np.max([0, -d]) for d in aabb[:3] ]
        post_pad = [ np.max([0, aabb[i+3] - volume.shape[i] ]) for i in range(3) ]
        padding = tuple([ (pre_pad[i], post_pad[i]) for i in range(3) ] )
        # extract valid/ clipped portion of the aabb
        data = volume[start[0]:stop[0],start[1]:stop[1],start[2]:stop[2]]
        # pad the slice to the required size
        data = np.pad(data, pad_width=padding, mode='reflect')
        return data

    def _getSlice(self, index):
        assert not self.image is None, 'No image data specified'
        # get the aabb of the unet input slice
        aabb = self._getInputTile(index)
        data = self._cropAndPadAABB(self.image, aabb)
        return data

    def _getMaskSlice(self, index):
        # get the aabb of the unet OUTPUT slice
        aabb = self._getOutputTile(index)
        data = self._cropAndPadAABB(self.mask, aabb)
        return data
    
    def _writeSlice(self, index, slice):
        assert slice.shape == self.output_shape, 'Slice needs to have the output shape of the unet'
        assert index>=0 and index < len(self), 'Index out of bounds'
        aabb = self._getOutputTile(index) # retrieve aabb that belongs to the slice index
        #print('unet output mask target aabb : {}'.format(aabb))

        # clip the target aabb if it protrudes from the image volume
        start = [ np.max([0, d]) for d in aabb[:3] ] # origo is at (0,0,0)
        stop = [ np.min([self.image.shape[i], aabb[i+3]]) for i in range(3) ]
        #print('unet output mask target {} to {} :'.format(start,stop))

        # calculate the padding which was applied to the source 
        slice_start = [ np.max([0, -d]) for d in aabb[:3] ]
        slice_stop = [ np.max([0, aabb[i+3] - self.image.shape[i] ]) for i in range(3) ]
        #print('unet output mask crop from {} to {}'.format(slice_start,slice_stop))

        # crop the padding away
        slice_cropped = slice[slice_start[0]:-slice_stop[0] or None,
                              slice_start[1]:-slice_stop[1] or None,
                              slice_start[2]:-slice_stop[2] or None]
        
        # write the cropped slice to the target position in the mask
        self.mask[start[0]:stop[0],start[1]:stop[1],start[2]:stop[2]] = slice_cropped

    def _process_slice(self, unet, i):
        # Read input slice from volume
        input_slice = self._getSlice(i)
        #print('input shape : '.format(input_slice.shape))
        # Add batch and channel dimension and feed to unet
        output_slice = unet.predict(input_slice[np.newaxis,:,:,:,np.newaxis])
        output_mask = np.argmax(output_slice, axis=-1)[0,...] # use argmax on channels and remove batch dimension
        #print('unet output mask shape : '.format(input_slice.shape))
        self._writeSlice(i, output_mask)
            
