"""This module defines tools to apply a Unet architecture in the context of the flood filling procedure

Linus Meienberg
August 2020
"""
#%%
import tensorflow as tf

import sys, random
sys.path.append('../tools')
import tilingStrategy, Dataset3D

import numpy as np
#%%
class SelfAvoidingQueue():
    def __init__(self):
        super().__init__()
        self.queue = [] # store all positions that should be visited in a queue
        self.visited = set() # store all visited positions in a set

    def hasNext(self):
        return len(self.queue)>0

    def putTile(self, i):
        """Add a tile index to the self avoiding queue. 
        This call is ignored of the tile has been visited before or is allready present in the queue.

        Parameters
        ----------
        i : int
            tile index
        """
        if not i in self.queue:
            if not i in self.visited:
                self.queue.append(i)

    def putTiles(self, indices):
        for i in indices:
            self.putTile(i)
    
    def getTile(self):
        """Return the index of the next tile in the queue.
        Marks this tile as visited.

        Returns
        -------
        int
            tile index
        """
        try:
            i = self.queue.pop()
            self.visited.add(i)
            return i
        except IndexError as ie:
            return None
        
#%%
class FloodFiller():
    def __init__(self, image, segmentation, mask, output_shape=(132,132,132), input_shape=(220,220,220), delta=(20,20,20), containTiling=False):
        super().__init__()
        # Tile the image using overlapping unet tiles
        self.tiling = tilingStrategy.OverlappingUnetTiling3D(image.shape, output_shape, input_shape, delta, containTiling)
        # maintain an image, segmentation and (if available) mask canvas
        self.image = tilingStrategy.Canvas(image) # Use a canvas for i/o on the image

        # Allocate an array to assemble the mask in if not specified
        if segmentation is None:
            self.segmentation = tilingStrategy.Canvas(np.zeros_like(image)) # Allocate a tensor where the segmentation mask is stored
        else:
            assert self.image.shape == segmentation.shape, 'The segmentation canvas and image array need to be of the same shape'
            self.segmentation = tilingStrategy.Canvas(segmentation)

        if not mask is None:
            assert self.image.shape == mask.shape, 'The mask needs to have the same shape as the image'
            self.mask = tilingStrategy.Canvas(mask)
        else:
            self.mask = None

        # Expose the shape of the tiling
        self.shape = self.tiling.shape

        # Use a self avoiding queue to keep track of the segmentation state
        self.queue = SelfAvoidingQueue()

    # Expose the number of tiles 
    def __len__(self):
        return np.prod(self.shape)

    def hasNext(self):
        return self.queue.hasNext()

    # Get the next tile from the queue
    def getNextIndex(self):
        return self.queue.getTile()
    
    # Get the next annotated tile
    def getNext(self) -> tuple or None:
        """Returns a tuple containig the image, segmentation canvas and mask of the next tile in the queue.

        Returns
        -------
        tuple or None
            (index, image, segmentation, mask) image and segmentation are input regions, mask is output region only
        """
        if self.hasNext():
            index = self.getNextIndex()
            image = self.getSlice(index, source='image', region='input')
            segmentation = self.getSlice(index, source='segmentation', region='input')
            if self.mask is None:
                mask = None
            else:
                mask = self.getSlice(index, source='mask', region='output')
            return (index, image, segmentation, mask)
        else:
            return None

    def storePredictionUpdateQueue(self, index, tile):
        # Write the predicted mask tile to the canvas
        self.writeSlice(index,tile)
        # Use an evaluation heuristic to determine which neighbours of the tile should be processed as well.
        self.queueHeuristic(self, index, tile)

    def queueHeuristic(self, index, tile: tf.Tensor, threshold = 0.9):
        """
        To search for new positions where the FCCN should be evaluated the Flood Filling Paper states:
            "potential new positions are searched by examining the current state
            of the mask at the 6 planes x = x0 ± ∆x, y = y0 ± ∆y and z = z0 ± ∆z further restricted to
            [x0 − ∆x ≤ x ≤ x0 + ∆x] × [y0 − ∆y ≤ y ≤ y0 + ∆y] × [z0 − ∆z ≤ z ≤ z0 + ∆z]. For every
            such plane the object mask voxel with the highest value v is found, and if v ≥ tmove the location of
            that voxel is added to a list of new positions for the FFN."
        
        """
        cp = [d//2 for d in tile.shape] # calculate the (relative) center point of the tile
        # construct the domain of the evaluation cube (d_pre, d_post) for x,y,z
        x = (cp[0]-self.tiling.delta[0],cp[0]+self.tiling.delta[0],)
        y = (cp[1]-self.tiling.delta[1],cp[1]+self.tiling.delta[1],)
        z = (cp[2]-self.tiling.delta[2],cp[2]+self.tiling.delta[2],)
        #print(cp)
        #print((x,y,z))

        # construct the slice arguments to obtain the evaluation plains
        planes = [
            tile[x[0],y[0]:y[1],z[0]:z[1]],
            tile[x[1]-1,y[0]:y[1],z[0]:z[1]],

            tile[x[0]:x[1],y[0],z[0]:z[1]],
            tile[x[0]:x[1],y[1]-1,z[0]:z[1]],

            tile[x[0]:x[1],y[0]:y[1],z[0]],
            tile[x[0]:x[1],y[0]:y[1],z[1]-1],
        ]

        # Calculate mean max object probability for each plane
        maxima = [ np.max(plane) for plane in planes]
        # Threshold means to get logical array of positions that should be enqueued
        steps = [ max > threshold for max in maxima]
        # get adjacent tile indices in (pre,post) format for x,y,z
        neighbours = self.tiling.getAdjacentTiles(index)

        # Add all neigbouring tiles if they exist and exceed the threshold
        for i, index in enumerate(neighbours):
            if not index is None:
                if steps[i]:
                    self.queue.putTile(index)

    def seedSlice(self, index:int):
        """Use the ground truth mask to seed a single pixel in a specified tile of the segmentation canvas.

        Parameters
        ----------
        index : int
            the index of the tile that should be seeded
        """
        assert not self.mask is None, 'Cannot seed without ground truth data'
        mask_tile = self.getSlice(index, source='mask', region='output')

        assert np.any(mask_tile), 'The specified tile does not contain the object - cannot seed here'
        segmentation_seed = np.zeros_like(mask_tile) # Allocate segmentation slice

        while True:
            # draw random coordinates in the mask_tile
            coords = np.s_[tuple([random.randrange(0,axis) for axis in mask_tile.shape])]
            if mask_tile[coords] > 0: # test if the object is present there
                segmentation_seed[coords] = 0.95 # seed segmentation canvas and break loop
                break

        self.writeSlice(index, segmentation_seed) # write the segmentation seed to the canvas


    def getSlice(self, index: int, source='image', region='input'):
        """Extract input or output slices from a source

        Parameters
        ----------
        index : int
            the tile index
        source : str, optional
            identifier of the canvas that should be accessed, either 'image', 'segmentation' or 'mask'
        region : str, optional
            wheter the input or output region of the tile should be extracted, either 'input' or 'output'

        Returns
        -------
        arraylike
            the specified slice
        """
        if region is 'output':
            aabb = self.tiling.getOutputTile(index)
        elif region is 'input':
            # get the aabb of the unet input slice
            aabb = self.tiling.getInputTile(index)
        else:
            raise ValueError('Unknown Region {}'.format(region))

        if source is 'image':
            # read out the aabb from the image data
            data = self.image.cropAndPadAABB(aabb)
        elif source is 'segmentation':
            data = self.segmentation.cropAndPadAABB(aabb)
        elif source is 'mask':
            assert not self.mask is None, 'There is no mask to read from'
            # read out the aabb from mask data
            data = self.mask.cropAndPadAABB(aabb)
        else:
            raise ValueError('Unknown Source {}'.format(source))

        return data

    def writeSlice(self, index, tile):
        """Writes the i-th tile of the predicted mask to the segmentation canvas.

        Parameters
        ----------
        index : int
            the index of the mask tile
        tile : 3d tensor
            the i-th output tile of the mask
        """
        assert tile.shape == self.tiling.output_shape, 'Slice needs to have the output shape of the unet'
        assert index>=0 and index < len(self), 'Index out of bounds'
        aabb = self.tiling.getOutputTile(index) # retrieve aabb that belongs to the slice index
        self.segmentation.writeAABB(aabb, tile)
# %%

# Tests
#ff = FloodFiller(np.zeros((100,100,100)),segmentation=None, mask=None, output_shape=(10,10,10), input_shape=(20,20,20), delta=(5,5,5))


# %%
"""
ff.queueHeuristic(0,np.ones((10,10,10)))
ff.queueHeuristic(0,np.concatenate((np.zeros((5,10,10)),np.ones((5,10,10))))) # only x pre should be false
"""

def constructExample(image: tf.Tensor, mask: tf.Tensor, output_shape, input_shape, delta) -> FloodFiller:
    """Constructs a training scene, given a image mask pair.

    Parameters
    ----------
    image : tf.tensor
    mask : tf.tensor

    Returns
    -------
    FloodFiller
        A flood filler instance defining a centered tiling on the image volume. The central tile is enqued for evaluation and a single seed pixel is set in the segmentation canvas.
    """
    ff = FloodFiller(image=image,
                     segmentation= None, # Allocate a new segmentation canvas
                     mask= mask,
                     output_shape=output_shape,
                     input_shape=input_shape,
                     delta=delta,
                     containTiling=True)
    
    # Verify that the scene alows the unet to move at least one step in each direction
    assert np.all([axis > 2 for axis in ff.shape]), 'A training scene should allow at least one step in each direction'

    # Locate the central tile
    center_index = len(ff)//2 # for a 3x3x3 tiling with len=27 -> 27//2 = 13 is the central tile
    try:
        # seed a slice
        ff.seedSlice(center_index)
        # enque 
        ff.queue.putTile(center_index)
    except AssertionError as ae:
        # affine transformations and elastic deformations can lead to situations where there is no more object left in the central tile
        # do not crash here, but also leave the queue of the flood filler empty
        print('failed to construct scene, no object to seed central tile')
   

    return ff

# %%
#ff = constructExample(np.zeros((100,100,100)), np.ones((100,100,100)), output_shape=(10,10,10), input_shape=(20,20,20), delta=(5,5,5))
# %%
