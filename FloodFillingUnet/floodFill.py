"""This module defines tools to apply a Unet architecture in the context of the flood filling procedure

Linus Meienberg
August 2020
"""
#%%
import sys
sys.path.append('../tools')
import tilingStrategy
import numpy as np
#%%
class SelfAvoidingQueue():
    def __init__(self):
        super().__init__()
        self.queue = [] # store all positions that should be visited in a queue
        self.visited = set() # store all visited positions in a set

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
    def __init__(self, image, mask, output_shape=(132,132,132), input_shape=(220,220,220), delta=(20,20,20)):
        super().__init__()
        # Tile the image using overlapping unet tiles
        self.tiling = tilingStrategy.OverlappingUnetTiling3D(image.shape, output_shape, input_shape, delta)
        # maintain an image and mask canvas
        self.image = tilingStrategy.Canvas(image) # Use a canvas for i/o on the image

        # Allocate an array to assemble the mask in if not specified
        if mask is None:
            self.mask = tilingStrategy.Canvas(np.zeros_like(image)) # Allocate a tensor where the segmentation mask is stored
        else:
            assert self.image.shape == mask.shape, 'The mask and image array need to be of the same shape'
            self.mask = tilingStrategy.Canvas(mask)

        # Expose the shape of the tiling
        self.shape = self.tiling.shape

        # Use a self avoiding queue to keep track of the segmentation state
        self.queue = SelfAvoidingQueue()

    # Expose the number of tiles 
    def __len__(self):
        return np.prod(self.shape)

    # Get the next tile from the queue
    def next(self):
        return self.queue.getTile()

    def storePredictionUpdateQueue(self, index, tile):
        # Write the predicted mask tile to the canvas
        self.writeSlice(index,tile)
        # Use an evaluation heuristic to determine which neighbours of the tile should be processed as well.
        adjacent = self.tiling.getAdjacentTiles(index)
        # TODO implement heuristic
        # Enqueue the neighbouring tiles
        self.queue.putTiles(adjacent)

    def queueHeuristic(self, index, tile):
        """
        To search for new positions where the FCCN should be evaluated the Flood Filling Paper states:
            "potential new positions are searched by examining the current state
            of the mask at the 6 planes x = x0 ± ∆x, y = y0 ± ∆y and z = z0 ± ∆z further restricted to
            [x0 − ∆x ≤ x ≤ x0 + ∆x] × [y0 − ∆y ≤ y ≤ y0 + ∆y] × [z0 − ∆z ≤ z ≤ z0 + ∆z]. For every
            such plane the object mask voxel with the highest value v is found, and if v ≥ tmove the location of
            that voxel is added to a list of new positions for the FFN."
        
        """
        aabb = self.tiling.getOutputTile(index)
        cp = [(aabb[d] + aabb[d+3])//2 for d in range(3)] # calculate the center point of the boundary box
        # construct the domain of the evaluation cube
        x = (cp[0]-self.tiling.delta[0],cp[0]+self.tiling.delta[0],)
        y = (cp[1]-self.tiling.delta[1],cp[1]+self.tiling.delta[1],)
        z = (cp[2]-self.tiling.delta[2],cp[2]+self.tiling.delta[2],)

        # construct the slice arguments to obtain the evaluation plains
        planes = [
            np.s_[x[0],y[0]:y[1],z[0]:z[1]],
            np.s_[x[1]-1,y[0]:y[1],z[0]:z[1]],

            np.s_[x[0]:x[1],y[0],z[0]:z[1]],
            np.s_[x[0]:x[1],y[1]-1,z[0]:z[1]],

            np.s_[x[0]:x[1],y[0]:y[1],z[0]],
            np.s_[x[0]:x[1],y[0]:y[1],z[1]-1],
        ]

        # TODO get mean of mask at planes, threshold and assemble tiles that should be explored

        return planes

    def getSlice(self, index, outputOnly=False):
        """Get the i-th input tile of the image.

        Parameters
        ----------
        index : int
            the index of the image tile
        outputOnly : bool
            wheter the slice should be narrowed down to the output region only

        Returns
        -------
        3d tensor
            the i-th input tile of the image
        """
        if outputOnly:
            aabb = self.tiling.getOutputTile(index)
        else:
            # get the aabb of the unet input slice
            aabb = self.tiling.getInputTile(index)
        # read out the aabb from the image data
        data = self.image.cropAndPadAABB(aabb)
        return data

    def writeSlice(self, index, tile):
        """Writes the i-th tile of the mask to it's corresponding position in the mask tensor.

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
        self.mask.writeAABB(aabb, tile)
# %%
# Tests
ff = FloodFiller(np.zeros((100,100,100)), mask=None, output_shape=(10,10,10), input_shape=(20,20,20), delta=(5,5,5))


# %%
