"""This module defines tools to apply a Unet architecture in the context of the flood filling procedure

Linus Meienberg
August 2020
"""

import tools.tilingStrategy as tilingStrategy
import queue

class SelfAvoidingQueue():
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue() # store all positions that should be visited in a queue
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
                self.queue.put(i)
    
    def getTile(self):
        """Return the index of the next tile in the queue.
        Marks this tile as visited.

        Returns
        -------
        int
            tile index
        """
        i = self.queue.get()
        self.visited.add(i)
        return i

def FloodFiller():
    pass