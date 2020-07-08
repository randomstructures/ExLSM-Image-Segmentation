#%%
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
  
#%% 
img = cv2.imread('grid.png')
"""
cv2 : image = numpy tensor (x, y, channels) but channels are (blue, green, red)
plt: image = numpy tensor (x,y,channels) but channels are (red,green,blue[,alpha])
inverse cv2 channel dimension to convert
"""


img = img[:,:,::-1]
rows, cols, ch = img.shape 
  
pts1 = np.float32([[50, 50], 
                   [200, 50],  
                   [50, 200]]) 
  
pts2 = np.float32([[10, 100], 
                   [200, 50],  
                   [100, 250]]) 
  
M = cv2.getAffineTransform(pts1, pts2)
"""
Open CV required the transformation matrix to be of type float32
""" 
dst = cv2.warpAffine(img, M, (cols, rows)) 
  
plt.subplot(121) 
plt.imshow(img) 
plt.title('Input') 
  
plt.subplot(122) 
plt.imshow(dst) 
plt.title('Output') 
  
plt.show() 
  
# %%
