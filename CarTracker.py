# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:14:36 2017

@author: ankura
"""

import numpy as np

#ALl arguments are tuples, but are internally converted to NUmpy arrays for calculation
#Track Centroid (x,y) and Bounding boxes (x1, y1, x2, y2)
class CarTracker:
    alpha = 0.1
    
    def __init__(self, centroid, bbox):
        self.centroid = centroid
        self.bbox = bbox
    
    def update(self, centroid, bbox):
        self.centroid = np.array(self.centroid)
        self.bbox = np.array(self.bbox)
        self.centroid = self.centroid + self.alpha*(centroid - self.centroid);
        self.bbox = self.bbox + self.alpha*(bbox - self.bbox);
        self.centroid = tuple(self.centroid.astype(int))
        self.bbox = tuple(self.bbox.astype(int))
        
    def __str__(self):
        return ("Centroid {}, Bbox {}").format(self.centroid, self.bbox) 
    
    def __repr__(self):
        return ("Centroid {}, Bbox {}").format(self.centroid, self.bbox) 
    
    
#def __main__():
#    centroid = (5,5)
#    bbox = (1,2,3,4)
#    c1 = CarTracker(centroid, bbox)
#    print(c1)
#    c1
#    centroid2 = (7,7)
#    bbox2 = (1,9,6,4)
#    c1.update(centroid2, bbox2)
#    c1