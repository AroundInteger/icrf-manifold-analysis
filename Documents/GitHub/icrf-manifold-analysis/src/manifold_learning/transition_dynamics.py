"""
Analysis of cell cycle transition dynamics.
"""

import numpy as np
from scipy.stats import gaussian_kde

class TransitionDynamics:
    def __init__(self):
        self.kde = None
        
    def fit(self, data):
        """
        Fit a kernel density estimator to the data.
        
        Parameters
        ----------
        data : array-like
            Input data matrix
        """
        self.kde = gaussian_kde(data.T)
        
    def compute_transition_probability(self, point):
        """
        Compute the transition probability at a given point.
        
        Parameters
        ----------
        point : array-like
            Point in the manifold space
            
        Returns
        -------
        float
            Transition probability
        """
        if self.kde is None:
            raise ValueError("Model must be fitted before computing probabilities")
        return self.kde(point)
    
    def compute_transition_map(self, grid_points):
        """
        Compute transition probabilities over a grid of points.
        
        Parameters
        ----------
        grid_points : array-like
            Grid of points to evaluate
            
        Returns
        -------
        array-like
            Transition probabilities at grid points
        """
        if self.kde is None:
            raise ValueError("Model must be fitted before computing probabilities")
        return self.kde(grid_points.T) 