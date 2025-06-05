"""
Cell cycle analysis using manifold learning techniques.
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class CellCycleAnalyzer:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.manifold = TSNE(n_components=n_components)
        
    def fit_transform(self, data):
        """
        Fit the manifold learning model and transform the data.
        
        Parameters
        ----------
        data : array-like
            Input data matrix
            
        Returns
        -------
        array-like
            Transformed data in the manifold space
        """
        scaled_data = self.scaler.fit_transform(data)
        return self.manifold.fit_transform(scaled_data)
    
    def transform(self, data):
        """
        Transform new data using the fitted model.
        
        Parameters
        ----------
        data : array-like
            New data matrix
            
        Returns
        -------
        array-like
            Transformed data in the manifold space
        """
        scaled_data = self.scaler.transform(data)
        return self.manifold.transform(scaled_data) 