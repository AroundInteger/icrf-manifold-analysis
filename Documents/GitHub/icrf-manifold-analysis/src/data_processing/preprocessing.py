"""
Data preprocessing utilities for flow cytometry data.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FlowDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        
    def scale_data(self, data):
        """
        Scale the data using StandardScaler.
        
        Parameters
        ----------
        data : array-like
            Input data matrix
            
        Returns
        -------
        array-like
            Scaled data
        """
        return self.scaler.fit_transform(data)
    
    def apply_pca(self, data, n_components=2):
        """
        Apply PCA to reduce dimensionality.
        
        Parameters
        ----------
        data : array-like
            Input data matrix
        n_components : int, optional
            Number of components to keep
            
        Returns
        -------
        array-like
            Transformed data
        """
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(data)
    
    def remove_outliers(self, data, threshold=3):
        """
        Remove outliers using z-score thresholding.
        
        Parameters
        ----------
        data : array-like
            Input data matrix
        threshold : float, optional
            Z-score threshold for outlier removal
            
        Returns
        -------
        array-like
            Data with outliers removed
        """
        z_scores = np.abs((data - data.mean()) / data.std())
        return data[(z_scores < threshold).all(axis=1)] 