"""
Flow cytometry data loading and preprocessing.
"""

import numpy as np
import pandas as pd
from fcsparser import parse

class FlowDataLoader:
    def __init__(self):
        self.data = None
        self.metadata = None
        
    def load_fcs(self, filepath):
        """
        Load flow cytometry data from an FCS file.
        
        Parameters
        ----------
        filepath : str
            Path to the FCS file
            
        Returns
        -------
        tuple
            (data, metadata) where data is a pandas DataFrame and metadata is a dict
        """
        self.data, self.metadata = parse(filepath, reformat_meta=True)
        return self.data, self.metadata
    
    def get_markers(self):
        """
        Get the list of markers from the loaded data.
        
        Returns
        -------
        list
            List of marker names
        """
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data.columns.tolist()
    
    def select_markers(self, markers):
        """
        Select specific markers from the data.
        
        Parameters
        ----------
        markers : list
            List of marker names to select
            
        Returns
        -------
        pandas.DataFrame
            Data with selected markers
        """
        if self.data is None:
            raise ValueError("No data loaded")
        return self.data[markers] 