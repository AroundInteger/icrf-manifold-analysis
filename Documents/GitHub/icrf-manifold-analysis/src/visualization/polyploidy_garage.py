"""
Visualization tools for polyploidy analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PolyploidyVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_manifold(self, data, labels=None, title="Manifold Visualization"):
        """
        Plot data points in the manifold space.
        
        Parameters
        ----------
        data : array-like
            Data points in manifold space
        labels : array-like, optional
            Labels for the data points
        title : str, optional
            Plot title
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        scatter = self.ax.scatter(data[:, 0], data[:, 1], c=labels if labels is not None else 'blue')
        if labels is not None:
            plt.colorbar(scatter)
        self.ax.set_title(title)
        self.ax.set_xlabel("Component 1")
        self.ax.set_ylabel("Component 2")
        
    def plot_transition_map(self, grid_points, probabilities, title="Transition Map"):
        """
        Plot the transition probability map.
        
        Parameters
        ----------
        grid_points : array-like
            Grid points
        probabilities : array-like
            Transition probabilities
        title : str, optional
            Plot title
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        contour = self.ax.contourf(grid_points[0], grid_points[1], probabilities)
        plt.colorbar(contour)
        self.ax.set_title(title)
        self.ax.set_xlabel("Component 1")
        self.ax.set_ylabel("Component 2")
        
    def save_plot(self, filepath):
        """
        Save the current plot to a file.
        
        Parameters
        ----------
        filepath : str
            Path to save the plot
        """
        if self.fig is None:
            raise ValueError("No plot to save")
        self.fig.savefig(filepath)
        plt.close(self.fig) 