"""
Quick start guide for using the ICRF Manifold Analysis package.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.flow_data_loader import FlowDataLoader
from src.data_processing.preprocessing import FlowDataPreprocessor
from src.manifold_learning.cell_cycle_analyzer import CellCycleAnalyzer
from src.visualization.polyploidy_garage import PolyploidyVisualizer

def main():
    # Initialize components
    loader = FlowDataLoader()
    preprocessor = FlowDataPreprocessor()
    analyzer = CellCycleAnalyzer()
    visualizer = PolyploidyVisualizer()
    
    # Load data
    data_path = "data/raw/0hr_control.fcs"
    data, metadata = loader.load_fcs(data_path)
    
    # Preprocess data
    markers = loader.get_markers()
    selected_data = loader.select_markers(markers)
    scaled_data = preprocessor.scale_data(selected_data)
    cleaned_data = preprocessor.remove_outliers(scaled_data)
    
    # Analyze data
    manifold_data = analyzer.fit_transform(cleaned_data)
    
    # Visualize results
    visualizer.plot_manifold(manifold_data, title="Cell Cycle Manifold")
    visualizer.save_plot("docs/images/manifold_visualization.png")

if __name__ == "__main__":
    main() 