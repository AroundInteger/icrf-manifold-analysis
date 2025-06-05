"""
Cell cycle analysis using manifold learning techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

class CellCycleManifoldAnalyzer:
    """
    Manifold learning analysis for cell cycle flow cytometry data
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.manifolds = {}
        self.embeddings = {}
        self.population_labels = {}
        
    def load_flow_data(self, gfp_data, draq5_data, timepoint_label):
        """
        Load and preprocess flow cytometry data
        
        Parameters:
        -----------
        gfp_data : array-like
            GFP-Cyclin B1 fluorescence values
        draq5_data : array-like  
            DRAQ5 DNA content values
        timepoint_label : str
            Label for this timepoint (e.g., '0hr', '24hr', '42hr')
        """
        # Combine into feature matrix
        X = np.column_stack([gfp_data, draq5_data])
        
        # Store raw data
        self.manifolds[timepoint_label] = {
            'raw_data': X,
            'gfp': gfp_data,
            'draq5': draq5_data,
            'n_cells': len(gfp_data)
        }
        
        print(f"Loaded {len(gfp_data)} cells for timepoint {timepoint_label}")
        return X
    
    def segment_populations_garage(self, timepoint_label, auto_threshold=True, manual_thresholds=None):
        """
        Segment populations using "polyploidy garage" model
        Each "floor" represents a 2^n ploidy level
        
        Parameters:
        -----------
        timepoint_label : str
            Which timepoint to segment
        auto_threshold : bool
            Automatically detect ploidy levels using peak detection
        manual_thresholds : list
            Manual DRAQ5 thresholds if auto_threshold=False
        """
        data = self.manifolds[timepoint_label]
        draq5 = data['draq5']
        
        if auto_threshold:
            # Detect ploidy levels automatically using histogram peaks
            from scipy.signal import find_peaks
            from scipy.stats import gaussian_kde
            
            # Create smooth histogram
            kde = gaussian_kde(draq5)
            x_range = np.linspace(draq5.min(), draq5.max(), 1000)
            density = kde(x_range)
            
            # Find peaks (ploidy levels)
            peaks, _ = find_peaks(density, height=0.1*density.max(), distance=50)
            peak_positions = x_range[peaks]
            
            # Sort peaks and assign ploidy levels
            peak_positions = np.sort(peak_positions)
            print(f"\nDetected ploidy peaks at DRAQ5 levels: {peak_positions}")
            
            # Create thresholds between peaks
            if len(peak_positions) > 1:
                thresholds = []
                for i in range(len(peak_positions)-1):
                    # Threshold = midpoint between consecutive peaks
                    threshold = (peak_positions[i] + peak_positions[i+1]) / 2
                    thresholds.append(threshold)
            else:
                thresholds = [peak_positions[0] * 1.5] if len(peak_positions) > 0 else [300]
                
        else:
            thresholds = manual_thresholds or [300, 500, 800]  # Default for 2N/4N/8N/16N
        
        # Assign cells to "garage floors"
        labels = np.zeros(len(draq5), dtype=int)
        for i, threshold in enumerate(thresholds):
            labels[draq5 >= threshold] = i + 1
        
        # Store labels and statistics
        self.population_labels[timepoint_label] = labels
        
        pop_counts = np.bincount(labels)
        pop_fractions = pop_counts / len(labels)
        
        # Create garage floor mapping
        garage_floors = {0: '2N (Ground)', 1: '4N (Floor 1)', 2: '8N (Floor 2)', 
                        3: '16N (Floor 3)', 4: '32N+ (Floor 4+)'}
        
        print(f"\n{timepoint_label} Polyploidy Garage Distribution:")
        for i, count in enumerate(pop_counts):
            floor_name = garage_floors.get(i, f'{2**(i+1)}N (Floor {i})')
            print(f"  {floor_name}: {count} cells ({pop_fractions[i]:.1%})")
        
        # Store garage metadata
        self.manifolds[timepoint_label]['garage_floors'] = garage_floors
        self.manifolds[timepoint_label]['thresholds'] = thresholds
            
        return labels, pop_fractions, thresholds
    
    def learn_manifold(self, timepoint_label, method='umap', **kwargs):
        """
        Learn manifold embedding for a given timepoint
        
        Parameters:
        -----------
        timepoint_label : str
            Which timepoint to analyze
        method : str
            'umap', 'pca', or 'diffusion_map'
        **kwargs : dict
            Parameters for the manifold learning method
        """
        X = self.manifolds[timepoint_label]['raw_data']
        
        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == 'umap':
            # UMAP parameters optimized for cell cycle data
            default_params = {
                'n_neighbors': 50,  # Larger neighborhood for smooth manifolds
                'min_dist': 0.1,    # Allow close packing
                'n_components': 2,  # 2D for visualization
                'metric': 'euclidean',
                'random_state': self.random_state
            }
            default_params.update(kwargs)
            
            reducer = UMAP(**default_params)
            embedding = reducer.fit_transform(X_scaled)
            
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=self.random_state)
            embedding = reducer.fit_transform(X_scaled)
            
        # Store results
        self.embeddings[timepoint_label] = {
            'embedding': embedding,
            'method': method,
            'reducer': reducer,
            'scaler': scaler
        }
        
        print(f"Learned {method} manifold for {timepoint_label}")
        return embedding
    
    def compute_manifold_metrics(self, timepoint_label):
        """
        Compute quantitative metrics for manifold structure
        """
        if timepoint_label not in self.embeddings:
            raise ValueError(f"No embedding found for {timepoint_label}")
            
        embedding = self.embeddings[timepoint_label]['embedding']
        
        # Compute pairwise distances in embedding space
        distances = pdist(embedding)
        
        # Manifold metrics
        metrics = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'diameter': np.max(distances),
            'compactness': np.std(distances) / np.mean(distances),
        }
        
        # If we have population labels, compute separation metrics
        if timepoint_label in self.population_labels:
            labels = self.population_labels[timepoint_label]
            unique_labels = np.unique(labels)
            
            if len(unique_labels) > 1:
                # Compute inter-population distances
                inter_distances = []
                for i in range(len(unique_labels)):
                    for j in range(i+1, len(unique_labels)):
                        pop_i = embedding[labels == unique_labels[i]]
                        pop_j = embedding[labels == unique_labels[j]]
                        
                        # Mean distance between population centroids
                        centroid_i = np.mean(pop_i, axis=0)
                        centroid_j = np.mean(pop_j, axis=0)
                        inter_distances.append(np.linalg.norm(centroid_i - centroid_j))
                
                metrics['population_separation'] = np.mean(inter_distances)
        
        return metrics
    
    def visualize_manifold(self, timepoint_label, color_by='ploidy', figsize=(10, 8)):
        """
        Visualize the learned manifold
        
        Parameters:
        -----------
        timepoint_label : str
            Which timepoint to visualize
        color_by : str
            'ploidy', 'gfp', 'draq5', or 'density'
        """
        if timepoint_label not in self.embeddings:
            raise ValueError(f"No embedding found for {timepoint_label}")
            
        embedding = self.embeddings[timepoint_label]['embedding']
        data = self.manifolds[timepoint_label]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Original space
        ax1 = axes[0]
        if color_by == 'ploidy' and timepoint_label in self.population_labels:
            labels = self.population_labels[timepoint_label]
            scatter = ax1.scatter(data['draq5'], data['gfp'], 
                                c=labels, cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax1, label='Population')
        else:
            ax1.scatter(data['draq5'], data['gfp'], alpha=0.6, s=1, c='blue')
            
        ax1.set_xlabel('DRAQ5 (DNA Content)')
        ax1.set_ylabel('GFP-Cyclin B1')
        ax1.set_title(f'Original Space - {timepoint_label}')
        
        # Manifold space
        ax2 = axes[1]
        if color_by == 'ploidy' and timepoint_label in self.population_labels:
            labels = self.population_labels[timepoint_label]
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=labels, cmap='viridis', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='Population')
        elif color_by == 'gfp':
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=data['gfp'], cmap='plasma', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='GFP-Cyclin B1')
        elif color_by == 'draq5':
            scatter = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                                c=data['draq5'], cmap='plasma', alpha=0.6, s=1)
            plt.colorbar(scatter, ax=ax2, label='DRAQ5')
        else:
            ax2.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=1, c='red')
            
        ax2.set_xlabel('Manifold Dimension 1')
        ax2.set_ylabel('Manifold Dimension 2')
        ax2.set_title(f'Manifold Space - {timepoint_label}')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_polyploidy_garage(self, timepoint_label, figsize=(15, 10)):
        """
        Visualize the "polyploidy garage" concept with floor-by-floor analysis
        """
        if timepoint_label not in self.manifolds:
            raise ValueError(f"No data found for {timepoint_label}")
            
        data = self.manifolds[timepoint_label]
        labels = self.population_labels[timepoint_label]
        embedding = self.embeddings[timepoint_label]['embedding']
        garage_floors = data.get('garage_floors', {0: '2N', 1: '4N', 2: '8N+'})
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Original data with garage floor coloring
        ax = axes[0, 0]
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            floor_name = garage_floors.get(label, f'Floor {label}')
            ax.scatter(data['draq5'][mask], data['gfp'][mask], 
                      c=[colors[i]], alpha=0.6, s=1, label=floor_name)
        
        ax.set_xlabel('DRAQ5 (DNA Content)')
        ax.set_ylabel('GFP-Cyclin B1')
        ax.set_title(f'Polyploidy Garage - {timepoint_label}')
        ax.legend()
        
        # Plot 2: Manifold embedding with garage floors
        ax = axes[0, 1]
        for i, label in enumerate(unique_labels):
            mask = labels == label
            floor_name = garage_floors.get(label, f'Floor {label}')
            ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                      c=[colors[i]], alpha=0.6, s=1, label=floor_name)
        
        ax.set_xlabel('Manifold Dimension 1')
        ax.set_ylabel('Manifold Dimension 2')
        ax.set_title(f'Manifold Space - {timepoint_label}')
        ax.legend()
        
        # Plot 3: DRAQ5 histogram with floor boundaries
        ax = axes[0, 2]
        ax.hist(data['draq5'], bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        
        # Add floor boundary lines
        if 'thresholds' in data:
            for i, threshold in enumerate(data['thresholds']):
                ax.axvline(threshold, color='red', linestyle='--', 
                          label=f'Floor {i+1} boundary' if i == 0 else '')
        
        ax.set_xlabel('DRAQ5 (DNA Content)')
        ax.set_ylabel('Cell Count')
        ax.set_title('DNA Content Distribution')
        ax.legend()
        
        # Plot 4: Floor-by-floor cycle analysis
        ax = axes[1, 0]
        floor_stats = []
        for label in unique_labels:
            mask = labels == label
            floor_gfp = data['gfp'][mask]
            floor_draq5 = data['draq5'][mask]
            
            # Normalize DRAQ5 within each floor (relative cell cycle position)
            if len(floor_draq5) > 10:  # Only if enough cells
                draq5_norm = (floor_draq5 - floor_draq5.min()) / (floor_draq5.max() - floor_draq5.min())
                ax.scatter(draq5_norm, floor_gfp, c=[colors[label]], alpha=0.5, s=1,
                          label=garage_floors.get(label, f'Floor {label}'))
                
                floor_stats.append({
                    'floor': label,
                    'n_cells': len(floor_gfp),
                    'gfp_mean': np.mean(floor_gfp),
                    'gfp_std': np.std(floor_gfp),
                    'draq5_range': floor_draq5.max() - floor_draq5.min()
                })
        
        ax.set_xlabel('Normalized Cell Cycle Position (within floor)')
        ax.set_ylabel('GFP-Cyclin B1')
        ax.set_title('Cell Cycle Patterns by Floor')
        ax.legend()
        
        # Plot 5: Floor statistics
        ax = axes[1, 1]
        if floor_stats:
            floors = [s['floor'] for s in floor_stats]
            n_cells = [s['n_cells'] for s in floor_stats]
            floor_names = [garage_floors.get(f, f'Floor {f}') for f in floors]
            
            bars = ax.bar(floor_names, n_cells, color=colors[:len(floors)], alpha=0.7)
            ax.set_ylabel('Number of Cells')
            ax.set_title('Population per Floor')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add percentage labels on bars
            total_cells = sum(n_cells)
            for bar, count in zip(bars, n_cells):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + total_cells*0.01,
                       f'{count/total_cells:.1%}', ha='center', va='bottom')
        
        # Plot 6: 3D visualization (DRAQ5 vs GFP vs Floor)
        ax = axes[1, 2]
        # Create a 2D projection showing floor separation
        for i, label in enumerate(unique_labels):
            mask = labels == label
            # Use floor number as third dimension
            floor_height = np.full(np.sum(mask), label * 0.1)
            ax.scatter(data['draq5'][mask], data['gfp'][mask], 
                      c=[colors[i]], alpha=0.6, s=1,
                      label=garage_floors.get(label, f'Floor {label}'))
        
        ax.set_xlabel('DRAQ5 (DNA Content)')
        ax.set_ylabel('GFP-Cyclin B1')
        ax.set_title('Garage Structure Overview')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, floor_stats
    
    def compare_timepoints(self, reference_timepoint, comparison_timepoints):
        """
        Compare manifold structure across timepoints
        """
        results = {}
        
        # Get reference metrics
        ref_metrics = self.compute_manifold_metrics(reference_timepoint)
        results[reference_timepoint] = ref_metrics
        
        print(f"\nManifold Comparison (Reference: {reference_timepoint})")
        print("-" * 50)
        
        for tp in comparison_timepoints:
            if tp in self.embeddings:
                metrics = self.compute_manifold_metrics(tp)
                results[tp] = metrics
                
                # Compute changes relative to reference
                compactness_change = (metrics['compactness'] - ref_metrics['compactness']) / ref_metrics['compactness']
                diameter_change = (metrics['diameter'] - ref_metrics['diameter']) / ref_metrics['diameter']
                
                print(f"\n{tp}:")
                print(f"  Compactness change: {compactness_change:.1%}")
                print(f"  Diameter change: {diameter_change:.1%}")
                
                if 'population_separation' in metrics:
                    print(f"  Population separation: {metrics['population_separation']:.3f}")
        
        return results

def analyze_icrf_experiment(data_0hr, data_24hr, data_42hr):
    """
    Complete analysis pipeline for ICRF-193 experiment
    
    Parameters:
    -----------
    data_0hr, data_24hr, data_42hr : dict
        Each should contain 'gfp' and 'draq5' arrays
    """
    
    analyzer = CellCycleManifoldAnalyzer()
    
    # Load data for all timepoints
    print("Loading flow cytometry data...")
    analyzer.load_flow_data(data_0hr['gfp'], data_0hr['draq5'], '0hr')
    analyzer.load_flow_data(data_24hr['gfp'], data_24hr['draq5'], '24hr')
    analyzer.load_flow_data(data_42hr['gfp'], data_42hr['draq5'], '42hr')
    
    # Segment populations
    print("\nSegmenting cell populations...")
    for tp in ['0hr', '24hr', '42hr']:
        analyzer.segment_populations_garage(tp)
    
    # Learn manifolds
    print("\nLearning manifold embeddings...")
    for tp in ['0hr', '24hr', '42hr']:
        analyzer.learn_manifold(tp, method='umap')
    
    # Visualize results
    print("\nGenerating visualizations...")
    for tp in ['0hr', '24hr', '42hr']:
        analyzer.visualize_manifold(tp, color_by='ploidy')
    
    # Compare timepoints
    print("\nComparing manifold structure across timepoints...")
    comparison = analyzer.compare_timepoints('0hr', ['24hr', '42hr'])
    
    return analyzer, comparison

def load_flow_data_from_mat(filepath, gfp_key='gfp', draq5_key='draq5'):
    """
    Load flow cytometry data from MATLAB .mat file
    
    Parameters:
    -----------
    filepath : str
        Path to .mat file
    gfp_key : str
        Key for GFP data in the .mat file
    draq5_key : str
        Key for DRAQ5 data in the .mat file
    
    Returns:
    --------
    dict with 'gfp' and 'draq5' arrays
    """
    from scipy.io import loadmat
    
    # Load MATLAB file
    mat_data = loadmat(filepath)
    
    # Extract flow cytometry data
    # Handle different possible data structures
    if gfp_key in mat_data:
        gfp_data = mat_data[gfp_key].flatten()
    else:
        # Try common alternative names
        possible_gfp_keys = ['GFP', 'gfp_cyclin', 'GFP_Cyclin_B1', 'cyclin']
        for key in possible_gfp_keys:
            if key in mat_data:
                gfp_data = mat_data[key].flatten()
                break
        else:
            raise KeyError(f"GFP data not found. Available keys: {list(mat_data.keys())}")
    
    if draq5_key in mat_data:
        draq5_data = mat_data[draq5_key].flatten()
    else:
        # Try common alternative names
        possible_draq5_keys = ['DRAQ5', 'DNA', 'draq', 'dna_content']
        for key in possible_draq5_keys:
            if key in mat_data:
                draq5_data = mat_data[key].flatten()
                break
        else:
            raise KeyError(f"DRAQ5 data not found. Available keys: {list(mat_data.keys())}")
    
    print(f"Loaded {len(gfp_data)} cells from {filepath}")
    print(f"GFP range: {gfp_data.min():.1f} - {gfp_data.max():.1f}")
    print(f"DRAQ5 range: {draq5_data.min():.1f} - {draq5_data.max():.1f}")
    
    return {
        'gfp': gfp_data,
        'draq5': draq5_data
    }

def load_flow_data_from_csv(filepath, gfp_col='GFP_Cyclin_B1', draq5_col='DRAQ5'):
    """
    Load flow cytometry data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    gfp_col : str
        Column name for GFP data
    draq5_col : str
        Column name for DRAQ5 data
    
    Returns:
    --------
    dict with 'gfp' and 'draq5' arrays
    """
    df = pd.read_csv(filepath)
    
    print(f"CSV columns available: {list(df.columns)}")
    
    return {
        'gfp': df[gfp_col].values,
        'draq5': df[draq5_col].values
    } 