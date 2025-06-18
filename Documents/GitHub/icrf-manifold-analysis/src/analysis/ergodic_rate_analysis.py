import numpy as np
from scipy import stats
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import vonmises

class ErgodicRateAnalysis:
    """
    Implementation of Ergodic Rate Analysis (ERA) for cell cycle progression
    in DRAQ5-GFP space.
    """
    def __init__(self, draq5, gfp, phases):
        """
        Initialize ERA with cell cycle data.
        
        Parameters:
        -----------
        draq5 : array-like
            DRAQ5 values (DNA content)
        gfp : array-like
            GFP values (Cyclin B1)
        phases : array-like
            Cell cycle phase labels (0: G1, 1: S, 2: G2/M)
        """
        self.draq5 = np.array(draq5)
        self.gfp = np.array(gfp)
        self.phases = np.array(phases)
        
        # Calculate the cell cycle trajectory parameter I
        self.I = self._calculate_trajectory_parameter()
        
        # Calculate probability density f(I)
        self.f_I = self._calculate_probability_density()
        
        # Calculate cumulative inverse density F(I)
        self.F_I = self._calculate_cumulative_inverse_density()
        
        # Population growth rate (assuming exponential growth)
        self.alpha = np.log(2) / 24  # Assuming 24-hour doubling time
        
    def _calculate_trajectory_parameter(self):
        """
        Calculate the cell cycle trajectory parameter I.
        For our DRAQ5-GFP system, we'll use a combination of both markers
        and make it periodic by wrapping G2/M back to G1.
        """
        # Normalize both markers to [0,1] range
        draq5_norm = (self.draq5 - self.draq5.min()) / (self.draq5.max() - self.draq5.min())
        gfp_norm = (self.gfp - self.gfp.min()) / (self.gfp.max() - self.gfp.min())
        
        # Combine markers with appropriate weights
        # DRAQ5 has higher weight as it's more directly related to DNA content
        I = 0.7 * draq5_norm + 0.3 * gfp_norm
        
        # Make the trajectory periodic by wrapping G2/M back to G1
        # First, identify G2/M cells (high DRAQ5 and high GFP)
        g2m_mask = (draq5_norm > 0.7) & (gfp_norm > 0.7)
        
        # For G2/M cells, map their position to the beginning of the cycle
        # We'll use a smooth transition to avoid discontinuities
        transition_start = 0.7  # Start of G2/M transition
        transition_width = 0.1  # Width of transition region
        
        # Calculate transition weight
        transition_weight = np.zeros_like(I)
        mask = I > transition_start
        transition_weight[mask] = (I[mask] - transition_start) / transition_width
        transition_weight = np.clip(transition_weight, 0, 1)
        
        # Create periodic mapping
        I_periodic = I.copy()
        I_periodic[g2m_mask] = transition_weight[g2m_mask] * (I[g2m_mask] - 1) + (1 - transition_weight[g2m_mask]) * I[g2m_mask]
        
        return I_periodic
    
    def _calculate_probability_density(self):
        """
        Estimate the probability density f(I) using a circular KDE (von Mises kernel) for periodicity.
        Returns f_I (density) and I_range (positions).
        """
        I_values = self.project_to_trajectory(self.gfp, self.draq5)
        I_range = np.linspace(0, 1, 200)
        theta_values = 2 * np.pi * I_values
        theta_range = 2 * np.pi * I_range
        # Bandwidth parameter for von Mises (kappa); higher kappa = narrower kernel
        kappa = 50  # You can tune this value for smoothing
        f_I = np.zeros_like(I_range)
        for i, theta0 in enumerate(theta_range):
            # vonmises.pdf is already normalized over the circle
            kernel_vals = vonmises.pdf(theta0 - theta_values, kappa)
            f_I[i] = np.sum(kernel_vals)
        f_I /= np.trapz(f_I, I_range)  # Normalize to area 1
        # Enforce periodicity explicitly
        avg = 0.5 * (f_I[0] + f_I[-1])
        f_I[0] = avg
        f_I[-1] = avg
        assert np.allclose(f_I[0], f_I[-1], rtol=1e-8), "Density is not periodic!"
        return f_I, I_range
    
    def _calculate_cumulative_inverse_density(self):
        """
        Calculate the cumulative inverse density F(I) = ∫(1/f(I))dI
        Enforce strict periodicity: F(0) = F(1), F(I) strictly increasing, and smooth at boundaries.
        """
        f_I, I_range = self._calculate_probability_density()
        
        # Calculate the inverse density 1/f(I)
        inverse_density = 1.0 / np.maximum(f_I, 1e-10)  # Avoid division by zero
        dI = I_range[1] - I_range[0]
        
        # Extend f_I and I_range periodically for smoothness at the boundaries
        I_extended = np.concatenate([I_range - 1, I_range, I_range + 1])
        inv_density_extended = np.concatenate([inverse_density, inverse_density, inverse_density])
        
        # Cumulative sum over the central period
        F_cumsum = np.cumsum(inv_density_extended) * dI
        # Extract the central period
        n = len(I_range)
        F_I = F_cumsum[n:2*n] - F_cumsum[n]
        
        # Normalize so that F(0) = 0 and F(1) = 1
        F_I = F_I - F_I[0]
        F_I = F_I / (F_I[-1] if F_I[-1] != 0 else 1)
        
        # Interpolate to ensure F(0) = F(1) exactly and smoothness at the boundary
        # Only append 1.0 if I_range does not already end at 1.0
        if np.isclose(I_range[-1], 1.0):
            periodic_I = I_range
            periodic_F = F_I
        else:
            periodic_I = np.append(I_range, 1.0)
            periodic_F = np.append(F_I, F_I[0])  # wrap around
        F_interp = PchipInterpolator(periodic_I, periodic_F, extrapolate=True)
        F_I = F_interp(I_range)
        
        # Ensure strict monotonicity
        F_I = np.maximum.accumulate(F_I)
        F_I = F_I - F_I[0]
        F_I = F_I / (F_I[-1] if F_I[-1] != 0 else 1)
        
        return F_I
    
    def calculate_progression_rate(self):
        """
        Calculate the rate of progression ω(I) using Eq. 2:
        ω(I) = α / (2 - F(I)/f(I))
        
        Where:
        - f(I) is the probability density
        - F(I) is the cumulative inverse density ∫(1/f(I))dI
        - α is the growth rate parameter
        """
        f_I, I_range = self._calculate_probability_density()
        F_I = self._calculate_cumulative_inverse_density()
        
        # Calculate F(I)/f(I) with safeguards
        ratio = F_I/f_I
        
        # Add safeguards to prevent unrealistic values
        ratio = np.clip(ratio, 0, 1.99)  # Ensure denominator is positive but less than 2
        
        # Calculate ω(I)
        omega_I = self.alpha / (2 - ratio)
        
        # Clip unrealistic values
        omega_I = np.clip(omega_I, 0, 10)  # Cap at 10 h⁻¹
        
        # Ensure smooth transition at the boundary
        transition_width = 0.1
        transition_mask = I_range > (1 - transition_width)
        if np.any(transition_mask):
            # Smoothly transition to the beginning of the cycle
            transition_weight = (I_range[transition_mask] - (1 - transition_width)) / transition_width
            omega_I[transition_mask] = (1 - transition_weight) * omega_I[transition_mask] + \
                                     transition_weight * omega_I[0]
        
        return omega_I, I_range, f_I, F_I, ratio
    
    def calculate_real_time(self):
        """
        Calculate the real-time axis t(I) by integrating 1/ω(I)
        """
        omega_I, I_range, _, _, _ = self.calculate_progression_rate()
        t_I = cumtrapz(1/omega_I, I_range, initial=0)
        return t_I, I_range
    
    def plot_era_results(self):
        """
        Plot the ERA results including probability density, F(I), F(I)/f(I),
        and real-time axis with enhanced visualization.
        """
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        
        # Define phase boundaries in I space
        phase_boundaries = {
            'G1/S': 0.3,  # Approximate boundary between G1 and S
            'S/G2': 0.7   # Approximate boundary between S and G2/M
        }
        
        # Plot 1: Probability density f(I)
        f_I, I_range = self._calculate_probability_density()
        ax1.plot(I_range, f_I, 'b-', linewidth=2, label='f(I)')
        
        # Add phase boundaries and annotations
        for phase, boundary in phase_boundaries.items():
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            ax1.text(boundary, ax1.get_ylim()[1]*0.9, phase, 
                    rotation=90, va='top', ha='right')
        
        # Add phase-specific histograms
        for phase, (start, end) in enumerate([
            (0, phase_boundaries['G1/S']),
            (phase_boundaries['G1/S'], phase_boundaries['S/G2']),
            (phase_boundaries['S/G2'], 1)
        ]):
            phase_mask = self.phases == phase
            if np.any(phase_mask):
                phase_I = self.I[phase_mask]
                ax1.hist(phase_I, bins=20, alpha=0.3, 
                        density=True, label=f'Phase {phase} Histogram')
        
        ax1.set_xlabel('Cell Cycle Position (I)', fontsize=12)
        ax1.set_ylabel('Probability Density', fontsize=12)
        ax1.set_title('Probability Density Distribution', fontsize=14, pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # Plot 2: F(I)
        F_I = self._calculate_cumulative_inverse_density()
        ax2.plot(I_range, F_I, 'g-', linewidth=2, label='F(I)')
        
        # Add phase boundaries and annotations
        for phase, boundary in phase_boundaries.items():
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            ax2.text(boundary, ax2.get_ylim()[1]*0.9, phase, 
                    rotation=90, va='top', ha='right')
        
        ax2.set_xlabel('Cell Cycle Position (I)', fontsize=12)
        ax2.set_ylabel('Cumulative Inverse Density', fontsize=12)
        ax2.set_title('Cumulative Inverse Density F(I)', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Plot 3: F(I)/f(I) ratio
        omega_I, I_range, f_I, F_I, ratio = self.calculate_progression_rate()
        ax3.plot(I_range, ratio, 'r-', linewidth=2, label='F(I)/f(I)')
        
        # Add phase boundaries and annotations
        for phase, boundary in phase_boundaries.items():
            ax3.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            ax3.text(boundary, ax3.get_ylim()[1]*0.9, phase, 
                    rotation=90, va='top', ha='right')
        
        ax3.set_xlabel('Cell Cycle Position (I)', fontsize=12)
        ax3.set_ylabel('F(I)/f(I)', fontsize=12)
        ax3.set_title('Cumulative Inverse Density Ratio', fontsize=14, pad=20)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        
        # Plot 4: Real-time axis t(I)
        t_I, I_range = self.calculate_real_time()
        ax4.plot(I_range, t_I, 'g-', linewidth=2, label='t(I)')
        
        # Add phase boundaries and annotations
        for phase, boundary in phase_boundaries.items():
            ax4.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
            ax4.text(boundary, ax4.get_ylim()[1]*0.9, phase, 
                    rotation=90, va='top', ha='right')
        
        # Add phase duration annotations
        phase_durations = {
            'G1': t_I[np.abs(I_range - phase_boundaries['G1/S']).argmin()],
            'S': t_I[np.abs(I_range - phase_boundaries['S/G2']).argmin()] - 
                 t_I[np.abs(I_range - phase_boundaries['G1/S']).argmin()],
            'G2/M': t_I[-1] - t_I[np.abs(I_range - phase_boundaries['S/G2']).argmin()]
        }
        
        # Annotate phase durations
        y_pos = ax4.get_ylim()[1] * 0.1
        for phase, duration in phase_durations.items():
            ax4.text(0.5, y_pos, f'{phase}: {duration:.1f}h', 
                    ha='center', va='bottom', fontsize=10)
            y_pos += ax4.get_ylim()[1] * 0.1
        
        ax4.set_xlabel('Cell Cycle Position (I)', fontsize=12)
        ax4.set_ylabel('Real Time (hours)', fontsize=12)
        ax4.set_title('Real-Time Axis', fontsize=14, pad=20)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=10)
        
        # Add overall title
        fig.suptitle('Ergodic Rate Analysis of Cell Cycle Progression', 
                    fontsize=16, y=0.95)
        
        plt.tight_layout()
        plt.savefig('docs/images/era_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig

    def plot_periodic_density(self, f_I, I_range, num_periods=3):
        """
        Plot the probability density f(I) across multiple periods to demonstrate periodicity.
        
        Args:
            f_I: Probability density values
            I_range: Cell cycle positions
            num_periods: Number of periods to show (default: 3)
        """
        plt.figure(figsize=(12, 6))
        
        # Create extended range for multiple periods
        period = 1.0
        points_per_period = len(I_range)
        total_points = (num_periods * points_per_period) - (num_periods - 1)  # Account for overlapping points
        
        # Create extended range
        extended_I = np.linspace(-period, num_periods*period, total_points)
        
        # Create extended density by repeating the base period
        # Each period shares its first point with the last point of the previous period
        extended_f_I = []
        for i in range(num_periods):
            if i == 0:
                extended_f_I.extend(f_I)
            else:
                extended_f_I.extend(f_I[1:])  # Skip first point as it's the same as last point of previous period
        
        extended_f_I = np.array(extended_f_I)
        
        # Plot the extended density
        plt.plot(extended_I, extended_f_I, 'b-', label='f(I)', alpha=0.7, linewidth=2)
        
        # Add vertical lines to mark period boundaries
        for i in range(-1, num_periods+1):
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
            plt.text(i, plt.ylim()[1]*0.95, f'I={i}', 
                    horizontalalignment='center', verticalalignment='top')
        
        # Add phase boundaries for the central period
        phase_boundaries = {
            'G1/S': 0.3,
            'S/G2': 0.7
        }
        for phase, pos in phase_boundaries.items():
            plt.axvline(x=pos, color='g', linestyle=':', alpha=0.5)
            plt.text(pos, plt.ylim()[1]*0.9, phase, 
                    horizontalalignment='center', verticalalignment='top')
        
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Probability Density f(I)')
        plt.title('Periodic Probability Density f(I)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig('docs/images/periodic_density.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_periodic_cumulative(self, F_I, I_range, num_periods=3):
        """
        Plot the cumulative inverse density F(I) across multiple periods to demonstrate periodicity.
        
        Args:
            F_I: Cumulative inverse density values (periodic with F(0) = F(1))
            I_range: Cell cycle positions
            num_periods: Number of periods to show (default: 3)
        """
        plt.figure(figsize=(12, 6))
        
        # Create extended range for multiple periods
        period = 1.0
        points_per_period = len(I_range)
        total_points = (num_periods * points_per_period) - (num_periods - 1)  # Account for overlapping points
        
        # Create extended range
        extended_I = np.linspace(-period, num_periods*period, total_points)
        
        # Create extended F(I) by repeating the base period
        # Since F(I) is periodic with F(0) = F(1), we can simply repeat it
        extended_F_I = []
        for i in range(num_periods):
            if i == 0:
                extended_F_I.extend(F_I)
            else:
                extended_F_I.extend(F_I[1:])  # Skip first point as it's the same as last point of previous period
        
        extended_F_I = np.array(extended_F_I)
        
        # Plot the extended cumulative
        plt.plot(extended_I, extended_F_I, 'b-', label='F(I)', alpha=0.7, linewidth=2)
        
        # Add vertical lines to mark period boundaries
        for i in range(-1, num_periods+1):
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
            plt.text(i, plt.ylim()[1]*0.95, f'I={i}', 
                    horizontalalignment='center', verticalalignment='top')
        
        # Add phase boundaries for the central period
        phase_boundaries = {
            'G1/S': 0.3,
            'S/G2': 0.7
        }
        for phase, pos in phase_boundaries.items():
            plt.axvline(x=pos, color='g', linestyle=':', alpha=0.5)
            plt.text(pos, plt.ylim()[1]*0.9, phase, 
                    horizontalalignment='center', verticalalignment='top')
        
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Cumulative Inverse Density F(I)')
        plt.title('Periodic Cumulative Inverse Density F(I)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig('docs/images/periodic_cumulative.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_periodic_ratio(self, F_I, f_I, I_range, num_periods=3):
        """
        Plot the ratio F(I)/f(I) across multiple periods to demonstrate periodicity.
        
        Args:
            F_I: Cumulative inverse density values
            f_I: Probability density values
            I_range: Cell cycle positions
            num_periods: Number of periods to show (default: 3)
        """
        plt.figure(figsize=(12, 6))
        
        # Create extended range for multiple periods
        period = 1.0
        points_per_period = len(I_range)
        total_points = (num_periods * points_per_period) - (num_periods - 1)  # Account for overlapping points
        
        # Create extended range
        extended_I = np.linspace(-period, num_periods*period, total_points)
        
        # Create extended f(I) by repeating the base period
        extended_f_I = []
        for i in range(num_periods):
            if i == 0:
                extended_f_I.extend(f_I)
            else:
                extended_f_I.extend(f_I[1:])  # Skip first point as it's the same as last point of previous period
        
        # Create extended F(I) by repeating the base period with offsets
        extended_F_I = []
        for i in range(num_periods):
            if i == 0:
                extended_F_I.extend(F_I)
            else:
                offset = F_I[-1] * i
                extended_F_I.extend(F_I[1:] + offset)
        
        # Calculate the ratio
        extended_ratio = np.array(extended_F_I) / np.array(extended_f_I)
        
        # Plot the extended ratio
        plt.plot(extended_I, extended_ratio, 'b-', label='F(I)/f(I)', alpha=0.7, linewidth=2)
        
        # Add horizontal line at y=2 to show the critical value
        plt.axhline(y=2, color='r', linestyle='--', alpha=0.3, label='Critical Value (2)')
        
        # Add vertical lines to mark period boundaries
        for i in range(-1, num_periods+1):
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
            plt.text(i, plt.ylim()[1]*0.95, f'I={i}', 
                    horizontalalignment='center', verticalalignment='top')
        
        # Add phase boundaries for the central period
        phase_boundaries = {
            'G1/S': 0.3,
            'S/G2': 0.7
        }
        for phase, pos in phase_boundaries.items():
            plt.axvline(x=pos, color='g', linestyle=':', alpha=0.5)
            plt.text(pos, plt.ylim()[1]*0.9, phase, 
                    horizontalalignment='center', verticalalignment='top')
        
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Ratio F(I)/f(I)')
        plt.title('Periodic Ratio F(I)/f(I)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig('docs/images/periodic_ratio.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_periodic_progression_rate(self, omega_I, I_range, num_periods=3):
        """
        Plot the progression rate ω(I) across multiple periods to demonstrate periodicity.
        
        Args:
            omega_I: Progression rate values
            I_range: Cell cycle positions
            num_periods: Number of periods to show (default: 3)
        """
        plt.figure(figsize=(12, 6))
        
        # Create extended range for multiple periods
        period = 1.0
        points_per_period = len(I_range)
        total_points = (num_periods * points_per_period) - (num_periods - 1)  # Account for overlapping points
        
        # Create extended range
        extended_I = np.linspace(-period, num_periods*period, total_points)
        
        # Create extended omega(I) by repeating the base period
        # Since omega(I) is periodic, we can simply repeat it
        extended_omega_I = []
        for i in range(num_periods):
            if i == 0:
                extended_omega_I.extend(omega_I)
            else:
                extended_omega_I.extend(omega_I[1:])  # Skip first point as it's the same as last point of previous period
        
        extended_omega_I = np.array(extended_omega_I)
        
        # Plot the extended progression rate
        plt.plot(extended_I, extended_omega_I, 'b-', label='ω(I)', alpha=0.7, linewidth=2)
        
        # Add vertical lines to mark period boundaries
        for i in range(-1, num_periods+1):
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.3)
            plt.text(i, plt.ylim()[1]*0.95, f'I={i}', 
                    horizontalalignment='center', verticalalignment='top')
        
        # Add phase boundaries for the central period
        phase_boundaries = {
            'G1/S': 0.3,
            'S/G2': 0.7
        }
        for phase, pos in phase_boundaries.items():
            plt.axvline(x=pos, color='g', linestyle=':', alpha=0.5)
            plt.text(pos, plt.ylim()[1]*0.9, phase, 
                    horizontalalignment='center', verticalalignment='top')
        
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Progression Rate ω(I) [h⁻¹]')
        plt.title('Periodic Progression Rate ω(I)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save the plot
        plt.savefig('docs/images/periodic_progression_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_progression_parameters(self, omega_I, I_range, f_I, F_I, ratio):
        """
        Plot the parameters involved in calculating ω(I) = α/(2 - F(I)/f(I))
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: F(I)/f(I) ratio
        plt.subplot(2, 2, 1)
        plt.plot(I_range, ratio, 'b-', label='F(I)/f(I)', alpha=0.7)
        plt.axhline(y=2, color='r', linestyle='--', alpha=0.3, label='Critical Value (2)')
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Ratio F(I)/f(I)')
        plt.title('Ratio F(I)/f(I)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Denominator (2 - F(I)/f(I))
        plt.subplot(2, 2, 2)
        denominator = 2 - ratio
        plt.plot(I_range, denominator, 'g-', label='2 - F(I)/f(I)', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero')
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Denominator')
        plt.title('Denominator (2 - F(I)/f(I))')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 3: Growth rate parameter α
        plt.subplot(2, 2, 3)
        plt.axhline(y=self.alpha, color='b', label=f'α = {self.alpha}', alpha=0.7)
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Growth Rate α')
        plt.title('Growth Rate Parameter α')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Final progression rate ω(I)
        plt.subplot(2, 2, 4)
        plt.plot(I_range, omega_I, 'r-', label='ω(I)', alpha=0.7)
        plt.xlabel('Cell Cycle Position I')
        plt.ylabel('Progression Rate ω(I) [h⁻¹]')
        plt.title('Progression Rate ω(I)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('docs/images/progression_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_draq5_vs_fI(self, draq5, I_values, f_I, I_range, phases=None):
        """
        Plot DRAQ5 histogram and f(I) side by side, with phase boundaries if available.
        """
        import seaborn as sns
        import numpy as np
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        # DRAQ5 histogram
        axs[0].hist(draq5, bins=50, color='skyblue', alpha=0.7, label='DRAQ5')
        axs[0].set_xlabel('DRAQ5 (DNA content)')
        axs[0].set_ylabel('Cell count')
        axs[0].set_title('DRAQ5 Histogram')
        # f(I) plot
        axs[1].plot(I_range, f_I, color='orange', lw=2, label='f(I)')
        axs[1].set_xlabel('Cell cycle position I')
        axs[1].set_ylabel('Density')
        axs[1].set_title('f(I) (ERA)')
        # Optionally add phase boundaries
        if phases is not None:
            phases = np.array(phases)
            for phase, color in zip(['G1', 'S', 'G2/M'], ['#1f77b4', '#2ca02c', '#d62728']):
                mask = (phases == phase)
                if np.any(mask):
                    axs[0].hist(draq5[mask], bins=50, alpha=0.3, label=phase, color=color)
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        plt.savefig('docs/images/draq5_vs_fI.png', dpi=300)
        plt.close()

    def plot_draq5_to_I_mapping(self, draq5, I_values):
        """
        Plot the mapping from DRAQ5 to I for each cell.
        """
        plt.figure(figsize=(7, 5))
        plt.scatter(draq5, I_values, s=8, alpha=0.5, c=I_values, cmap='viridis')
        plt.xlabel('DRAQ5 (DNA content)')
        plt.ylabel('Cell cycle position I')
        plt.title('Mapping from DRAQ5 to I')
        plt.colorbar(label='I')
        plt.tight_layout()
        plt.savefig('docs/images/draq5_to_I_mapping.png', dpi=300)
        plt.close()

    def plot_fI_bandwidths(self, I_values, I_range, bandwidths=[0.01, 0.05, 0.1, 0.2]):
        """
        Plot f(I) for several KDE bandwidths to visualize smoothing effect.
        """
        from sklearn.neighbors import KernelDensity
        plt.figure(figsize=(8, 6))
        for bw in bandwidths:
            kde = KernelDensity(kernel='gaussian', bandwidth=bw)
            kde.fit(I_values[:, None])
            log_density = kde.score_samples(I_range[:, None])
            fI = np.exp(log_density)
            plt.plot(I_range, fI, label=f'bandwidth={bw}')
        plt.xlabel('Cell cycle position I')
        plt.ylabel('Density f(I)')
        plt.title('Effect of KDE Bandwidth on f(I)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('docs/images/fI_bandwidths.png', dpi=300)
        plt.close()

    def project_to_trajectory(self, gfp, draq5):
        """
        Project each (GFP, DRAQ5) point to the closest point on the parametric trajectory and return the corresponding I value.
        """
        # Assume the parametric trajectory is defined as (gfp_traj(I), draq5_traj(I)) for I in [0, 1]
        I_range = np.linspace(0, 1, 500)
        gfp_traj, draq5_traj = self.parametric_trajectory(I_range)
        points = np.column_stack([gfp, draq5])
        traj_points = np.column_stack([gfp_traj, draq5_traj])
        I_values = np.zeros(len(gfp))
        for idx, pt in enumerate(points):
            dists = np.linalg.norm(traj_points - pt, axis=1)
            I_values[idx] = I_range[np.argmin(dists)]
        return I_values

    def parametric_trajectory(self, I_range):
        """
        Return the parametric trajectory (gfp_traj, draq5_traj) for given I_range.
        This should match the model used to generate synthetic data.
        """
        # Example: piecewise linear model matching synthetic generator
        gfp_traj = np.zeros_like(I_range)
        draq5_traj = np.zeros_like(I_range)
        # G1: I in [0, 0.4)
        mask_g1 = (I_range < 0.4)
        gfp_traj[mask_g1] = 0.5 + 0.5 * I_range[mask_g1] / 0.4
        draq5_traj[mask_g1] = 2.0 + 0.0 * I_range[mask_g1]
        # S: I in [0.4, 0.7)
        mask_s = (I_range >= 0.4) & (I_range < 0.7)
        gfp_traj[mask_s] = 1.0 + 0.5 * (I_range[mask_s] - 0.4) / 0.3
        draq5_traj[mask_s] = 2.0 + 2.0 * (I_range[mask_s] - 0.4) / 0.3
        # G2/M: I in [0.7, 1.0]
        mask_g2m = (I_range >= 0.7)
        gfp_traj[mask_g2m] = 1.5 - 1.0 * (I_range[mask_g2m] - 0.7) / 0.3
        draq5_traj[mask_g2m] = 4.0 - 2.0 * (I_range[mask_g2m] - 0.7) / 0.3
        return gfp_traj, draq5_traj

def main():
    # Load the initial population data
    data = np.load('tests/output/initial_population.npz')
    gfp = data['gfp']
    draq5 = data['draq5']
    phases = data['phases']
    
    # Create analyzer
    analyzer = ErgodicRateAnalysis(draq5, gfp, phases)
    
    # Calculate progression rate
    omega_I, I_range, f_I, F_I, ratio = analyzer.calculate_progression_rate()
    
    # Plot results
    analyzer.plot_era_results()
    
    # Plot periodic density, cumulative, ratio, and progression rate
    analyzer.plot_periodic_density(f_I, I_range)
    analyzer.plot_periodic_cumulative(F_I, I_range)
    analyzer.plot_periodic_ratio(F_I, f_I, I_range)
    analyzer.plot_periodic_progression_rate(omega_I, I_range)
    
    # Plot progression parameters
    analyzer.plot_progression_parameters(omega_I, I_range, f_I, F_I, ratio)
    
    # Calculate I_values for each cell (project to trajectory)
    I_values = analyzer.project_to_trajectory(gfp, draq5)
    # Plot DRAQ5 vs f(I)
    analyzer.plot_draq5_vs_fI(draq5, I_values, f_I, I_range, phases)
    # Plot mapping from DRAQ5 to I
    analyzer.plot_draq5_to_I_mapping(draq5, I_values)
    # Plot f(I) for different bandwidths
    analyzer.plot_fI_bandwidths(I_values, I_range)
    
    print("Analysis complete. Results saved to docs/images/")

if __name__ == "__main__":
    main() 