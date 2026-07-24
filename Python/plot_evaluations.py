"""
Plot evaluation metrics from evaluations.npz file.
Creates a dual-axis graph with timesteps, results (left), and episode lengths (right).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.transforms import Transform
from Mapper import get_file_path


class ExpandedUpperScale(mscale.ScaleBase):
    """Custom scale that expands the upper range (340-370) to take half the plot."""
    name = 'expanded_upper'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis)

    def get_transform(self):
        return self.ExpandedUpperTransform()

    def set_default_locators_and_formatters(self, axis):
        """Set default locators and formatters."""
        axis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(nbins=5))
        axis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())

    class ExpandedUpperTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            a = np.asarray(a, dtype=float)
            result = np.zeros_like(a)
            mask_lower = a < 340
            mask_upper = a >= 340
            # 0-340 maps to 0-0.5 (lower half)
            result[mask_lower] = a[mask_lower] * 0.5 / 340.0
            # 340-370 maps to 0.5-1.0 (upper half)
            result[mask_upper] = 0.5 + (a[mask_upper] - 340.0) * 0.5 / 30.0
            return result

        def inverted(self):
            return self.InvertedExpandedUpperTransform()

        class InvertedExpandedUpperTransform(Transform):
            input_dims = 1
            output_dims = 1

            def transform_non_affine(self, a):
                a = np.asarray(a, dtype=float)
                result = np.zeros_like(a)
                mask_lower = a < 0.5
                mask_upper = a >= 0.5
                # 0-0.5 maps to 0-340
                result[mask_lower] = a[mask_lower] * 340.0 / 0.5
                # 0.5-1.0 maps to 340-370
                result[mask_upper] = 340.0 + (a[mask_upper] - 0.5) * 30.0 / 0.5
                return result

            def inverted(self):
                return ExpandedUpperScale.ExpandedUpperTransform()


# Register the custom scale
mscale.register_scale(ExpandedUpperScale)


def plot_evaluations():
    """Load evaluation data and create dual-subplot plot with smoothing and failure detection."""
    # Load evaluation data using Mapper for proper pathing
    eval_path = get_file_path("runs", "ppo_safe_release_only", "evaluation", "evaluations.npz")
    data = np.load(eval_path)
    
    # Extract data
    timesteps = data["timesteps"] / 1e6  # Convert to millions
    results = data["results"]  # Shape: (417, 10) - [min, max, mean, run1, ..., run7]
    ep_lengths = data["ep_lengths"]  # Shape: (417, 10) - [min, max, mean, run1, ..., run7]
    
    # Use the pre-calculated mean values (index 2) from the 10 columns
    results_mean = results[:, 2]
    ep_lengths_mean = ep_lengths[:, 2]
    
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # --- Plot 1: Results with smoothing and failure detection ---
    # Plot line connecting ALL points (including failures) - thin blue line without markers
    ax1.plot(timesteps, results_mean, color="purple", linewidth=1.5, label="Results")
    
    # Apply smoothing using convolution
    window = 10
    smooth = np.convolve(results_mean, np.ones(window) / window, mode="same")
    ax1.plot(timesteps, smooth, linewidth=3, label="Smoothed (window=10)", color="orange", ls=":")
    
    # Highlight failures (results < 100) with red scatter points
    mask = results_mean < 100
    ax1.scatter(
        timesteps[mask],
        results_mean[mask],
        color="red",
        s=40,
        zorder=3,
        label="Failure (< 100)"
    )
    
    ax1.set_ylabel("Reward", fontsize=12)
    ax1.set_title("Evaluation Metrics over Training Timesteps", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    
    # --- Plot 2: Episode Lengths ---
    ax2.plot(timesteps, ep_lengths_mean, color="red", lw=2, label="Episode Length")
    ax2.set_ylabel("Episode Length", fontsize=12)
    ax2.set_xlabel("Timesteps (1e6)", fontsize=12)
    ax2.set_yscale('expanded_upper')  # Custom scale
    ax2.set_ylim(240, 370)  # Show 0-370, with 340-370 taking upper half
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    
    # Adjust layout and save
    fig.tight_layout()
    
    # Save the figure
    output_path = get_file_path("Python", "evaluation_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    
    # Display the plot
    plt.show()


if __name__ == "__main__":
    plot_evaluations()
