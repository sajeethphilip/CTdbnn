"""
Enhanced Adaptive CT-DBNN Wrapper
=================================

A scientifically rigorous implementation of adaptive learning system that leverages
the enhanced ct_dbnn module for core model operations while preserving sophisticated
adaptive learning logic.

Scientific Foundation:
- Complex Tensor Difference Boosting Bayesian Neural Networks
- Orthogonal Weight Initialization in Complex Space
- Adaptive Sample Selection with Acid Test Validation
- Bayesian Probability Theory for Likelihood Computation
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import json
import os
import time
import glob
from datetime import datetime
from collections import defaultdict
import gc
import sys

# Import the enhanced ct_dbnn module
try:
    import ct_dbnn
except ImportError:
    print("❌ CT-DBNN module not found. Please ensure ct_dbnn.py is in the same directory.")
    exit(1)

# GUI and visualization components
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans


class ModelSerializer:
    """
    Enhanced model serializer that uses ct_dbnn's native binary format
    and ensures proper feature name preservation and label encoding.
    """

    @staticmethod
    def save_model(adaptive_model, filepath: str = None) -> bool:
        """
        Save the complete adaptive model state using ct_dbnn's native binary format.

        Args:
            adaptive_model: The AdaptiveCTDBNN instance to save
            filepath: Path to save the model (uses default if None)

        Returns:
            bool: True if successful
        """
        try:
            # Create default filepath if not provided
            if filepath is None:
                os.makedirs("Models", exist_ok=True)
                dataset_name = adaptive_model.dataset_name or "unknown_dataset"
                filepath = f"Models/{dataset_name}.bin"

            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            print(f"💾 Saving model to: {filepath}")
            print(f"📊 Feature names: {adaptive_model.feature_names}")

            # Use ct_dbnn's native save_model method for the core model
            adaptive_model.model.save_model(filepath)

            # Save adaptive learning state separately in JSON format
            adaptive_state_file = filepath.replace('.bin', '_adaptive_state.json')
            adaptive_state = {
                'dataset_name': adaptive_model.dataset_name,
                'best_accuracy': float(adaptive_model.best_accuracy),
                'best_training_indices': [int(idx) for idx in adaptive_model.best_training_indices],
                'best_round': int(adaptive_model.best_round),
                'adaptive_round': int(adaptive_model.adaptive_round),
                'training_indices': [int(idx) for idx in adaptive_model.training_indices],
                'adaptive_config': adaptive_model.adaptive_config,
                'config': adaptive_model.config,
                'feature_names': adaptive_model.feature_names,
                'target_column': adaptive_model.target_column,
                'selected_features': adaptive_model.selected_features,
                'preprocessor_info': adaptive_model.model.preprocessor.get_feature_info() if hasattr(adaptive_model.model, 'preprocessor') and adaptive_model.model.preprocessor else {},
                'data_shape': {
                    'n_samples': int(adaptive_model.X_full.shape[0]) if adaptive_model.X_full is not None else 0,
                    'n_features': int(adaptive_model.X_full.shape[1]) if adaptive_model.X_full is not None else 0,
                    'n_classes': len(np.unique(adaptive_model.y_full)) if adaptive_model.y_full is not None else 0
                } if adaptive_model.X_full is not None else {},
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'model_type': 'AdaptiveCTDBNN'
                }
            }

            # Save adaptive state as JSON
            with open(adaptive_state_file, 'w') as f:
                json.dump(adaptive_state, f, indent=2, default=str)

            print(f"✅ Model saved successfully to {filepath}")
            print(f"📁 Adaptive state saved to: {adaptive_state_file}")
            print(f"🔧 Feature names preserved: {adaptive_model.feature_names}")
            print(f"🎯 Target column: {adaptive_model.target_column}")
            print(f"🏆 Best accuracy: {adaptive_model.best_accuracy:.4f}")

            return True

        except Exception as e:
            print(f"❌ Error saving model: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return False

    @staticmethod
    def load_model(filepath: str):
        """
        Load a complete adaptive model state from ct_dbnn's binary file.

        Args:
            filepath: Path to the saved model

        Returns:
            tuple: (adaptive_model, success_message) or (None, error_message)
        """
        try:
            if not os.path.exists(filepath):
                return None, f"Model file not found: {filepath}"

            print(f"📂 Loading model from: {filepath}")

            # Load adaptive state first to get dataset name
            adaptive_state_file = filepath.replace('.bin', '_adaptive_state.json')
            if not os.path.exists(adaptive_state_file):
                return None, f"Adaptive state file not found: {adaptive_state_file}"

            with open(adaptive_state_file, 'r') as f:
                adaptive_state = json.load(f)

            # Verify model type
            if adaptive_state.get('metadata', {}).get('model_type') != 'AdaptiveCTDBNN':
                return None, "Invalid model file format"

            # Create new adaptive model
            dataset_name = adaptive_state.get('dataset_name', 'unknown_dataset')
            config = adaptive_state.get('config', {})
            adaptive_model = AdaptiveCTDBNN(dataset_name, config)

            # Load core model using ct_dbnn's native load_model
            adaptive_model.model.load_model(filepath)

            # Restore adaptive state
            adaptive_model.best_accuracy = float(adaptive_state.get('best_accuracy', 0.0))
            adaptive_model.best_training_indices = [int(idx) for idx in adaptive_state.get('best_training_indices', [])]
            adaptive_model.best_round = int(adaptive_state.get('best_round', 0))
            adaptive_model.adaptive_round = int(adaptive_state.get('adaptive_round', 0))
            adaptive_model.training_indices = [int(idx) for idx in adaptive_state.get('training_indices', [])]
            adaptive_model.adaptive_config.update(adaptive_state.get('adaptive_config', {}))
            adaptive_model.feature_names = adaptive_state.get('feature_names', [])
            adaptive_model.target_column = adaptive_state.get('target_column', 'target')
            adaptive_model.selected_features = adaptive_state.get('selected_features', [])

            print(f"✅ Model loaded successfully from {filepath}")
            print(f"📊 Feature names restored: {adaptive_model.feature_names}")
            print(f"🎯 Target column: {adaptive_model.target_column}")
            print(f"🏆 Best accuracy: {adaptive_model.best_accuracy:.4f}")
            print(f"📦 Data shape: {adaptive_state.get('data_shape', {})}")

            return adaptive_model, "Model loaded successfully"

        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(f"❌ {error_msg}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return None, error_msg


class CTDBNNVisualizer:
    """
    Scientifically rigorous visualization system for CT-DBNN.
    Provides comprehensive analysis of complex tensor operations and model behavior.
    """

    def __init__(self, model, output_dir='visualizations'):
        """
        Initialize visualizer with CT-DBNN model.

        Args:
            model: ParallelCTDBNN instance (not AdaptiveCTDBNN wrapper)
            output_dir: Directory to save visualizations
        """
        self.model = model  # This should be the actual ParallelCTDBNN instance
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_visualizations(self, X, y, predictions=None):
        """
        Create comprehensive visualizations including complex tensor analysis.

        Args:
            X: Feature matrix
            y: True labels
            predictions: Model predictions (optional)
        """
        print("📊 Creating enhanced visualizations...")

        try:
            # Ensure directory exists
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"✅ Created visualization directory: {self.output_dir}")

            # Basic statistical visualizations
            self.plot_class_distribution(y)

            if predictions is not None:
                self.plot_confusion_matrix(y, predictions)

            # Complex tensor visualizations (only if available)
            if (hasattr(self.model, 'complex_weights') and
                self.model.complex_weights is not None and
                hasattr(self.model, 'is_trained') and
                self.model.is_trained):

                print("🎨 Creating complex tensor visualizations...")
                self.plot_complex_tensor_orientations()
                self.plot_orthogonalization_comparison()
                self.plot_complex_weight_distributions()

            # Feature interaction analysis
            if (hasattr(self.model, 'global_anti_net') and
                self.model.global_anti_net is not None):
                self.plot_feature_interaction_heatmap()

            # Training history visualization
            if (hasattr(self.model, 'training_history') and
                self.model.training_history):
                self.plot_training_history()

            print("✅ Enhanced visualizations created")

        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")

    def plot_complex_tensor_orientations(self):
        """
        Visualize complex tensor orientations for each target class in 3D space.
        Shows real, imaginary, and phase components with class-specific coloring.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from mpl_toolkits.mplot3d import Axes3D

            if not hasattr(self.model, 'complex_weights') or self.model.complex_weights is None:
                print("⚠️ No complex weights available for visualization")
                return

            complex_weights = self.model.complex_weights
            n_features = self.model.innodes
            resol = self.model.config.get('resol', 8)
            n_classes = self.model.outnodes

            print(f"🎨 Creating complex tensor orientation visualization...")
            print(f"   Features: {n_features}, Resolution: {resol}, Classes: {n_classes}")

            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Complex Tensor Orientations by Class\n(3D Representation: Real vs Imaginary vs Phase)',
                        fontsize=16, fontweight='bold')

            # Select representative feature pairs for visualization
            feature_pairs = []
            max_features_to_show = min(4, n_features)

            for i in range(1, max_features_to_show):
                for j in range(i + 1, min(i + 3, n_features + 1)):
                    if i <= n_features and j <= n_features:
                        feature_pairs.append((i, j))
                    if len(feature_pairs) >= 4:  # Limit to 4 subplots
                        break
                if len(feature_pairs) >= 4:
                    break

            # Ensure we have at least one pair
            if not feature_pairs and n_features >= 2:
                feature_pairs = [(1, 2)]

            colors = list(mcolors.TABLEAU_COLORS.values())

            for idx, (f1, f2) in enumerate(feature_pairs[:4]):  # Show max 4 pairs
                ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

                # Track plotted classes for legend
                plotted_classes = set()

                # Sample bins for visualization
                bin_samples = min(3, resol)
                bins_to_sample = np.linspace(1, resol, bin_samples, dtype=int)

                for bin1 in bins_to_sample:
                    for bin2 in bins_to_sample:
                        for class_idx in range(1, n_classes + 1):
                            # Check bounds
                            if (f1 < complex_weights.shape[0] and f2 < complex_weights.shape[2] and
                                bin1 < complex_weights.shape[1] and bin2 < complex_weights.shape[3] and
                                class_idx < complex_weights.shape[4]):

                                weight_complex = complex_weights[f1, bin1, f2, bin2, class_idx]

                                # Convert to 3D representation
                                real = weight_complex.real
                                imag = weight_complex.imag
                                phase = np.angle(weight_complex)
                                magnitude = np.abs(weight_complex)

                                # Scale by magnitude for visibility
                                scale = 0.5 + 0.5 * magnitude
                                real *= scale
                                imag *= scale

                                # Plot with class-specific color
                                color = colors[(class_idx - 1) % len(colors)]
                                label = f'Class {class_idx}' if class_idx not in plotted_classes else ""

                                ax.quiver(0, 0, 0, real, imag, phase,
                                         color=color, alpha=0.6,
                                         label=label,
                                         arrow_length_ratio=0.1,
                                         linewidth=1.5 * magnitude)

                                # Add a point at the vector end
                                ax.scatter([real], [imag], [phase],
                                          color=color, s=20 * magnitude, alpha=0.8)

                                plotted_classes.add(class_idx)

                ax.set_xlabel('Real Component')
                ax.set_ylabel('Imaginary Component')
                ax.set_zlabel('Phase (radians)')
                ax.set_title(f'Feature {f1} vs Feature {f2}\nComplex Weight Orientations')

                if plotted_classes:
                    ax.legend()

            plt.tight_layout()
            output_path = f'{self.output_dir}/complex_tensor_orientations.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Complex tensor orientation plot saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting complex tensor orientations: {e}")

    def plot_orthogonalization_comparison(self):
        """
        Compare theoretical orthogonal vectors with actual complex weights.
        Demonstrates the effectiveness of orthogonal weight initialization.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            if not hasattr(self.model, 'complex_weights') or self.model.complex_weights is None:
                print("⚠️ No complex weights available for orthogonalization comparison")
                return

            n_classes = self.model.outnodes

            print("🔄 Creating orthogonalization comparison visualization...")

            # Theoretical orthogonal phases
            theoretical_phases = np.array([(k-1) * 2 * np.pi / n_classes
                                         for k in range(1, n_classes + 1)])
            theoretical_vectors = np.exp(1j * theoretical_phases)

            # Create comparison plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Complex Weight Orthogonalization Analysis', fontsize=16, fontweight='bold')

            colors = list(mcolors.TABLEAU_COLORS.values())

            # Plot 1: Theoretical orthogonal vectors (Unit Circle)
            theta = np.linspace(0, 2*np.pi, 100)
            ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Unit Circle')

            for class_idx in range(n_classes):
                vec = theoretical_vectors[class_idx]
                color = colors[class_idx % len(colors)]

                ax1.quiver(0, 0, vec.real, vec.imag,
                          color=color, alpha=0.8,
                          label=f'Class {class_idx+1}',
                          angles='xy', scale_units='xy', scale=1,
                          width=0.015)
                ax1.scatter([vec.real], [vec.imag], color=color, s=100, alpha=0.8)

            ax1.set_xlabel('Real Component')
            ax1.set_ylabel('Imaginary Component')
            ax1.set_title('Theoretical Orthogonal Vectors\n(Perfect Orthogonalization)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')

            # Plot 2: Actual complex weights (sampled)
            if hasattr(self.model, 'complex_weights'):
                complex_weights = self.model.complex_weights

                # Sample some weights from the tensor
                sample_indices = []
                max_samples_per_class = 10

                for class_idx in range(1, n_classes + 1):
                    # Find non-zero weights for this class
                    class_weights = []

                    # Sample from different feature pairs and bins
                    for f1 in range(1, min(4, complex_weights.shape[0])):
                        for f2 in range(1, min(4, complex_weights.shape[2])):
                            for b1 in range(1, min(4, complex_weights.shape[1])):
                                for b2 in range(1, min(4, complex_weights.shape[3])):
                                    if (f1 < complex_weights.shape[0] and f2 < complex_weights.shape[2] and
                                        b1 < complex_weights.shape[1] and b2 < complex_weights.shape[3] and
                                        class_idx < complex_weights.shape[4]):

                                        weight = complex_weights[f1, b1, f2, b2, class_idx]
                                        if np.abs(weight) > 0.1:  # Filter very small weights
                                            class_weights.append(weight)

                    # Take a sample for visualization
                    if class_weights:
                        sample_size = min(max_samples_per_class, len(class_weights))
                        samples = np.random.choice(class_weights, sample_size, replace=False)

                        for weight in samples:
                            color = colors[(class_idx-1) % len(colors)]
                            ax2.quiver(0, 0, weight.real, weight.imag,
                                      color=color, alpha=0.6,
                                      angles='xy', scale_units='xy', scale=1,
                                      width=0.008)
                            ax2.scatter([weight.real], [weight.imag],
                                       color=color, s=30, alpha=0.7)

            ax2.set_xlabel('Real Component')
            ax2.set_ylabel('Imaginary Component')
            ax2.set_title('Actual Complex Weights\n(Sampled from Tensor)')
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')

            plt.tight_layout()
            output_path = f'{self.output_dir}/orthogonalization_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Orthogonalization comparison plot saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting orthogonalization comparison: {e}")

    def plot_complex_weight_distributions(self):
        """Plot distributions of complex weight components by class."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            if not hasattr(self.model, 'complex_weights') or self.model.complex_weights is None:
                return

            complex_weights = self.model.complex_weights
            n_classes = self.model.outnodes

            print("📈 Creating complex weight distribution visualization...")

            # Extract weight components with efficient sampling
            sample_size = min(2000, complex_weights.size // 10)  # Sample 10% or 2000, whichever smaller
            flat_weights = complex_weights.flatten()

            if len(flat_weights) > sample_size:
                sample_indices = np.random.choice(len(flat_weights), sample_size, replace=False)
                sampled_weights = flat_weights[sample_indices]
            else:
                sampled_weights = flat_weights

            # Convert to components
            real_parts = sampled_weights.real
            imag_parts = sampled_weights.imag
            magnitudes = np.abs(sampled_weights)
            phases = np.angle(sampled_weights)

            # Create distribution plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Complex Weight Component Distributions', fontsize=16, fontweight='bold')

            colors = list(mcolors.TABLEAU_COLORS.values())

            # Plot 1: Real parts distribution
            ax1.hist(real_parts, bins=50, alpha=0.7, color='blue', density=True)
            ax1.set_xlabel('Real Component Value')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Distribution of Real Components')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Imaginary parts distribution
            ax2.hist(imag_parts, bins=50, alpha=0.7, color='red', density=True)
            ax2.set_xlabel('Imaginary Component Value')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Distribution of Imaginary Components')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Magnitude distribution
            ax3.hist(magnitudes, bins=50, alpha=0.7, color='green', density=True)
            ax3.set_xlabel('Magnitude')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('Distribution of Magnitudes')
            ax3.grid(True, alpha=0.3)

            # Plot 4: Phase distribution
            ax4.hist(phases, bins=50, alpha=0.7, color='purple', density=True)
            ax4.set_xlabel('Phase (radians)')
            ax4.set_ylabel('Probability Density')
            ax4.set_title('Distribution of Phases')
            ax4.grid(True, alpha=0.3)

            # Add statistical information
            stats_text = f"""
            Statistical Summary:
            Real: μ={np.mean(real_parts):.3f}, σ={np.std(real_parts):.3f}
            Imag: μ={np.mean(imag_parts):.3f}, σ={np.std(imag_parts):.3f}
            Mag:  μ={np.mean(magnitudes):.3f}, σ={np.std(magnitudes):.3f}
            Phase: μ={np.mean(phases):.3f}, σ={np.std(phases):.3f}
            """
            fig.text(0.02, 0.02, stats_text, fontfamily='monospace', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            plt.tight_layout()
            output_path = f'{self.output_dir}/complex_weight_distributions.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Complex weight distribution plot saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting complex weight distributions: {e}")

    def plot_feature_interaction_heatmap(self):
        """Plot heatmap of feature interactions from global_anti_net."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            if not hasattr(self.model, 'global_anti_net') or self.model.global_anti_net is None:
                return

            global_anti_net = self.model.global_anti_net
            n_features = self.model.innodes

            print("🔥 Creating feature interaction heatmap...")

            # Create feature interaction matrix
            interaction_matrix = np.zeros((n_features, n_features))

            for i in range(n_features):
                for j in range(n_features):
                    # Sum interactions across all bins and classes
                    feature_i, feature_j = i + 1, j + 1
                    if (feature_i < global_anti_net.shape[0] and feature_j < global_anti_net.shape[2]):
                        # Sum over bins and classes, exclude padding
                        interaction = np.sum(global_anti_net[feature_i, 1:-1, feature_j, 1:-1, 1:-1])
                        interaction_matrix[i, j] = interaction

            # Normalize and apply log scale for better visualization
            interaction_matrix = np.log1p(interaction_matrix)  # log(1+x) to handle zeros

            if np.max(interaction_matrix) > 0:
                interaction_matrix = interaction_matrix / np.max(interaction_matrix)

            # Plot heatmap
            plt.figure(figsize=(12, 10))

            feature_names = self.model.feature_names if hasattr(self.model, 'feature_names') and self.model.feature_names else [f'F{i+1}' for i in range(n_features)]

            sns.heatmap(interaction_matrix,
                       xticklabels=feature_names,
                       yticklabels=feature_names,
                       cmap='viridis',
                       square=True,
                       cbar_kws={'label': 'Normalized Log Interaction Strength'})

            plt.title('Feature Interaction Heatmap\n(Log-scaled Global Likelihood Network)')
            plt.xlabel('Feature j')
            plt.ylabel('Feature i')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)

            plt.tight_layout()
            output_path = f'{self.output_dir}/feature_interaction_heatmap.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Feature interaction heatmap saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting feature interaction heatmap: {e}")

    def plot_training_history(self):
        """Plot training history and metrics."""
        try:
            import matplotlib.pyplot as plt

            if not hasattr(self.model, 'training_history') or not self.model.training_history:
                return

            history = self.model.training_history

            print("📈 Creating training history visualization...")

            plt.figure(figsize=(12, 8))

            # Plot available metrics
            metrics_to_plot = []
            if 'train_accuracy' in history:
                metrics_to_plot.append(('train_accuracy', 'Training Accuracy', 'blue'))
            if 'test_accuracy' in history:
                metrics_to_plot.append(('test_accuracy', 'Test Accuracy', 'red'))
            if 'training_time' in history:
                metrics_to_plot.append(('training_time', 'Training Time (s)', 'green'))

            for i, (metric, label, color) in enumerate(metrics_to_plot):
                plt.subplot(2, 2, i + 1)
                if isinstance(history[metric], (list, np.ndarray)):
                    plt.plot(history[metric], color=color, linewidth=2, label=label)
                    plt.xlabel('Epoch/Round')
                else:
                    plt.bar([0], [history[metric]], color=color, alpha=0.7, label=label)
                    plt.xticks([])

                plt.ylabel(label)
                plt.title(f'{label} Evolution')
                plt.legend()
                plt.grid(True, alpha=0.3)

            # Add configuration summary
            if 'config' in history:
                config_text = "Model Configuration:\n"
                for key, value in list(history['config'].items())[:6]:  # Show first 6 configs
                    config_text += f"{key}: {value}\n"

                plt.figtext(0.02, 0.02, config_text, fontfamily='monospace', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            plt.tight_layout()
            output_path = f'{self.output_dir}/training_history.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Training history plot saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting training history: {e}")

    def plot_class_distribution(self, y):
        """Plot class distribution with enhanced styling."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors

            plt.figure(figsize=(12, 6))

            unique_classes, counts = np.unique(y, return_counts=True)
            colors = list(mcolors.TABLEAU_COLORS.values())

            bars = plt.bar(range(len(unique_classes)), counts,
                          alpha=0.7,
                          color=colors[:len(unique_classes)])

            plt.title('Class Distribution\n(Data Distribution Across Classes)')
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            plt.xticks(range(len(unique_classes)), [str(cls) for cls in unique_classes])
            plt.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')

            # Add statistical summary
            total_samples = np.sum(counts)
            plt.figtext(0.02, 0.02, f'Total Samples: {total_samples}\nClasses: {len(unique_classes)}',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            plt.tight_layout()
            output_path = f'{self.output_dir}/class_distribution.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Class distribution plot saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting class distribution: {e}")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix with enhanced styling."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(10, 8))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.unique(np.concatenate([y_true, y_pred])),
                       yticklabels=np.unique(np.concatenate([y_true, y_pred])),
                       cbar_kws={'label': 'Count'})

            plt.title('Confusion Matrix\n(Model Prediction Performance)')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Calculate and display accuracy
            accuracy = accuracy_score(y_true, y_pred)
            plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.4f}',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))

            plt.tight_layout()
            output_path = f'{self.output_dir}/confusion_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Confusion matrix plot saved: {output_path}")

        except Exception as e:
            print(f"⚠️ Error plotting confusion matrix: {e}")


class AdaptiveCTDBNN:
    """
    Scientifically rigorous Adaptive CT-DBNN implementation.

    Implements adaptive learning with acid test validation while maintaining
    mathematical consistency with the original CT-DBNN formulation.
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        """
        Initialize the adaptive learning wrapper with scientific rigor.

        Args:
            dataset_name: Name of the dataset for tracking
            config: Configuration parameters for adaptive learning
        """
        self.dataset_name = dataset_name
        self.config = config or {}

        # Enhanced adaptive learning configuration
        self.adaptive_config = self.config.get('adaptive_learning', {})

        # Scientific default configuration
        default_config = {
            "enable_adaptive": True,
            "initial_samples_per_class": 5,
            "max_adaptive_rounds": 20,
            "patience": 10,
            "min_improvement": 0.001,
            "enable_acid_test": True,
            "divergence_threshold": 0.1,
            "max_samples_per_round": 2,
            "sample_selection_strategy": "margin",
            "class_balancing": True,
            "hard_sample_mining": True,
            "confidence_threshold": 0.8,
            "diversity_weight": 0.3,
            "uncertainty_weight": 0.7,
            "exploration_factor": 0.1,
        }
        for key, default_value in default_config.items():
            if key not in self.adaptive_config:
                self.adaptive_config[key] = default_value

        # Initialize the core CT-DBNN model with scientific parameters
        ctdbnn_config = self.config.get('ctdbnn_config', {})

        # Scientifically validated default CT-DBNN configuration
        default_ctdbnn_config = {
            'resol': 8,
            'use_complex_tensor': True,
            'orthogonalize_weights': True,
            'parallel_processing': True,
            'smoothing_factor': 1e-8,
            'n_jobs': -1,
            'batch_size': 1000,
            'missing_value_placeholder': -99999,
        }

        # Merge with provided config
        for key, value in default_ctdbnn_config.items():
            if key not in ctdbnn_config:
                ctdbnn_config[key] = value

        # Initialize the core ParallelCTDBNN model
        self.model = ct_dbnn.ParallelCTDBNN(ctdbnn_config)

        # Initialize visualizer with the core model (not self)
        self.visualizer = CTDBNNVisualizer(self.model)

        # Adaptive learning state tracking
        self.training_indices = []
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.best_round = 0
        self.adaptive_round = 0
        self.patience_counter = 0

        # Scientific tracking and statistics
        self.round_stats = []
        self.start_time = datetime.now()
        self.adaptive_start_time = None

        # Data storage with scientific validation
        self.X_full = None
        self.y_full = None
        self.feature_names = None
        self.target_column = 'target'
        self.selected_features = None
        self.original_data = None

        # Sample tracking for scientific analysis
        self.all_selected_samples = defaultdict(list)
        self.sample_selection_history = []

    def load_and_preprocess_data(self, file_path: str = None, target_column: str = None, selected_features: List[str] = None) -> bool:
        """
        Load and preprocess data with scientific rigor and proper encoding.

        Args:
            file_path: Path to data file
            target_column: Name of the target column
            selected_features: List of feature columns to use

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("📥 Loading and preprocessing data with scientific validation...")

            # Use ct_dbnn's UCI dataset loader if it's a known dataset
            if self.dataset_name and self.dataset_name in ct_dbnn.UCI_DATASETS:
                print(f"🎯 Loading UCI dataset: {self.dataset_name}")
                dataset_info = ct_dbnn.UCI_DATASETS[self.dataset_name]
                df = ct_dbnn.UCIDatasetLoader.download_uci_data(dataset_info)

                if df is not None and not df.empty:
                    self.original_data = df

                    # Scientific target column determination
                    if target_column and target_column in df.columns:
                        self.target_column = target_column
                    else:
                        # Use last column as target for UCI datasets (scientific convention)
                        self.target_column = df.columns[-1]
                        print(f"🎯 Using last column as target (UCI convention): {self.target_column}")

                    # Scientific feature selection
                    if selected_features:
                        self.selected_features = selected_features
                        if self.target_column not in selected_features:
                            features_to_use = selected_features + [self.target_column]
                        else:
                            features_to_use = selected_features
                    else:
                        # Use all features except target (scientific default)
                        self.selected_features = [col for col in df.columns if col != self.target_column]
                        features_to_use = df.columns.tolist()

                    # Filter data scientifically
                    df = df[features_to_use]

                    # CRITICAL: Use ct_dbnn's preprocessor for mathematical consistency
                    self.model.preprocessor = ct_dbnn.DataPreprocessor()

                    # Scientific preprocessing with proper encoding
                    features_processed = self.model.preprocessor.fit_transform_features(
                        df.drop(columns=[self.target_column]),
                        self.selected_features
                    )
                    targets_encoded = self.model.preprocessor.fit_transform_targets(df[self.target_column])

                    self.X_full = features_processed
                    self.y_full = targets_encoded
                    self.feature_names = self.model.preprocessor.get_feature_names()

                    print(f"✅ UCI dataset preprocessed with scientific encoding")
                else:
                    print(f"❌ Failed to download UCI dataset: {self.dataset_name}")
                    return False
            else:
                # Load from file using scientific pandas approach
                if file_path is None:
                    # Scientific file discovery
                    possible_files = [
                        f"{self.dataset_name}.csv" if self.dataset_name else "data.csv",
                        f"{self.dataset_name}.data",
                        "data.csv", "train.csv"
                    ]
                    for file in possible_files:
                        if os.path.exists(file):
                            file_path = file
                            print(f"📁 Found data file: {file_path}")
                            break

                if file_path and os.path.exists(file_path):
                    print(f"📁 Loading data from: {file_path}")
                    try:
                        # Scientific file reading with error handling
                        try:
                            df = pd.read_csv(file_path)
                            has_header = True
                        except:
                            df = pd.read_csv(file_path, header=None)
                            has_header = False
                            n_cols = df.shape[1]
                            df.columns = [f'feature_{i}' for i in range(n_cols)]
                            print(f"📝 No header found, using scientific naming: {df.columns.tolist()}")

                        self.original_data = df

                        if df.empty:
                            print("❌ Data file is empty")
                            return False

                        # Scientific target column determination
                        if target_column:
                            if target_column in df.columns:
                                self.target_column = target_column
                            else:
                                print(f"❌ Target column '{target_column}' not found")
                                return False
                        else:
                            # Scientific target auto-detection
                            target_candidates = ['target', 'class', 'label', 'outcome', 'diagnosis', 'type', 'species']
                            for candidate in target_candidates + [df.columns[-1]]:
                                if candidate in df.columns:
                                    self.target_column = candidate
                                    print(f"🎯 Auto-detected target column: {self.target_column}")
                                    break
                            else:
                                self.target_column = df.columns[-1]
                                print(f"🎯 Using last column as target: {self.target_column}")

                        # Scientific feature selection
                        if selected_features:
                            missing_features = [f for f in selected_features if f not in df.columns]
                            if missing_features:
                                print(f"❌ Selected features not found: {missing_features}")
                                return False

                            self.selected_features = selected_features
                            if self.target_column not in selected_features:
                                features_to_use = selected_features + [self.target_column]
                            else:
                                features_to_use = selected_features
                        else:
                            self.selected_features = [col for col in df.columns if col != self.target_column]
                            features_to_use = df.columns.tolist()

                        if not self.selected_features:
                            print("❌ No features selected for training")
                            return False

                        # Scientific data filtering
                        df = df[features_to_use]

                        # CRITICAL: Use ct_dbnn's preprocessor
                        self.model.preprocessor = ct_dbnn.DataPreprocessor()

                        # Scientific preprocessing
                        features_processed = self.model.preprocessor.fit_transform_features(
                            df.drop(columns=[self.target_column]),
                            self.selected_features
                        )
                        targets_encoded = self.model.preprocessor.fit_transform_targets(df[self.target_column])

                        self.X_full = features_processed
                        self.y_full = targets_encoded
                        self.feature_names = self.model.preprocessor.get_feature_names()

                    except Exception as e:
                        print(f"❌ Error reading data file: {e}")
                        return False
                else:
                    print("❌ No data file found and no UCI dataset specified")
                    return False

            # Scientific data validation
            if self.X_full is None or self.y_full is None or len(self.X_full) == 0:
                print("❌ Failed to load data - no samples found")
                return False

            print(f"✅ Data loaded successfully: {self.X_full.shape[0]} samples, {self.X_full.shape[1]} features")
            print(f"🎯 Target column: {self.target_column}")
            print(f"📊 Feature names: {self.feature_names}")
            print(f"🎯 Classes: {np.unique(self.y_full)}")

            # Scientific encoding verification
            if hasattr(self.model.preprocessor, 'feature_encoders'):
                print("🔧 Feature encoding summary (scientific preprocessing):")
                for feature, encoder in self.model.preprocessor.feature_encoders.items():
                    if encoder != 'numeric':
                        print(f"   {feature}: {len(encoder)} categories")
                    else:
                        print(f"   {feature}: numeric (continuous)")

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return False

    def get_data_columns(self) -> List[str]:
        """Get all available columns from the loaded data."""
        if self.original_data is not None:
            return self.original_data.columns.tolist()
        return []

    def get_numeric_columns(self) -> List[str]:
        """Get numeric columns from the loaded data."""
        if self.original_data is not None:
            numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns.tolist()
            return numeric_cols
        return []

    def get_categorical_columns(self) -> List[str]:
        """Get categorical columns from the loaded data."""
        if self.original_data is not None:
            categorical_cols = self.original_data.select_dtypes(include=['object', 'category']).columns.tolist()
            return categorical_cols
        return []

    def initialize_model(self):
        """
        Initialize the CT-DBNN model with full dataset architecture.
        This computes global likelihoods and prepares for adaptive training.
        """
        if self.X_full is None:
            raise ValueError("No data available. Call load_and_preprocess_data() first.")

        print("🏗️ Initializing CT-DBNN architecture with scientific rigor...")

        # Use ct_dbnn's compute_global_likelihoods for mathematical consistency
        normalized_features = self.model.compute_global_likelihoods(
            self.X_full,
            self.y_full,
            self.feature_names
        )

        print("✅ Model architecture initialized with global likelihood computation")

    def adaptive_learn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main adaptive learning algorithm with scientific acid test validation.

        Implements the core adaptive learning logic with mathematical rigor:
        1. Start with diverse initial samples using k-means++ clustering
        2. Train model and run acid test on entire dataset
        3. Select most divergent misclassified samples using Bayesian criteria
        4. Add them to training set and repeat
        5. Stop when no improvement or maximum rounds reached

        Returns:
            Tuple: (X_train, y_train, X_test, y_test) - Best training/test split found
        """
        print("\n🚀 STARTING SCIENTIFIC ADAPTIVE LEARNING WITH CT-DBNN")
        print("=" * 60)

        if self.X_full is None:
            success = self.load_and_preprocess_data()
            if not success:
                raise ValueError("Failed to load data")

        X, y = self.X_full, self.y_full

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes: {np.unique(y)}")
        print(f"📊 Feature names: {self.feature_names}")
        print(f"🎯 Target column: {self.target_column}")

        # Scientific model initialization
        if not hasattr(self.model, 'likelihoods_computed') or not self.model.likelihoods_computed:
            self.initialize_model()

        # STEP 1: Select initial diverse training samples scientifically
        if hasattr(self, 'best_training_indices') and self.best_training_indices:
            print(f"📚 Using existing best training set with {len(self.best_training_indices)} samples")
            initial_indices = self.best_training_indices.copy()
        else:
            X_train, y_train, initial_indices = self._select_initial_training_samples(X, y)
            print(f"🎯 Selected new initial training set: {len(X_train)} samples")

        remaining_indices = [i for i in range(len(X)) if i not in initial_indices]

        print(f"📊 Initial training set: {len(initial_indices)} samples")
        print(f"📊 Remaining pool: {len(remaining_indices)} samples")

        # Scientific tracking initialization
        self.best_accuracy = 0.0
        self.best_training_indices = initial_indices.copy()
        self.best_round = 0
        acid_test_history = []
        patience_counter = 0

        max_rounds = self.adaptive_config['max_adaptive_rounds']
        patience = self.adaptive_config['patience']
        min_improvement = self.adaptive_config['min_improvement']

        print(f"\n🔄 Starting adaptive learning for up to {max_rounds} rounds...")
        self.adaptive_start_time = datetime.now()

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num

            print(f"\n🎯 Round {round_num}/{max_rounds}")
            print("-" * 40)

            # Get current training data
            X_train = X[initial_indices]
            y_train = y[initial_indices]

            # STEP 2: Train model with current training data
            print("🎯 Training model with current samples...")
            training_time = self.model.train(X_train, y_train)
            print(f"✅ Training completed in {training_time:.3f}s")

            # STEP 3: Run acid test on ENTIRE dataset (scientific validation)
            print("🧪 Running acid test on entire dataset...")
            try:
                all_predictions = self.model.predict(X)
                acid_test_accuracy = accuracy_score(y, all_predictions)
                acid_test_history.append(acid_test_accuracy)
                print(f"📊 Acid test accuracy: {acid_test_accuracy:.4f}")

                # Scientific stopping criterion: 100% accuracy
                if acid_test_accuracy >= 0.9999:
                    print("🎉 REACHED 100% ACCURACY! Stopping adaptive learning.")
                    self.best_accuracy = acid_test_accuracy
                    self.best_training_indices = initial_indices.copy()
                    self.best_round = round_num
                    break

            except Exception as e:
                print(f"❌ Acid test failed: {e}")
                acid_test_accuracy = 0.0
                acid_test_history.append(0.0)

            # STEP 4: Check if we have remaining samples
            if not remaining_indices:
                print("💤 No more samples to add")
                if acid_test_accuracy > self.best_accuracy:
                    self.best_accuracy = acid_test_accuracy
                    self.best_training_indices = initial_indices.copy()
                    self.best_round = round_num
                break

            # STEP 5: Identify failed candidates scientifically
            X_remaining = X[remaining_indices]
            y_remaining = y[remaining_indices]

            remaining_predictions = self.model.predict(X_remaining)
            remaining_probs = self.model.predict_proba(X_remaining)

            # Find misclassified samples using Bayesian criteria
            misclassified_mask = remaining_predictions != y_remaining
            misclassified_indices = np.where(misclassified_mask)[0]

            if len(misclassified_indices) == 0:
                print("✅ No misclassified samples in remaining data!")
            else:
                print(f"📊 Found {len(misclassified_indices)} misclassified samples")

                # STEP 6: Select most divergent failed candidates scientifically
                samples_to_add_indices = self._select_divergent_samples(
                    X_remaining, y_remaining, remaining_predictions, remaining_probs,
                    misclassified_indices, remaining_indices
                )

                if samples_to_add_indices:
                    initial_indices.extend(samples_to_add_indices)
                    remaining_indices = [i for i in remaining_indices if i not in samples_to_add_indices]

                    print(f"📈 Training set size: {len(initial_indices)} samples "
                          f"({len(initial_indices)/len(X)*100:.1f}% of total)")
                    print(f"📊 Remaining pool: {len(remaining_indices)} samples")
                else:
                    print("💤 No divergent samples to add in this round")

            # STEP 7: Update best model and check for scientific improvement
            if acid_test_accuracy > self.best_accuracy + min_improvement:
                improvement = acid_test_accuracy - self.best_accuracy
                self.best_accuracy = acid_test_accuracy
                self.best_training_indices = initial_indices.copy()
                self.best_round = round_num
                patience_counter = 0
                print(f"🏆 New best accuracy: {acid_test_accuracy:.4f} (+{improvement:.4f})")
            else:
                patience_counter += 1
                if acid_test_accuracy > self.best_accuracy:
                    small_improvement = acid_test_accuracy - self.best_accuracy
                    print(f"↗️  Small improvement: {acid_test_accuracy:.4f} (+{small_improvement:.4f})")
                else:
                    print(f"🔄 No improvement - Patience: {patience_counter}/{patience}")

            # Scientific stopping criteria
            if patience_counter >= patience:
                print(f"🛑 PATIENCE EXCEEDED: No improvement for {patience} rounds")
                break

            if round_num >= max_rounds:
                print(f"🛑 MAXIMUM ROUNDS REACHED: {max_rounds} rounds")
                break

        # Finalize with best scientific configuration
        print(f"\n🎉 Adaptive learning completed after {self.adaptive_round} rounds!")
        print(f"🏆 Best acid test accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
        print(f"📊 Final training set: {len(self.best_training_indices)} samples "
              f"({len(self.best_training_indices)/len(X)*100:.1f}% of total)")

        # Train final model with best configuration scientifically
        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        self.model.train(X_train_best, y_train_best)

        # Final scientific evaluation
        final_predictions = self.model.predict(X)
        final_accuracy = accuracy_score(y, final_predictions)
        print(f"📊 Final acid test accuracy: {final_accuracy:.4f}")

        # Create comprehensive scientific visualizations
        print("🎨 Creating scientific visualizations...")
        try:
            self.visualizer.create_visualizations(X, y, final_predictions)
            print("✅ Scientific visualizations created successfully")
        except Exception as e:
            print(f"⚠️ Error during visualization: {e}")

        # Create test set from remaining samples scientifically
        test_indices = [i for i in range(len(X)) if i not in self.best_training_indices]
        X_test_best = X[test_indices] if test_indices else np.array([])
        y_test_best = y[test_indices] if test_indices else np.array([])

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _select_initial_training_samples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Select initial diverse training samples using k-means clustering scientifically.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple: (X_train, y_train, selected_indices)
        """
        initial_samples = self.adaptive_config['initial_samples_per_class']
        unique_classes = np.unique(y)

        initial_indices = []

        print("🎯 Selecting initial diverse training samples using k-means++...")

        for class_id in unique_classes:
            class_indices = np.where(y == class_id)[0]

            if len(class_indices) > initial_samples:
                # Use k-means++ for scientific diversity sampling
                class_data = X[class_indices]
                kmeans = KMeans(n_clusters=initial_samples, init='k-means++', n_init=1, random_state=42)
                kmeans.fit(class_data)

                # Find samples closest to cluster centers scientifically
                distances = kmeans.transform(class_data)
                closest_indices = np.argmin(distances, axis=0)
                selected_indices = class_indices[closest_indices]
            else:
                # Use all available samples
                selected_indices = class_indices

            initial_indices.extend(selected_indices)
            print(f"   Class {class_id}: Selected {len(selected_indices)} samples")

        X_train = X[initial_indices]
        y_train = y[initial_indices]

        return X_train, y_train, initial_indices

    def _select_divergent_samples(self, X_remaining: np.ndarray, y_remaining: np.ndarray,
                                predictions: np.ndarray, probabilities: np.ndarray,
                                misclassified_indices: np.ndarray, remaining_indices: List[int]) -> List[int]:
        """
        Select most divergent failed candidates based on Bayesian confidence criteria.

        Args:
            X_remaining: Remaining feature matrix
            y_remaining: Remaining target labels
            predictions: Model predictions
            probabilities: Prediction probabilities
            misclassified_indices: Indices of misclassified samples
            remaining_indices: Original indices of remaining samples

        Returns:
            List: Indices of samples to add to training set
        """
        samples_to_add = []
        unique_classes = np.unique(y_remaining)

        print("🔍 Selecting most divergent failed candidates using Bayesian criteria...")

        # Group misclassified samples by true class scientifically
        class_samples = defaultdict(list)

        for idx_in_remaining in misclassified_indices:
            original_idx = remaining_indices[idx_in_remaining]
            true_class = y_remaining[idx_in_remaining]
            pred_class = predictions[idx_in_remaining]

            # Convert to indices for probability access scientifically
            true_class_idx = np.where(unique_classes == true_class)[0][0]
            pred_class_idx = np.where(unique_classes == pred_class)[0][0]

            # Calculate Bayesian margin (divergence)
            true_prob = probabilities[idx_in_remaining, true_class_idx]
            pred_prob = probabilities[idx_in_remaining, pred_class_idx]
            margin = pred_prob - true_prob  # Negative for misclassified

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_prob': true_prob,
                'pred_prob': pred_prob
            })

        # For each class, select most divergent samples scientifically
        max_samples = self.adaptive_config['max_samples_per_round']

        for class_id in unique_classes:
            if class_id not in class_samples or not class_samples[class_id]:
                continue

            class_data = class_samples[class_id]

            # Sort by margin (most negative first - most divergent)
            class_data.sort(key=lambda x: x['margin'])

            # Select top divergent samples scientifically
            selected_for_class = class_data[:max_samples]

            for sample in selected_for_class:
                samples_to_add.append(sample['index'])

            if selected_for_class:
                print(f"   ✅ Class {class_id}: Selected {len(selected_for_class)} divergent samples")

        print(f"📥 Total divergent samples to add: {len(samples_to_add)}")
        return samples_to_add

    def save_model(self, filepath: str = None) -> bool:
        """
        Save the trained model in ct_dbnn's native binary format scientifically.

        Args:
            filepath: Path where to save the model (optional)

        Returns:
            bool: True if successful
        """
        return ModelSerializer.save_model(self, filepath)

    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from ct_dbnn's binary file scientifically.

        Args:
            filepath: Path to the saved model

        Returns:
            bool: True if successful
        """
        result, message = ModelSerializer.load_model(filepath)
        if result is not None:
            # Copy all attributes from loaded model scientifically
            for attr, value in result.__dict__.items():
                setattr(self, attr, value)
            return True
        else:
            print(f"❌ {message}")
            return False

    def evaluate(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> float:
        """
        Evaluate the model on test data scientifically.

        Args:
            X_test: Test features (uses full data if None)
            y_test: Test labels (uses full data if None)

        Returns:
            float: Accuracy score
        """
        if X_test is None or y_test is None:
            if self.X_full is None:
                raise ValueError("No test data provided and no full data available")
            X_test, y_test = self.X_full, self.y_full

        accuracy = self.model.evaluate(X_test, y_test)
        print(f"📊 Evaluation accuracy: {accuracy:.4f}")
        return accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model scientifically.

        Args:
            X: Input features

        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)

    def predict_with_confidence(self, X: np.ndarray, top_n: int = 3):
        """
        Make predictions with confidence scores scientifically.

        Args:
            X: Input features
            top_n: Number of top predictions to return

        Returns:
            Tuple: (primary_predictions, probabilities, top_predictions, top_confidences)
        """
        return self.model.predict_with_confidence(X, top_n)

    def predict_file(self, file_path: str, output_path: str = None) -> bool:
        """
        Predict on a file using the trained model scientifically.

        Args:
            file_path: Path to input file
            output_path: Path for output predictions (optional)

        Returns:
            bool: True if successful
        """
        try:
            if output_path is None:
                base_name = os.path.splitext(file_path)[0]
                output_path = f"{base_name}_predictions.csv"

            results_df = self.model.predict_on_file(
                filepath=file_path,
                output_file=output_path
            )

            print(f"✅ Predictions saved to: {output_path}")
            return True

        except Exception as e:
            print(f"❌ Error during file prediction: {e}")
            return False

    def update_hyperparameters(self, ctdbnn_config: Dict, adaptive_config: Dict):
        """
        Update model hyperparameters scientifically.

        Args:
            ctdbnn_config: CT-DBNN core configuration
            adaptive_config: Adaptive learning configuration
        """
        # Update CT-DBNN configuration scientifically
        if ctdbnn_config:
            self.model.config.update(ctdbnn_config)
            self.config['ctdbnn_config'] = self.model.config

        # Update adaptive configuration scientifically
        if adaptive_config:
            self.adaptive_config.update(adaptive_config)
            self.config['adaptive_learning'] = self.adaptive_config

        print("✅ Hyperparameters updated scientifically")


class AdaptiveCTDBNNGUI:
    """
    Enhanced GUI for Adaptive CT-DBNN with feature selection and hyperparameter configuration.
    Provides an interactive interface for the adaptive learning system.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Adaptive CT-DBNN with Feature Selection")
        self.root.geometry("1400x900")

        self.adaptive_model = None
        self.model_trained = False
        self.data_loaded = False

        # Feature selection state
        self.feature_vars = {}
        self.target_var = tk.StringVar()

        self.setup_gui()

    def setup_gui(self):
        """Setup the main GUI interface with tabs."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Data Management Tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 Data Management")

        # Hyperparameters Tab
        self.hyperparams_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hyperparams_tab, text="⚙️ Hyperparameters")

        # Training Tab
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="🚀 Training & Evaluation")

        # Setup each tab
        self.setup_data_tab()
        self.setup_hyperparameters_tab()
        self.setup_training_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_data_tab(self):
        """Setup data management tab with feature selection."""
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(self.data_tab, text="Dataset Selection", padding="10")
        dataset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dataset_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, width=30)
        self.dataset_combo.grid(row=0, column=1, padx=5)

        # Populate with available datasets
        available_uci = list(ct_dbnn.UCI_DATASETS.keys())
        self.dataset_combo['values'] = available_uci

        ttk.Button(dataset_frame, text="Load Dataset Info",
                  command=self.load_dataset_info).grid(row=0, column=2, padx=5)
        ttk.Button(dataset_frame, text="Load Data",
                  command=self.load_data).grid(row=0, column=3, padx=5)
        ttk.Button(dataset_frame, text="Browse File",
                  command=self.browse_file).grid(row=0, column=4, padx=5)

        # Feature selection frame
        feature_frame = ttk.LabelFrame(self.data_tab, text="Feature Selection", padding="10")
        feature_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Target selection
        ttk.Label(feature_frame, text="Target Column:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_combo = ttk.Combobox(feature_frame, textvariable=self.target_var, width=20, state="readonly")
        self.target_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Feature selection area with scrollbar
        ttk.Label(feature_frame, text="Feature Columns:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=5)

        # Create frame for feature list with scrollbar
        feature_list_frame = ttk.Frame(feature_frame)
        feature_list_frame.grid(row=1, column=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)

        # Create canvas and scrollbar for feature list
        self.feature_canvas = tk.Canvas(feature_list_frame, height=200)
        feature_scrollbar = ttk.Scrollbar(feature_list_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_scroll_frame = ttk.Frame(self.feature_canvas)

        self.feature_scroll_frame.bind(
            "<Configure>",
            lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))
        )

        self.feature_canvas.create_window((0, 0), window=self.feature_scroll_frame, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=feature_scrollbar.set)

        self.feature_canvas.pack(side="left", fill="both", expand=True)
        feature_scrollbar.pack(side="right", fill="y")

        # Feature selection buttons
        button_frame = ttk.Frame(feature_frame)
        button_frame.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Button(button_frame, text="Select All Features",
                  command=self.select_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Deselect All Features",
                  command=self.deselect_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Select Only Numeric",
                  command=self.select_numeric_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Apply Selection",
                  command=self.apply_feature_selection).pack(side=tk.LEFT, padx=2)

        # Configure grid weights
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        feature_frame.columnconfigure(1, weight=1)
        feature_frame.rowconfigure(1, weight=1)

    def setup_hyperparameters_tab(self):
        """Setup hyperparameters configuration tab."""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.hyperparams_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # CT-DBNN Core Parameters Frame
        core_frame = ttk.LabelFrame(scrollable_frame, text="CT-DBNN Core Parameters", padding="10")
        core_frame.pack(fill=tk.X, pady=5, padx=10)

        # Resolution
        ttk.Label(core_frame, text="Resolution:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.resolution_var = tk.StringVar(value="8")
        ttk.Entry(core_frame, textvariable=self.resolution_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Learning Rate
        ttk.Label(core_frame, text="Learning Rate:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.learning_rate_var = tk.StringVar(value="0.01")
        ttk.Entry(core_frame, textvariable=self.learning_rate_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        # Max Epochs
        ttk.Label(core_frame, text="Max Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_epochs_var = tk.StringVar(value="100")
        ttk.Entry(core_frame, textvariable=self.max_epochs_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Batch Size
        ttk.Label(core_frame, text="Batch Size:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(core_frame, textvariable=self.batch_size_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        # Advanced CT-DBNN Options
        ttk.Label(core_frame, text="Advanced Options:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)

        self.use_complex_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(core_frame, text="Use Complex Tensor", variable=self.use_complex_var).grid(row=2, column=1, sticky=tk.W, padx=5)

        self.orthogonalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(core_frame, text="Orthogonalize Weights", variable=self.orthogonalize_var).grid(row=2, column=2, sticky=tk.W, padx=5)

        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(core_frame, text="Parallel Processing", variable=self.parallel_var).grid(row=2, column=3, sticky=tk.W, padx=5)

        # Adaptive Learning Parameters Frame
        adaptive_frame = ttk.LabelFrame(scrollable_frame, text="Adaptive Learning Parameters", padding="10")
        adaptive_frame.pack(fill=tk.X, pady=5, padx=10)

        # Initial samples per class
        ttk.Label(adaptive_frame, text="Initial Samples/Class:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.initial_samples_var = tk.StringVar(value="5")
        ttk.Entry(adaptive_frame, textvariable=self.initial_samples_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Max adaptive rounds
        ttk.Label(adaptive_frame, text="Max Adaptive Rounds:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.max_rounds_var = tk.StringVar(value="20")
        ttk.Entry(adaptive_frame, textvariable=self.max_rounds_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        # Patience
        ttk.Label(adaptive_frame, text="Patience:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.patience_var = tk.StringVar(value="10")
        ttk.Entry(adaptive_frame, textvariable=self.patience_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Min improvement
        ttk.Label(adaptive_frame, text="Min Improvement:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.min_improvement_var = tk.StringVar(value="0.001")
        ttk.Entry(adaptive_frame, textvariable=self.min_improvement_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        # Advanced Adaptive Options
        ttk.Label(adaptive_frame, text="Advanced Adaptive:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)

        self.enable_acid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adaptive_frame, text="Enable Acid Test", variable=self.enable_acid_var).grid(row=2, column=1, sticky=tk.W, padx=5)

        self.class_balance_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adaptive_frame, text="Class Balancing", variable=self.class_balance_var).grid(row=2, column=2, sticky=tk.W, padx=5)

        self.hard_mining_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adaptive_frame, text="Hard Sample Mining", variable=self.hard_mining_var).grid(row=2, column=3, sticky=tk.W, padx=5)

        # Sample selection strategy
        ttk.Label(adaptive_frame, text="Selection Strategy:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.selection_strategy_var = tk.StringVar(value="margin")
        strategy_combo = ttk.Combobox(adaptive_frame, textvariable=self.selection_strategy_var,
                                     values=["margin", "entropy", "random"], width=10)
        strategy_combo.grid(row=3, column=1, padx=5, pady=2)

        # Max samples per round
        ttk.Label(adaptive_frame, text="Max Samples/Round:").grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        self.max_samples_round_var = tk.StringVar(value="2")
        ttk.Entry(adaptive_frame, textvariable=self.max_samples_round_var, width=10).grid(row=3, column=3, padx=5, pady=2)

        # Control buttons frame
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Button(button_frame, text="Load Default Parameters",
                  command=self.load_default_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Current Parameters",
                  command=self.save_current_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Parameters",
                  command=self.apply_hyperparameters).pack(side=tk.RIGHT, padx=5)

    def setup_training_tab(self):
        """Setup training and evaluation tab."""
        # Control frame
        control_frame = ttk.LabelFrame(self.training_tab, text="Model Control", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Initialize Model",
                  command=self.initialize_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Run Adaptive Learning",
                  command=self.run_adaptive_learning, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Evaluate Model",
                  command=self.evaluate_model, width=15).pack(side=tk.LEFT, padx=2)

        # Prediction frame
        prediction_frame = ttk.LabelFrame(self.training_tab, text="Prediction", padding="10")
        prediction_frame.pack(fill=tk.X, pady=5)

        ttk.Button(prediction_frame, text="Predict File",
                  command=self.predict_file, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(prediction_frame, text="Test Model",
                  command=self.test_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(prediction_frame, text="Basic Visualizations",
                  command=self.show_visualizations, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(prediction_frame, text="Tensor Visualizations",
                  command=self.show_advanced_visualizations, width=16).pack(side=tk.LEFT, padx=2)

        # Model I/O frame
        io_frame = ttk.LabelFrame(self.training_tab, text="Model I/O", padding="10")
        io_frame.pack(fill=tk.X, pady=5)

        ttk.Button(io_frame, text="Save Model",
                  command=self.save_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_frame, text="Load Model",
                  command=self.load_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_frame, text="Exit",
                  command=self.exit_application, width=12).pack(side=tk.LEFT, padx=2)

        # Output frame
        output_frame = ttk.LabelFrame(self.training_tab, text="Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=20, width=100)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def log_output(self, message: str):
        """Add message to output text."""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()
        self.status_var.set(message)

    def load_dataset_info(self):
        """Load and display dataset information."""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return

        dataset_info = ct_dbnn.UCI_DATASETS.get(dataset_name, {})
        if not dataset_info:
            self.log_output(f"❌ Dataset '{dataset_name}' not found.")
            return

        # Display dataset information
        info_text = f"📊 Dataset: {dataset_name}\n"
        info_text += f"📝 Description: {dataset_info.get('description', 'N/A')}\n"
        info_text += f"🎯 Features: {dataset_info.get('features', 'N/A')}\n"
        info_text += f"📦 Samples: {dataset_info.get('samples', 'N/A')}\n"
        info_text += f"🏆 Best Known Accuracy: {dataset_info.get('best_accuracy', 'N/A')}\n"
        info_text += f"📚 Reference: {dataset_info.get('reference', 'N/A')}\n"

        self.log_output(info_text)

    def load_data(self):
        """Load dataset with feature selection."""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return

        try:
            self.adaptive_model = AdaptiveCTDBNN(dataset_name)
            success = self.adaptive_model.load_and_preprocess_data()

            if success:
                self.data_loaded = True
                self.log_output(f"✅ Dataset '{dataset_name}' loaded successfully")
                self.log_output(f"📊 Data shape: {self.adaptive_model.X_full.shape}")

                # Update feature selection UI
                self.update_feature_selection_ui()

            else:
                self.log_output(f"❌ Failed to load dataset '{dataset_name}'")

        except Exception as e:
            self.log_output(f"❌ Error loading data: {e}")

    def browse_file(self):
        """Browse for data file."""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            self.dataset_var.set(dataset_name)
            self.log_output(f"📁 Selected file: {file_path}")

            try:
                self.adaptive_model = AdaptiveCTDBNN(dataset_name)
                success = self.adaptive_model.load_and_preprocess_data(file_path)

                if success:
                    self.data_loaded = True
                    self.log_output(f"✅ File loaded successfully")

                    # Update feature selection UI
                    self.update_feature_selection_ui()

                else:
                    self.log_output(f"❌ Failed to load file")

            except Exception as e:
                self.log_output(f"❌ Error loading file: {e}")

    def update_feature_selection_ui(self):
        """Update the feature selection UI with available columns."""
        if not self.data_loaded or self.adaptive_model is None:
            return

        # Clear existing feature checkboxes
        for widget in self.feature_scroll_frame.winfo_children():
            widget.destroy()

        self.feature_vars = {}
        columns = self.adaptive_model.get_data_columns()

        if not columns:
            self.log_output("❌ No columns found in data")
            return

        numeric_cols = self.adaptive_model.get_numeric_columns()
        categorical_cols = self.adaptive_model.get_categorical_columns()

        # Update target combo with ALL columns
        self.target_combo['values'] = columns
        if self.adaptive_model.target_column in columns:
            self.target_var.set(self.adaptive_model.target_column)
        elif columns:
            # Default to last column for UCI datasets
            self.target_var.set(columns[-1])

        # Create feature checkboxes (exclude target column from features)
        for i, col in enumerate(columns):
            var = tk.BooleanVar(value=col != self.target_var.get())  # Auto-select non-target columns
            self.feature_vars[col] = var

            # Determine column type for styling
            if col in numeric_cols:
                col_type = "numeric"
                color = "blue"
            elif col in categorical_cols:
                col_type = "categorical"
                color = "green"
            else:
                col_type = "other"
                color = "gray"

            display_text = f"{col} ({col_type})"

            # Highlight target column
            if col == self.target_var.get():
                display_text = f"🎯 {display_text} [TARGET]"
                # Don't allow target to be selected as feature
                cb = ttk.Checkbutton(self.feature_scroll_frame, text=display_text, variable=var, state="disabled")
            else:
                cb = ttk.Checkbutton(self.feature_scroll_frame, text=display_text, variable=var)

            cb.pack(anchor=tk.W, padx=5, pady=2)

        self.log_output(f"🔧 Available columns: {len(columns)} total, {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        self.log_output(f"🎯 Current target: {self.target_var.get()}")

    def select_all_features(self):
        """Select all features."""
        for var in self.feature_vars.values():
            var.set(True)

    def deselect_all_features(self):
        """Deselect all features."""
        for var in self.feature_vars.values():
            var.set(False)

    def select_numeric_features(self):
        """Select only numeric features."""
        if not self.data_loaded:
            return

        numeric_cols = self.adaptive_model.get_numeric_columns()
        for col, var in self.feature_vars.items():
            var.set(col in numeric_cols)

    def apply_feature_selection(self):
        """Apply the current feature selection."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            # Get selected features
            selected_features = []
            for col, var in self.feature_vars.items():
                if var.get() and col != self.target_var.get():
                    selected_features.append(col)

            # Get target column
            target_column = self.target_var.get()

            if not selected_features:
                messagebox.showwarning("Warning", "Please select at least one feature.")
                return

            if not target_column:
                messagebox.showwarning("Warning", "Please select a target column.")
                return

            # Reload data with selected features
            success = self.adaptive_model.load_and_preprocess_data(
                target_column=target_column,
                selected_features=selected_features
            )

            if success:
                self.log_output(f"✅ Feature selection applied")
                self.log_output(f"🎯 Target: {target_column}")
                self.log_output(f"📊 Selected features: {len(selected_features)}")
                self.log_output(f"🔧 Features: {', '.join(selected_features)}")
            else:
                self.log_output("❌ Failed to apply feature selection")

        except Exception as e:
            self.log_output(f"❌ Error applying feature selection: {e}")

    def load_default_parameters(self):
        """Load default hyperparameters for the current dataset."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            dataset_name = self.dataset_var.get()
            dataset_info = ct_dbnn.UCI_DATASETS.get(dataset_name, {})

            # Set CT-DBNN parameters based on dataset characteristics
            if dataset_info:
                recommended_resol = dataset_info.get('recommended_resolution', 8)
                self.resolution_var.set(str(recommended_resol))
                self.log_output(f"✅ Set resolution to dataset default: {recommended_resol}")

            # Set adaptive parameters based on dataset size
            if self.adaptive_model.X_full is not None:
                n_samples = self.adaptive_model.X_full.shape[0]
                if n_samples < 100:
                    self.initial_samples_var.set("3")
                    self.max_rounds_var.set("10")
                elif n_samples > 1000:
                    self.initial_samples_var.set("8")
                    self.max_rounds_var.set("30")

            self.log_output("✅ Loaded default parameters based on dataset characteristics")

        except Exception as e:
            self.log_output(f"❌ Error loading default parameters: {e}")

    def save_current_parameters(self):
        """Save current hyperparameters to configuration file."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            dataset_name = self.dataset_var.get()
            config = {
                'dataset_name': dataset_name,
                'target_column': self.target_var.get(),

                'ctdbnn_config': {
                    'resol': int(self.resolution_var.get()),
                    'learning_rate': float(self.learning_rate_var.get()),
                    'max_epochs': int(self.max_epochs_var.get()),
                    'batch_size': int(self.batch_size_var.get()),
                    'use_complex_tensor': self.use_complex_var.get(),
                    'orthogonalize_weights': self.orthogonalize_var.get(),
                    'parallel_processing': self.parallel_var.get(),
                },

                'adaptive_learning': {
                    'initial_samples_per_class': int(self.initial_samples_var.get()),
                    'max_adaptive_rounds': int(self.max_rounds_var.get()),
                    'patience': int(self.patience_var.get()),
                    'min_improvement': float(self.min_improvement_var.get()),
                    'enable_acid_test': self.enable_acid_var.get(),
                    'class_balancing': self.class_balance_var.get(),
                    'hard_sample_mining': self.hard_mining_var.get(),
                    'sample_selection_strategy': self.selection_strategy_var.get(),
                    'max_samples_per_round': int(self.max_samples_round_var.get()),
                }
            }

            config_file = f"{dataset_name}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)

            self.log_output(f"✅ Saved current parameters to {config_file}")

        except Exception as e:
            self.log_output(f"❌ Error saving parameters: {e}")

    def apply_hyperparameters(self):
        """Apply current hyperparameters to the model."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            # Prepare CT-DBNN configuration
            ctdbnn_config = {
                'resol': int(self.resolution_var.get()),
                'use_complex_tensor': self.use_complex_var.get(),
                'orthogonalize_weights': self.orthogonalize_var.get(),
                'parallel_processing': self.parallel_var.get(),
                'smoothing_factor': 1e-8,
                'n_jobs': -1,
                'batch_size': 1000,
                'missing_value_placeholder': -99999,
            }

            # Prepare adaptive learning configuration
            adaptive_config = {
                'initial_samples_per_class': int(self.initial_samples_var.get()),
                'max_adaptive_rounds': int(self.max_rounds_var.get()),
                'patience': int(self.patience_var.get()),
                'min_improvement': float(self.min_improvement_var.get()),
                'enable_acid_test': self.enable_acid_var.get(),
                'class_balancing': self.class_balance_var.get(),
                'hard_sample_mining': self.hard_mining_var.get(),
                'sample_selection_strategy': self.selection_strategy_var.get(),
                'max_samples_per_round': int(self.max_samples_round_var.get()),
            }

            # Update model hyperparameters
            self.adaptive_model.update_hyperparameters(ctdbnn_config, adaptive_config)

            self.log_output("✅ Hyperparameters applied to model")
            self.log_output(f"   Resolution: {self.resolution_var.get()}")
            self.log_output(f"   Initial Samples/Class: {self.initial_samples_var.get()}")
            self.log_output(f"   Max Rounds: {self.max_rounds_var.get()}")

        except Exception as e:
            self.log_output(f"❌ Error applying hyperparameters: {e}")

    def initialize_model(self):
        """Initialize the model."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            self.adaptive_model.initialize_model()
            self.log_output("✅ Model initialized successfully")

        except Exception as e:
            self.log_output(f"❌ Error initializing model: {e}")

    def run_adaptive_learning(self):
        """Run adaptive learning."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data and initialize model first.")
            return

        try:
            self.log_output("🚀 Starting adaptive learning...")
            X_train, y_train, X_test, y_test = self.adaptive_model.adaptive_learn()

            self.model_trained = True
            self.log_output("✅ Adaptive learning completed successfully!")
            self.log_output(f"📊 Final training set: {len(X_train)} samples")
            self.log_output(f"📈 Final test set: {len(X_test)} samples")
            self.log_output(f"🏆 Best accuracy: {self.adaptive_model.best_accuracy:.4f}")

        except Exception as e:
            self.log_output(f"❌ Error during adaptive learning: {e}")

    def evaluate_model(self):
        """Evaluate the model."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please run adaptive learning first.")
            return

        try:
            self.log_output("📊 Evaluating model...")
            accuracy = self.adaptive_model.evaluate()
            self.log_output(f"🎯 Model accuracy: {accuracy:.4f}")

        except Exception as e:
            self.log_output(f"❌ Error during evaluation: {e}")

    def predict_file(self):
        """Predict on a file."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for prediction.")
            return

        file_path = filedialog.askopenfilename(
            title="Select File for Prediction",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.log_output(f"🔮 Predicting on file: {file_path}")
                success = self.adaptive_model.predict_file(file_path)

                if success:
                    self.log_output("✅ File prediction completed successfully")
                else:
                    self.log_output("❌ File prediction failed")

            except Exception as e:
                self.log_output(f"❌ Error during file prediction: {e}")

    def test_model(self):
        """Test the model on current data."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for testing.")
            return

        try:
            self.log_output("🧪 Testing model on current data...")
            predictions = self.adaptive_model.predict(self.adaptive_model.X_full)
            accuracy = accuracy_score(self.adaptive_model.y_full, predictions)

            self.log_output(f"🎯 Test accuracy: {accuracy:.4f}")
            self.log_output(f"📊 Sample predictions: {predictions[:10]}...")

        except Exception as e:
            self.log_output(f"❌ Error during testing: {e}")

    def show_visualizations(self):
        """Show basic model visualizations."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for visualization.")
            return

        try:
            self.log_output("📊 Generating basic visualizations...")
            # Create basic visualizations
            self.adaptive_model.visualizer.plot_class_distribution(self.adaptive_model.y_full)
            predictions = self.adaptive_model.predict(self.adaptive_model.X_full)
            self.adaptive_model.visualizer.plot_confusion_matrix(self.adaptive_model.y_full, predictions)

            self.log_output("✅ Basic visualizations created")
            self.log_output("   - Class distribution")
            self.log_output("   - Confusion matrix")

        except Exception as e:
            self.log_output(f"❌ Error showing visualizations: {e}")

    def show_advanced_visualizations(self):
        """Show advanced tensor visualizations."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for visualization.")
            return

        try:
            self.log_output("🎨 Generating advanced tensor visualizations...")

            # Create comprehensive visualizations
            predictions = self.adaptive_model.predict(self.adaptive_model.X_full)
            self.adaptive_model.visualizer.create_visualizations(
                self.adaptive_model.X_full,
                self.adaptive_model.y_full,
                predictions
            )

            self.log_output("✅ Advanced visualizations created!")
            self.log_output("📁 Check the 'visualizations' directory for:")
            self.log_output("   - Complex tensor orientations")
            self.log_output("   - Orthogonalization comparison")
            self.log_output("   - Weight distributions")
            self.log_output("   - Feature interaction heatmaps")

            # Offer to open the directory
            if messagebox.askyesno("Visualizations Ready",
                                 "Advanced tensor visualizations have been created!\n\nWould you like to open the visualizations directory?"):
                import subprocess
                try:
                    vis_dir = "visualizations"
                    abs_path = os.path.abspath(vis_dir)
                    if os.name == 'nt':  # Windows
                        subprocess.Popen(f'explorer "{abs_path}"')
                    elif os.name == 'posix':  # macOS, Linux
                        subprocess.Popen(['open', vis_dir] if sys.platform == 'darwin' else ['xdg-open', vis_dir])
                    self.log_output(f"📂 Opened directory: {abs_path}")
                except Exception as e:
                    self.log_output(f"⚠️ Could not open directory automatically: {e}")

        except Exception as e:
            self.log_output(f"❌ Error creating advanced visualizations: {e}")

    def save_model(self):
        """Save the model."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Model As",
            initialdir="Models",
            defaultextension=".bin",
            filetypes=[("Model files", "*.bin"), ("All files", "*.*")]
        )

        if file_path:
            success = self.adaptive_model.save_model(file_path)
            if success:
                self.log_output(f"✅ Model saved to: {file_path}")
            else:
                self.log_output(f"❌ Failed to save model")

    def load_model(self):
        """Load a model."""
        file_path = filedialog.askopenfilename(
            title="Load Model",
            initialdir="Models",
            filetypes=[("Model files", "*.bin"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.adaptive_model = AdaptiveCTDBNN()
                success = self.adaptive_model.load_model(file_path)

                if success:
                    self.model_trained = True
                    self.data_loaded = True
                    self.dataset_var.set(self.adaptive_model.dataset_name or "loaded_model")
                    self.log_output(f"✅ Model loaded from: {file_path}")
                    self.log_output(f"🏆 Loaded model accuracy: {self.adaptive_model.best_accuracy:.4f}")
                    self.log_output(f"🔧 Feature names: {self.adaptive_model.feature_names}")
                    self.log_output(f"🎯 Target column: {self.adaptive_model.target_column}")

                    # Update feature selection UI
                    self.update_feature_selection_ui()

                else:
                    self.log_output(f"❌ Failed to load model")

            except Exception as e:
                self.log_output(f"❌ Error loading model: {e}")

    def exit_application(self):
        """Exit the application."""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.quit()

def main():
    """Main function to run adaptive CT-DBNN."""
    import sys

    # Check for GUI flag
    if "--gui" in sys.argv or "-g" in sys.argv or len(sys.argv) == 1:
        if GUI_AVAILABLE:
            print("🎨 Launching Enhanced Adaptive CT-DBNN GUI...")
            root = tk.Tk()
            app = AdaptiveCTDBNNGUI(root)
            root.mainloop()
        else:
            print("❌ GUI not available. Using command line interface.")
            run_command_line()
    else:
        run_command_line()


def run_command_line():
    """Run the command line interface."""

    import sys

    # Parse command line arguments
    dataset_name = None
    file_path = None
    config_file = None
    target_column = None
    features = None

    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ["--csv", "--file"] and i + 1 < len(sys.argv):
            file_path = sys.argv[i + 1]
            i += 2
        elif arg in ["--dataset", "-d"] and i + 1 < len(sys.argv):
            dataset_name = sys.argv[i + 1]
            i += 2
        elif arg in ["--config", "-c"] and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            i += 2
        elif arg in ["--target", "-t"] and i + 1 < len(sys.argv):
            target_column = sys.argv[i + 1]
            i += 2
        elif arg in ["--features", "-f"] and i + 1 < len(sys.argv):
            features = [f.strip() for f in sys.argv[i + 1].split(',')]
            i += 2
        elif arg in ["--help", "-h"]:
            print_help()
            return
        elif not arg.startswith("--"):
            # Assume it's a dataset name or file path
            if arg.endswith('.csv') or arg.endswith('.data'):
                file_path = arg
            else:
                dataset_name = arg
            i += 1
        else:
            i += 1

    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {config_file}")
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")

    # Determine what to run
    if file_path:
        print(f"🎯 Running adaptive learning on file: {file_path}")
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        adaptive_model = AdaptiveCTDBNN(dataset_name, config)

        if adaptive_model.load_and_preprocess_data(file_path=file_path,
                                                 target_column=target_column,
                                                 selected_features=features):
            X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()
            print(f"✅ Adaptive learning completed!")
            print(f"📊 Results: {len(X_train)} training samples, {len(X_test)} test samples")
            print(f"🏆 Best accuracy: {adaptive_model.best_accuracy:.4f}")

            # Save model automatically
            adaptive_model.save_model()

            # Create visualizations
            print("🎨 Creating visualizations...")
            adaptive_model.visualizer.create_visualizations(adaptive_model.X_full, adaptive_model.y_full)
            print("✅ Visualizations created in 'visualizations/' directory")
        else:
            print("❌ Failed to load data from file")

    elif dataset_name:
        print(f"🎯 Running adaptive learning on dataset: {dataset_name}")
        adaptive_model = AdaptiveCTDBNN(dataset_name, config)

        if adaptive_model.load_and_preprocess_data(target_column=target_column,
                                                 selected_features=features):
            X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()
            print(f"✅ Adaptive learning completed!")
            print(f"📊 Results: {len(X_train)} training samples, {len(X_test)} test samples")
            print(f"🏆 Best accuracy: {adaptive_model.best_accuracy:.4f}")

            # Save model automatically
            adaptive_model.save_model()

            # Create visualizations
            print("🎨 Creating visualizations...")
            adaptive_model.visualizer.create_visualizations(adaptive_model.X_full, adaptive_model.y_full)
            print("✅ Visualizations created in 'visualizations/' directory")
        else:
            print("❌ Failed to load dataset")
    else:
        print_help()


def print_help():
    """Print command line help."""
    print("""
Usage: python adaptive_ctdbnn.py [OPTIONS] [DATASET_OR_FILE]

Options:
  --csv FILE, --file FILE    Use CSV file for training
  --dataset NAME, -d NAME    Use UCI dataset by name
  --config FILE, -c FILE     Load configuration from JSON file
  --target COL, -t COL       Specify target column name
  --features F1,F2,...       Specify features to use (comma-separated)
  --help, -h                 Show this help message

Examples:
  python adaptive_ctdbnn.py --csv data.csv
  python adaptive_ctdbnn.py --dataset iris
  python adaptive_ctdbnn.py iris
  python adaptive_ctdbnn.py data.csv
  python adaptive_ctdbnn.py --config my_config.json --csv data.csv
  python adaptive_ctdbnn.py --csv data.csv --target outcome --features age,income,score

Available UCI datasets:""")
    available_uci = list(ct_dbnn.UCI_DATASETS.keys())
    for dataset in available_uci:
        print(f"  - {dataset}")


if __name__ == "__main__":
    print("""
    ╔═════════════════════════════════════════════════════════════╗
    ║              🧠    CT-DBNN CLASSIFIER                       ║
    ║ Complex Tensor Difference Boosting Bayesian Neural Network  ║
    ║                 author: nsp@airis4d.com                     ║
    ║  Artificial Intelligence Research and Intelligent Systems   ║
    ║                 Thelliyoor 689544, India                    ║
    ║         Complex Tensor + Parallel + Orthogonisation         ║
    ║                 implementation: deepseek                    ║
    ╚═════════════════════════════════════════════════════════════╝
    """)
    main()
