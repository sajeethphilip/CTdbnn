"""
Simplified Adaptive CT-DBNN Wrapper
===================================

A streamlined adaptive learning system that leverages the enhanced ct_dbnn module
for core model operations while preserving the sophisticated adaptive learning logic.

Key Features:
- Adaptive sample selection with acid test validation
- Memory-efficient training with progressive sample addition
- UCI dataset integration
- GUI interface with feature selection and hyperparameter configuration
- Enhanced 3D visualizations of tensor operations
- Uses ct_dbnn's native binary format for model serialization
- Uses ct_dbnn for all model operations (training, prediction, serialization)

Architecture:
adaptive_ctdbnn.py (Wrapper, focuses on adaptive logic)
    ↓
ct_dbnn.py (Core engine for model operations)
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
                # Use ct_dbnn's preferred .bin extension
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
                'best_accuracy': float(adaptive_model.best_accuracy),  # Convert to float for JSON
                'best_training_indices': [int(idx) for idx in adaptive_model.best_training_indices],  # Convert to int
                'best_round': int(adaptive_model.best_round),
                'adaptive_round': int(adaptive_model.adaptive_round),
                'training_indices': [int(idx) for idx in adaptive_model.training_indices],
                'adaptive_config': adaptive_model.adaptive_config,
                'config': adaptive_model.config,
                'feature_names': adaptive_model.feature_names,
                'target_column': adaptive_model.target_column,
                'selected_features': adaptive_model.selected_features,
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
                json.dump(adaptive_state, f, indent=2, default=str)  # Use str for any non-serializable objects

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
    Basic visualization system for CT-DBNN.
    """

    def __init__(self, model, output_dir='visualizations'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_visualizations(self, X, y, predictions=None):
        """Create basic visualizations."""
        print("📊 Creating basic visualizations...")

        try:
            # Basic plots would go here
            self.plot_class_distribution(y)

            if predictions is not None:
                self.plot_confusion_matrix(y, predictions)

            print("✅ Basic visualizations created")

        except Exception as e:
            print(f"⚠️ Error creating visualizations: {e}")

    def plot_class_distribution(self, y):
        """Plot class distribution."""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            unique_classes, counts = np.unique(y, return_counts=True)
            plt.bar(unique_classes.astype(str), counts, alpha=0.7, color='skyblue')
            plt.title('Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/class_distribution.png')
            plt.close()

        except Exception as e:
            print(f"⚠️ Error plotting class distribution: {e}")

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        try:
            import matplotlib.pyplot as plt

            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()

            classes = np.unique(np.concatenate([y_true, y_pred]))
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes)
            plt.yticks(tick_marks, classes)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/confusion_matrix.png')
            plt.close()

        except Exception as e:
            print(f"⚠️ Error plotting confusion matrix: {e}")


class AdaptiveCTDBNN:
    """
    Simplified Adaptive CT-DBNN Wrapper

    This class implements adaptive learning logic while delegating all model operations
    to the enhanced ct_dbnn module. It focuses on intelligent sample selection and
    validation through acid testing.
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        """
        Initialize the adaptive learning wrapper.

        Args:
            dataset_name: Name of the dataset for tracking
            config: Configuration parameters for adaptive learning
        """
        self.dataset_name = dataset_name
        self.config = config or {}

        # Enhanced adaptive learning configuration
        self.adaptive_config = self.config.get('adaptive_learning', {})
        default_config = {
            "enable_adaptive": True,
            "initial_samples_per_class": 5,
            "max_adaptive_rounds": 20,
            "patience": 10,
            "min_improvement": 0.001,
            "enable_acid_test": True,
            "divergence_threshold": 0.1,
            "max_samples_per_round": 2,
            "sample_selection_strategy": "margin",  # margin, entropy, random
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

        # Initialize the core CT-DBNN model from ct_dbnn module
        ctdbnn_config = self.config.get('ctdbnn_config', {})
        default_ctdbnn_config = {
            'resol': 100,
            'use_complex_tensor': True,
            'orthogonalize_weights': True,
            'parallel_processing': True,
            'smoothing_factor': 1e-8,
            'n_jobs': -1,
            'memory_safe': True,
            'learning_rate': 0.01,
            'max_epochs': 100,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'gradient_clip': 1.0,
            'activation_function': 'relu',
            'hidden_layers': [64, 32],
            'dropout_rate': 0.2,
            'batch_normalization': True,
            'early_stopping_patience': 10,
            'validation_split': 0.1,
        }
        # Merge with provided config
        for key, value in default_ctdbnn_config.items():
            if key not in ctdbnn_config:
                ctdbnn_config[key] = value

        self.model = ct_dbnn.ParallelCTDBNN(ctdbnn_config)

        # Initialize visualizer
        self.visualizer = CTDBNNVisualizer(self.model)

        # Adaptive learning state
        self.training_indices = []
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.best_round = 0
        self.adaptive_round = 0
        self.patience_counter = 0

        # Statistics tracking
        self.round_stats = []
        self.start_time = datetime.now()
        self.adaptive_start_time = None

        # Data storage
        self.X_full = None
        self.y_full = None
        self.feature_names = None
        self.target_column = 'target'
        self.selected_features = None
        self.original_data = None

        # Sample tracking
        self.all_selected_samples = defaultdict(list)
        self.sample_selection_history = []

    def load_and_preprocess_data(self, file_path: str = None, target_column: str = None, selected_features: List[str] = None) -> bool:
        """
        Load and preprocess data with feature selection.

        Args:
            file_path: Path to data file (optional, uses dataset_name if not provided)
            target_column: Name of the target column
            selected_features: List of feature columns to use

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print("📥 Loading and preprocessing data...")

            # Use ct_dbnn's UCI dataset loader if it's a known dataset
            if self.dataset_name and self.dataset_name in ct_dbnn.UCI_DATASETS:
                print(f"🎯 Loading UCI dataset: {self.dataset_name}")
                dataset_info = ct_dbnn.UCI_DATASETS[self.dataset_name]
                df = ct_dbnn.UCIDatasetLoader.download_uci_data(dataset_info)
                if df is not None and not df.empty:
                    self.original_data = df

                    # For UCI datasets, the target is typically the last column
                    # Let the user specify target column or use the last column by default
                    if target_column and target_column in df.columns:
                        self.target_column = target_column
                    else:
                        # Use last column as target for UCI datasets
                        self.target_column = df.columns[-1]
                        print(f"🎯 Using last column as target: {self.target_column}")

                    if selected_features:
                        # Use selected features
                        self.selected_features = selected_features
                        if self.target_column not in selected_features:
                            features_to_use = selected_features + [self.target_column]
                        else:
                            features_to_use = selected_features
                    else:
                        # Use all features except target
                        self.selected_features = [col for col in df.columns if col != self.target_column]
                        features_to_use = df.columns.tolist()

                    # Filter data
                    df = df[features_to_use]
                    self.X_full = df.drop(columns=[self.target_column]).values
                    self.y_full = df[self.target_column].values
                    self.feature_names = df.drop(columns=[self.target_column]).columns.tolist()
                else:
                    print(f"❌ Failed to download UCI dataset: {self.dataset_name}")
                    return False
            else:
                # Load from file using pandas
                if file_path is None:
                    # Try to find dataset file
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
                        # Try to read with header first
                        try:
                            df = pd.read_csv(file_path)
                            has_header = True
                        except:
                            # If fails, read without header
                            df = pd.read_csv(file_path, header=None)
                            has_header = False
                            # Create generic column names
                            n_cols = df.shape[1]
                            df.columns = [f'col_{i}' for i in range(n_cols)]
                            print(f"📝 No header found, using generic column names: {df.columns.tolist()}")

                        self.original_data = df

                        if df.empty:
                            print("❌ Data file is empty")
                            return False

                        # Determine target column
                        if target_column:
                            if target_column in df.columns:
                                self.target_column = target_column
                            else:
                                print(f"❌ Target column '{target_column}' not found in data")
                                print(f"   Available columns: {df.columns.tolist()}")
                                return False
                        else:
                            # Auto-detect target (last column or common names)
                            target_candidates = ['target', 'class', 'label', 'outcome', 'diagnosis', 'type', 'species']
                            for candidate in target_candidates + [df.columns[-1]]:
                                if candidate in df.columns:
                                    self.target_column = candidate
                                    print(f"🎯 Auto-detected target column: {self.target_column}")
                                    break
                            else:
                                # Use last column as default
                                self.target_column = df.columns[-1]
                                print(f"🎯 Using last column as target: {self.target_column}")

                        # Determine features to use
                        if selected_features:
                            # Verify selected features exist
                            missing_features = [f for f in selected_features if f not in df.columns]
                            if missing_features:
                                print(f"❌ Selected features not found: {missing_features}")
                                print(f"   Available columns: {df.columns.tolist()}")
                                return False

                            self.selected_features = selected_features
                            if self.target_column not in selected_features:
                                features_to_use = selected_features + [self.target_column]
                            else:
                                features_to_use = selected_features
                        else:
                            # Use all features except target
                            self.selected_features = [col for col in df.columns if col != self.target_column]
                            features_to_use = df.columns.tolist()

                        # Check if we have features to use
                        if not self.selected_features:
                            print("❌ No features selected for training")
                            return False

                        # Filter data
                        df = df[features_to_use]
                        self.X_full = df.drop(columns=[self.target_column]).values
                        self.y_full = df[self.target_column].values
                        self.feature_names = df.drop(columns=[self.target_column]).columns.tolist()

                    except Exception as e:
                        print(f"❌ Error reading data file: {e}")
                        return False
                else:
                    print("❌ No data file found and no UCI dataset specified")
                    print("💡 Available UCI datasets:", list(ct_dbnn.UCI_DATASETS.keys()))
                    return False

            if self.X_full is None or self.y_full is None or len(self.X_full) == 0:
                print("❌ Failed to load data - no samples found")
                return False

            print(f"✅ Data loaded: {self.X_full.shape[0]} samples, {self.X_full.shape[1]} features")
            print(f"🎯 Target column: {self.target_column}")
            print(f"📊 Feature names: {self.feature_names}")
            print(f"🎯 Classes: {np.unique(self.y_full)}")
            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return False
    def get_data_columns(self) -> List[str]:
        """
        Get all available columns from the loaded data.

        Returns:
            List[str]: List of column names
        """
        if self.original_data is not None:
            return self.original_data.columns.tolist()
        return []

    def get_numeric_columns(self) -> List[str]:
        """
        Get numeric columns from the loaded data.

        Returns:
            List[str]: List of numeric column names
        """
        if self.original_data is not None:
            numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns.tolist()
            return numeric_cols
        return []

    def get_categorical_columns(self) -> List[str]:
        """
        Get categorical columns from the loaded data.

        Returns:
            List[str]: List of categorical column names
        """
        if self.original_data is not None:
            categorical_cols = self.original_data.select_dtypes(include=['object', 'category']).columns.tolist()
            return categorical_cols
        return []

    def initialize_model(self):
        """
        Initialize the CT-DBNN model with the full dataset architecture.

        This computes global likelihoods and prepares the model for adaptive training.
        """
        if self.X_full is None:
            raise ValueError("No data available. Call load_and_preprocess_data() first.")

        print("🏗️ Initializing CT-DBNN architecture with full dataset...")

        # Use ct_dbnn's compute_global_likelihoods to initialize the model
        self.model.compute_global_likelihoods(
            self.X_full,
            self.y_full,
            self.feature_names
        )

        print("✅ Model architecture initialized")

    def adaptive_learn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Main adaptive learning algorithm with acid test validation.

        This implements the core adaptive learning logic:
        1. Start with diverse initial samples
        2. Train model and run acid test on entire dataset
        3. Select most divergent misclassified samples
        4. Add them to training set and repeat
        5. Stop when no improvement or maximum rounds reached

        Returns:
            Tuple: (X_train, y_train, X_test, y_test) - Best training/test split found
        """
        print("\n🚀 STARTING ADAPTIVE LEARNING WITH CT-DBNN")
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

        # Initialize model architecture if not already done
        if not hasattr(self.model, 'likelihoods_computed') or not self.model.likelihoods_computed:
            self.initialize_model()

        # STEP 1: Select initial diverse training samples
        if hasattr(self, 'best_training_indices') and self.best_training_indices:
            print(f"📚 Using existing best training set with {len(self.best_training_indices)} samples")
            initial_indices = self.best_training_indices.copy()
        else:
            X_train, y_train, initial_indices = self._select_initial_training_samples(X, y)
            print(f"🎯 Selected new initial training set: {len(X_train)} samples")

        remaining_indices = [i for i in range(len(X)) if i not in initial_indices]

        print(f"📊 Initial training set: {len(initial_indices)} samples")
        print(f"📊 Remaining pool: {len(remaining_indices)} samples")

        # Initialize tracking
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

            # STEP 3: Run acid test on ENTIRE dataset
            print("🧪 Running acid test on entire dataset...")
            try:
                # Use ct_dbnn's predict method
                all_predictions = self.model.predict(X)
                acid_test_accuracy = accuracy_score(y, all_predictions)
                acid_test_history.append(acid_test_accuracy)
                print(f"📊 Acid test accuracy: {acid_test_accuracy:.4f}")

                # PRIMARY STOPPING CRITERION 1: 100% accuracy
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

            # STEP 5: Identify failed candidates
            X_remaining = X[remaining_indices]
            y_remaining = y[remaining_indices]

            remaining_predictions = self.model.predict(X_remaining)
            remaining_probs = self.model.predict_proba(X_remaining)

            # Find misclassified samples
            misclassified_mask = remaining_predictions != y_remaining
            misclassified_indices = np.where(misclassified_mask)[0]

            if len(misclassified_indices) == 0:
                print("✅ No misclassified samples in remaining data!")
            else:
                print(f"📊 Found {len(misclassified_indices)} misclassified samples")

                # STEP 6: Select most divergent failed candidates
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

            # STEP 7: Update best model and check for improvement
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

            # STOPPING CRITERION: No significant improvement
            if patience_counter >= patience:
                print(f"🛑 PATIENCE EXCEEDED: No improvement for {patience} rounds")
                break

            # STOPPING CRITERION: Maximum rounds
            if round_num >= max_rounds:
                print(f"🛑 MAXIMUM ROUNDS REACHED: {max_rounds} rounds")
                break

        # Finalize with best configuration
        print(f"\n🎉 Adaptive learning completed after {self.adaptive_round} rounds!")
        print(f"🏆 Best acid test accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
        print(f"📊 Final training set: {len(self.best_training_indices)} samples "
              f"({len(self.best_training_indices)/len(X)*100:.1f}% of total)")

        # Train final model with best configuration
        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        self.model.train(X_train_best, y_train_best)

        # Final evaluation
        final_predictions = self.model.predict(X)
        final_accuracy = accuracy_score(y, final_predictions)
        print(f"📊 Final acid test accuracy: {final_accuracy:.4f}")

        # Create visualizations
        self.visualizer.create_visualizations(X, y, final_predictions)

        # Create test set from remaining samples
        test_indices = [i for i in range(len(X)) if i not in self.best_training_indices]
        X_test_best = X[test_indices] if test_indices else np.array([])
        y_test_best = y[test_indices] if test_indices else np.array([])

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _select_initial_training_samples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Select initial diverse training samples using k-means clustering.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Tuple: (X_train, y_train, selected_indices)
        """
        initial_samples = self.adaptive_config['initial_samples_per_class']
        unique_classes = np.unique(y)

        initial_indices = []

        print("🎯 Selecting initial diverse training samples...")

        for class_id in unique_classes:
            class_indices = np.where(y == class_id)[0]

            if len(class_indices) > initial_samples:
                # Use k-means++ to select diverse samples
                class_data = X[class_indices]
                kmeans = KMeans(n_clusters=initial_samples, init='k-means++', n_init=1, random_state=42)
                kmeans.fit(class_data)

                # Find samples closest to cluster centers
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
        Select most divergent failed candidates based on prediction confidence.

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

        print("🔍 Selecting most divergent failed candidates...")

        # Group misclassified samples by true class
        class_samples = defaultdict(list)

        for idx_in_remaining in misclassified_indices:
            original_idx = remaining_indices[idx_in_remaining]
            true_class = y_remaining[idx_in_remaining]
            pred_class = predictions[idx_in_remaining]

            # Convert to indices for probability access
            true_class_idx = np.where(unique_classes == true_class)[0][0]
            pred_class_idx = np.where(unique_classes == pred_class)[0][0]

            # Calculate margin (divergence)
            true_prob = probabilities[idx_in_remaining, true_class_idx]
            pred_prob = probabilities[idx_in_remaining, pred_class_idx]
            margin = pred_prob - true_prob  # Negative for misclassified

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_prob': true_prob,
                'pred_prob': pred_prob
            })

        # For each class, select most divergent samples
        max_samples = self.adaptive_config['max_samples_per_round']

        for class_id in unique_classes:
            if class_id not in class_samples or not class_samples[class_id]:
                continue

            class_data = class_samples[class_id]

            # Sort by margin (most negative first - most divergent)
            class_data.sort(key=lambda x: x['margin'])

            # Select top divergent samples
            selected_for_class = class_data[:max_samples]

            for sample in selected_for_class:
                samples_to_add.append(sample['index'])

            if selected_for_class:
                print(f"   ✅ Class {class_id}: Selected {len(selected_for_class)} divergent samples")

        print(f"📥 Total divergent samples to add: {len(samples_to_add)}")
        return samples_to_add

    def save_model(self, filepath: str = None) -> bool:
        """
        Save the trained model in ct_dbnn's native binary format to Models/ directory.

        Args:
            filepath: Path where to save the model (optional)

        Returns:
            bool: True if successful
        """
        return ModelSerializer.save_model(self, filepath)

    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from ct_dbnn's binary file.

        Args:
            filepath: Path to the saved model

        Returns:
            bool: True if successful
        """
        result, message = ModelSerializer.load_model(filepath)
        if result is not None:
            # Copy all attributes from loaded model
            for attr, value in result.__dict__.items():
                setattr(self, attr, value)
            return True
        else:
            print(f"❌ {message}")
            return False

    def evaluate(self, X_test: np.ndarray = None, y_test: np.ndarray = None) -> float:
        """
        Evaluate the model on test data.

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
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            np.ndarray: Predictions
        """
        return self.model.predict(X)

    def predict_with_confidence(self, X: np.ndarray, top_n: int = 3):
        """
        Make predictions with confidence scores.

        Args:
            X: Input features
            top_n: Number of top predictions to return

        Returns:
            Tuple: (primary_predictions, probabilities, top_predictions, top_confidences)
        """
        return self.model.predict_with_confidence(X, top_n)

    def predict_file(self, file_path: str, output_path: str = None) -> bool:
        """
        Predict on a file using the trained model.

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
        Update model hyperparameters.

        Args:
            ctdbnn_config: CT-DBNN core configuration
            adaptive_config: Adaptive learning configuration
        """
        # Update CT-DBNN configuration
        if ctdbnn_config:
            self.model.config.update(ctdbnn_config)
            self.config['ctdbnn_config'] = self.model.config

        # Update adaptive configuration
        if adaptive_config:
            self.adaptive_config.update(adaptive_config)
            self.config['adaptive_learning'] = self.adaptive_config

        print("✅ Hyperparameters updated")


class AdaptiveCTDBNNGUI:
    """
    Enhanced GUI for Adaptive CT-DBNN with feature selection and hyperparameter configuration.

    Provides an interactive interface for the adaptive learning system
    while leveraging ct_dbnn for all model operations.
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
        self.notebook.add(self.data_tab, text="Data Management")

        # Hyperparameters Tab
        self.hyperparams_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hyperparams_tab, text="Hyperparameters")

        # Training Tab
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="Training & Evaluation")

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
        self.resolution_var = tk.StringVar(value="100")
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
        ttk.Button(prediction_frame, text="Show Visualizations",
                  command=self.show_visualizations, width=15).pack(side=tk.LEFT, padx=2)

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
        info_text += f"🎯 Target Column: {dataset_info.get('target_column', 'N/A')}\n"
        info_text += f"🏆 Best Known Accuracy: {dataset_info.get('best_accuracy', 'N/A')}\n"
        info_text += f"🔧 Best Method: {dataset_info.get('best_method', 'N/A')}\n"
        info_text += f"📚 Reference: {dataset_info.get('reference', 'N/A')}\n"
        info_text += f"💾 Recommended Resolution: {dataset_info.get('recommended_resolution', 100)}\n"

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

            # Set CT-DBNN parameters
            if dataset_info:
                recommended_resol = dataset_info.get('recommended_resolution', 100)
                self.resolution_var.set(str(recommended_resol))
                self.log_output(f"✅ Set resolution to dataset default: {recommended_resol}")

            # You can add more dataset-specific defaults here

            self.log_output("✅ Loaded default parameters")

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

            config_file = f"{dataset_name}.conf"
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
                'learning_rate': float(self.learning_rate_var.get()),
                'max_epochs': int(self.max_epochs_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'use_complex_tensor': self.use_complex_var.get(),
                'orthogonalize_weights': self.orthogonalize_var.get(),
                'parallel_processing': self.parallel_var.get(),
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
            self.log_output(f"   Learning Rate: {self.learning_rate_var.get()}")
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
        """Show model visualizations."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for visualization.")
            return

        try:
            self.log_output("📊 Generating visualizations...")
            # Visualizations are automatically created during adaptive learning
            self.log_output("✅ Visualizations available in 'visualizations' directory")
            self.log_output("   - Class distribution")
            self.log_output("   - Confusion matrix")

        except Exception as e:
            self.log_output(f"❌ Error showing visualizations: {e}")

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
    print("""
    ╔═════════════════════════════════════════════════════════════╗
    ║              🧠    CT-DBNN CLASSIFIER                       ║
    ║ Complex Tensor Difference Boosting Bayesian Neural Network  ║
    ║                 author: nsp@airis4d.com                     ║
    ║  Artificial Intelligence Research and Intelligent Systems   ║
    ║                 Thelliyoor 689544, India                    ║
    ║         Complex Tensor + Parallel + Orthogonisation         ║
    ╚═════════════════════════════════════════════════════════════╝
    """)

    import sys

    # Parse command line arguments
    dataset_name = None
    file_path = None
    config_file = None

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

        if adaptive_model.load_and_preprocess_data(file_path=file_path):
            X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()
            print(f"✅ Adaptive learning completed!")
            print(f"📊 Results: {len(X_train)} training samples, {len(X_test)} test samples")
            print(f"🏆 Best accuracy: {adaptive_model.best_accuracy:.4f}")

            # Save model automatically
            adaptive_model.save_model()
        else:
            print("❌ Failed to load data from file")

    elif dataset_name:
        print(f"🎯 Running adaptive learning on dataset: {dataset_name}")
        adaptive_model = AdaptiveCTDBNN(dataset_name, config)

        if adaptive_model.load_and_preprocess_data():
            X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()
            print(f"✅ Adaptive learning completed!")
            print(f"📊 Results: {len(X_train)} training samples, {len(X_test)} test samples")
            print(f"🏆 Best accuracy: {adaptive_model.best_accuracy:.4f}")

            # Save model automatically
            adaptive_model.save_model()
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
  --help, -h                 Show this help message

Examples:
  python adaptive_ctdbnn.py --csv data.csv
  python adaptive_ctdbnn.py --dataset iris
  python adaptive_ctdbnn.py iris
  python adaptive_ctdbnn.py data.csv
  python adaptive_ctdbnn.py --config my_config.json --csv data.csv

Available UCI datasets:""")
    available_uci = list(ct_dbnn.UCI_DATASETS.keys())
    for dataset in available_uci:
        print(f"  - {dataset}")


if __name__ == "__main__":
    main()
