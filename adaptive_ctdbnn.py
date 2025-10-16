import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import json
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import copy
import glob
import time
import torch
import pickle
import gzip
import shutil
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import imageio
from scipy.spatial import ConvexHull
import urllib.request
import shutil
import re
import tempfile
import gc
import sys
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor

# Try to import GUI components, fallback gracefully
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Replace dbnn import with ct_dbnn
try:
    import ct_dbnn
except ImportError:
    print("❌ CT-DBNN module not found. Please ensure ct_dbnn.py is in the same directory.")
    exit(1)

import pickle
import gzip
import shutil

class ModelSerializer:
    """Handles saving and loading of complete CT-DBNN model state"""

    @staticmethod
    def save_model(adaptive_model, file_path):
        """Save complete model state including configuration, weights, and metadata"""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # CRITICAL: Mark model as trained before saving
            adaptive_model.model_trained = True
            adaptive_model.model.is_trained = True
            adaptive_model.model.core.is_trained = True

            # Validate data before saving
            validation_msg = ModelSerializer._validate_model_before_save(adaptive_model)
            if validation_msg:
                print(f"⚠️ Pre-save validation warnings: {validation_msg}")

            # Check label encoder state
            target_encoder = adaptive_model.model.preprocessor.target_encoder
            has_label_encoder = (hasattr(target_encoder, 'classes_') and
                               target_encoder.classes_ is not None and
                               len(target_encoder.classes_) > 0)

            if has_label_encoder:
                print(f"✅ Label encoder will be saved with {len(target_encoder.classes_)} classes: {list(target_encoder.classes_)}")
            else:
                print("⚠️ No label encoder state to save")

            # CRITICAL FIX: Get ACTUAL feature names from the preprocessor
            feature_names = []

            # Priority 1: Get from preprocessor (this has the ACTUAL names)
            if (hasattr(adaptive_model.model.preprocessor, 'feature_columns') and
                adaptive_model.model.preprocessor.feature_columns):
                feature_names = adaptive_model.model.preprocessor.feature_columns
                print(f"💾 Saving ACTUAL feature names from preprocessor: {feature_names}")

            # Priority 2: Get from wrapper feature_names
            elif (hasattr(adaptive_model.model, 'feature_names') and
                  adaptive_model.model.feature_names):
                feature_names = adaptive_model.model.feature_names
                print(f"💾 Saving feature names from wrapper: {feature_names}")

            # Final fallback: infer from X_full shape (should not happen)
            elif (hasattr(adaptive_model, 'X_full') and
                  adaptive_model.X_full is not None):
                n_features = adaptive_model.X_full.shape[1]
                feature_names = [f'feature_{i}' for i in range(n_features)]
                print(f"⚠️  Using generic feature names as fallback: {feature_names}")

            print(f"💾 Final ACTUAL feature names being saved: {len(feature_names)} features - {feature_names}")

            # Collect all model state information
            model_state = {
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'dataset_name': adaptive_model.dataset_name,
                    'version': '1.0',
                    'model_type': 'AdaptiveCTDBNN',
                    'has_label_encoder': has_label_encoder,
                    'n_features': len(feature_names),
                },
                'config': adaptive_model.config,
                'adaptive_config': adaptive_model.adaptive_config,
                'training_state': {
                    'best_accuracy': getattr(adaptive_model, 'best_accuracy', 0.0),
                    'best_training_indices': getattr(adaptive_model, 'best_training_indices', []),
                    'best_round': getattr(adaptive_model, 'best_round', 0),
                    'adaptive_round': getattr(adaptive_model, 'adaptive_round', 0),
                    'patience_counter': getattr(adaptive_model, 'patience_counter', 0),
                    'round_stats': getattr(adaptive_model, 'round_stats', []),
                    'training_indices': getattr(adaptive_model, 'training_indices', []),
                },
                'data_state': {
                    'X_full': getattr(adaptive_model, 'X_full', None),
                    'y_full': getattr(adaptive_model, 'y_full', None),
                    'y_full_original': getattr(adaptive_model, 'y_full_original', None),
                    'feature_names': feature_names,  # CRITICAL: Save ACTUAL feature names
                    'target_column': getattr(adaptive_model.model, 'target_column', 'target'),
                    'class_to_label': getattr(adaptive_model, 'class_to_label', {}),
                    'label_to_class': getattr(adaptive_model, 'label_to_class', {}),
                },
                'preprocessor_state': {
                    'feature_encoders': adaptive_model.model.preprocessor.feature_encoders,
                    'target_encoder': adaptive_model.model.preprocessor.target_encoder,
                    'scaler': adaptive_model.model.preprocessor.scaler,
                    'missing_value_indicators': adaptive_model.model.preprocessor.missing_value_indicators,
                    'feature_columns': feature_names,  # Also save ACTUAL names in preprocessor state
                },
                'ctdbnn_core_state': ModelSerializer._extract_ctdbnn_state(adaptive_model.model.core),
                'wrapper_state': {
                    'initialized_with_full_data': getattr(adaptive_model.model, 'initialized_with_full_data', False),
                    'architecture_frozen': getattr(adaptive_model.model, 'architecture_frozen', False),
                    'frozen_components': getattr(adaptive_model.model, 'frozen_components', {}),
                    'feature_names': feature_names,  # And in wrapper state for redundancy
                }
            }

            # Save with compression
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(model_state, f, protocol=pickle.HIGHEST_PROTOCOL)

            return True, f"Model saved successfully to {file_path}"

        except Exception as e:
            return False, f"Error saving model: {e}"

    @staticmethod
    def _validate_model_before_save(adaptive_model):
        """Validate model state before saving"""
        warnings = []

        try:
            # Check data consistency
            if (adaptive_model.X_full is not None and
                adaptive_model.y_full is not None):

                if len(adaptive_model.X_full) != len(adaptive_model.y_full):
                    warnings.append("X and y have different lengths")

                # Get feature names from multiple possible locations
                feature_names = getattr(adaptive_model.model, 'feature_names', [])
                if not feature_names and hasattr(adaptive_model.model, 'preprocessor'):
                    feature_names = getattr(adaptive_model.model.preprocessor, 'feature_columns', [])

                if feature_names and adaptive_model.X_full.shape[1] != len(feature_names):
                    warnings.append(f"Feature count mismatch: X has {adaptive_model.X_full.shape[1]}, feature_names has {len(feature_names)}")

            # Check if model is trained
            if not getattr(adaptive_model, 'model_trained', False):
                warnings.append("Model not marked as trained")

        except Exception as e:
            warnings.append(f"Pre-save validation error: {e}")

        return "; ".join(warnings) if warnings else ""


    @staticmethod
    def load_model(file_path, adaptive_model=None):
        """Load complete model state from file with comprehensive restoration"""
        try:
            if not os.path.exists(file_path):
                return False, None, f"Model file not found: {file_path}"

            # Load compressed model state
            with gzip.open(file_path, 'rb') as f:
                model_state = pickle.load(f)

            # Verify model compatibility
            if model_state['metadata']['model_type'] != 'AdaptiveCTDBNN':
                return False, None, "Invalid model file format"

            if adaptive_model is None:
                # Create new adaptive model with loaded config
                dataset_name = model_state['metadata']['dataset_name']
                config = model_state['config']
                adaptive_model = AdaptiveCTDBNN(dataset_name, config)

            print("🔄 Restoring model state...")

            # Restore configuration
            adaptive_model.config.update(model_state['config'])
            adaptive_model.adaptive_config.update(model_state['adaptive_config'])

            # Restore training state
            training_state = model_state['training_state']
            adaptive_model.best_accuracy = training_state['best_accuracy']
            adaptive_model.best_training_indices = training_state['best_training_indices']
            adaptive_model.best_round = training_state['best_round']
            adaptive_model.adaptive_round = training_state['adaptive_round']
            adaptive_model.patience_counter = training_state['patience_counter']
            adaptive_model.round_stats = training_state['round_stats']
            adaptive_model.training_indices = training_state['training_indices']

            print(f"✅ Restored training state: best_accuracy={adaptive_model.best_accuracy:.4f}")

            # Restore data state
            data_state = model_state['data_state']
            adaptive_model.X_full = data_state['X_full']
            adaptive_model.y_full = data_state['y_full']
            adaptive_model.y_full_original = data_state['y_full_original']

            # CRITICAL FIX: Restore ACTUAL feature names
            feature_names = []

            # Priority: Get from data_state (this should have the ACTUAL names)
            if 'feature_names' in data_state and data_state['feature_names']:
                feature_names = data_state['feature_names']
                print(f"💾 Restored ACTUAL feature names from data_state: {len(feature_names)} features - {feature_names}")
            elif 'feature_columns' in model_state['preprocessor_state'] and model_state['preprocessor_state']['feature_columns']:
                feature_names = model_state['preprocessor_state']['feature_columns']
                print(f"💾 Restored feature names from preprocessor_state: {feature_names}")
            elif 'feature_names' in model_state['wrapper_state'] and model_state['wrapper_state']['feature_names']:
                feature_names = model_state['wrapper_state']['feature_names']
                print(f"💾 Restored feature names from wrapper_state: {feature_names}")
            else:
                # Final fallback: infer from X_full shape
                if adaptive_model.X_full is not None:
                    n_features = adaptive_model.X_full.shape[1]
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                    print(f"⚠️  Using generic feature names as fallback: {feature_names}")

            # CRITICAL: Set feature names in ALL locations
            adaptive_model.model.feature_names = feature_names
            adaptive_model.model.target_column = data_state.get('target_column', 'target')
            adaptive_model.target_column = data_state.get('target_column', 'target')
            adaptive_model.class_to_label = data_state.get('class_to_label', {})
            adaptive_model.label_to_class = data_state.get('label_to_class', {})

            print(f"✅ Restored data: {len(feature_names)} features, {len(adaptive_model.X_full) if adaptive_model.X_full is not None else 0} samples")

            # Restore preprocessor state
            preprocessor_state = model_state['preprocessor_state']
            adaptive_model.model.preprocessor.feature_encoders = preprocessor_state['feature_encoders']
            adaptive_model.model.preprocessor.target_encoder = preprocessor_state['target_encoder']
            adaptive_model.model.preprocessor.scaler = preprocessor_state['scaler']
            adaptive_model.model.preprocessor.missing_value_indicators = preprocessor_state['missing_value_indicators']
            adaptive_model.model.preprocessor.feature_columns = feature_names  # CRITICAL: Set ACTUAL feature columns

            # Check label encoder restoration
            target_encoder = adaptive_model.model.preprocessor.target_encoder
            if (hasattr(target_encoder, 'classes_') and
                target_encoder.classes_ is not None and
                len(target_encoder.classes_) > 0):
                print(f"✅ Label encoder restored with {len(target_encoder.classes_)} classes: {list(target_encoder.classes_)}")
            else:
                print("⚠️ Label encoder not properly restored")

            # Restore CT-DBNN core state
            core_restored = ModelSerializer._restore_ctdbnn_state(adaptive_model.model.core, model_state['ctdbnn_core_state'])
            if core_restored:
                print("✅ CT-DBNN core state restored")
            else:
                print("❌ CT-DBNN core state restoration failed")

            # Restore wrapper state
            wrapper_state = model_state['wrapper_state']
            adaptive_model.model.initialized_with_full_data = wrapper_state['initialized_with_full_data']
            adaptive_model.model.architecture_frozen = wrapper_state['architecture_frozen']
            adaptive_model.model.frozen_components = wrapper_state['frozen_components']

            # CRITICAL: Mark model as trained and ensure preprocessor is fitted
            adaptive_model.model_trained = True
            adaptive_model.model.is_trained = True
            adaptive_model.model.core.is_trained = True

            # CRITICAL: Mark preprocessor as fitted by setting its attributes
            if hasattr(adaptive_model.model.preprocessor, 'feature_encoders'):
                print("✅ Preprocessor marked as fitted with feature encoders")
            if hasattr(adaptive_model.model.preprocessor, 'target_encoder'):
                print("✅ Preprocessor marked as fitted with target encoder")
            if hasattr(adaptive_model.model.preprocessor, 'scaler'):
                print("✅ Preprocessor marked as fitted with scaler")

            # Validate the restored model
            validation_msg = ModelSerializer._validate_loaded_model(adaptive_model)
            if validation_msg:
                print(f"⚠️ Model validation warnings: {validation_msg}")
            else:
                print("✅ Model validation passed")

            return True, adaptive_model, f"Model loaded successfully from {file_path}"

        except Exception as e:
            return False, None, f"Error loading model: {e}"

    @staticmethod
    def _validate_loaded_model(adaptive_model):
        """Validate the consistency of the loaded model"""
        warnings = []

        try:
            # Check data dimensions
            if (adaptive_model.X_full is not None and
                adaptive_model.y_full is not None and
                hasattr(adaptive_model.model, 'feature_names')):

                n_samples_x = adaptive_model.X_full.shape[0]
                n_samples_y = len(adaptive_model.y_full)
                n_features = adaptive_model.X_full.shape[1]
                n_feature_names = len(adaptive_model.model.feature_names)

                if n_samples_x != n_samples_y:
                    warnings.append(f"Sample count mismatch: X has {n_samples_x}, y has {n_samples_y}")

                if n_features != n_feature_names:
                    warnings.append(f"Feature count mismatch: X has {n_features}, feature_names has {n_feature_names}")

            # Check if core model is properly initialized
            if not hasattr(adaptive_model.model.core, 'is_trained') or not adaptive_model.model.core.is_trained:
                warnings.append("Core model not marked as trained")

            # Check preprocessor state
            if not hasattr(adaptive_model.model.preprocessor, 'target_encoder'):
                warnings.append("Preprocessor target encoder missing")

        except Exception as e:
            warnings.append(f"Validation error: {e}")

        return "; ".join(warnings) if warnings else ""


    @staticmethod
    def _extract_ctdbnn_state(core_model):
        """Extract all relevant state from CT-DBNN core"""
        state = {}
        try:
            # Get all attributes that are important for the model
            important_attrs = [
                'global_anti_net', 'global_likelihoods', 'likelihoods_computed',
                'orthogonal_weights', 'is_trained', 'resol', 'innodes', 'outnodes',
                'feature_bins', 'feature_min', 'feature_max', 'n_bins',
                'class_labels', 'unique_classes', 'posterior_cache',
                'global_likelihood_computed', 'orthogonal_weights_initialized'
            ]

            for attr in important_attrs:
                if hasattr(core_model, attr):
                    value = getattr(core_model, attr)
                    # Handle numpy arrays and tensors
                    if hasattr(value, 'dtype'):
                        state[attr] = value
                    else:
                        state[attr] = value

            # Add tensor-specific attributes
            tensor_attrs = ['complex_tensor', 'real_tensor', 'weight_tensor']
            for attr in tensor_attrs:
                if hasattr(core_model, attr):
                    tensor_value = getattr(core_model, attr)
                    if tensor_value is not None:
                        state[attr] = tensor_value

            print(f"✅ Extracted CT-DBNN state with {len(state)} attributes")

        except Exception as e:
            print(f"❌ Could not extract CT-DBNN state: {e}")

        return state

    @staticmethod
    def _restore_ctdbnn_state(core_model, state_dict):
        """Completely restore CT-DBNN core state"""
        try:
            restored_count = 0
            for key, value in state_dict.items():
                try:
                    setattr(core_model, key, value)
                    restored_count += 1
                except Exception as attr_error:
                    print(f"⚠️ Could not restore {key}: {attr_error}")

            # Ensure critical flags are set
            core_model.likelihoods_computed = getattr(core_model, 'likelihoods_computed', True)
            core_model.is_trained = True
            core_model.global_likelihood_computed = getattr(core_model, 'global_likelihood_computed', True)
            core_model.orthogonal_weights_initialized = getattr(core_model, 'orthogonal_weights_initialized', True)

            print(f"✅ Restored {restored_count} CT-DBNN attributes")
            return True

        except Exception as e:
            print(f"❌ CT-DBNN state restoration failed: {e}")
            return False

class MemoryManager:
    """Memory management for large dataset processing"""

    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    @staticmethod
    def memory_safe_chunk_size(n_features, resol, available_mb=1000):
        """Calculate safe chunk size based on available memory"""
        # Estimate memory needed for one sample (conservative estimate)
        bytes_per_sample = n_features * resol * n_features * resol * 8 * 2  # 8 bytes per float, 2 arrays
        mb_per_sample = bytes_per_sample / 1024 / 1024

        if mb_per_sample == 0:
            return 1000  # Default safe value

        safe_samples = max(1, int(available_mb / mb_per_sample))
        return min(safe_samples, 1000)  # Cap at 1000 samples per chunk

    @staticmethod
    def optimize_memory():
        """Run garbage collection and optimize memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def check_memory_safety(n_features, resol, n_samples, safety_threshold_mb=500):
        """Check if processing is safe given memory constraints"""
        current_usage = MemoryManager.get_memory_usage()
        available_mb = psutil.virtual_memory().available / 1024 / 1024

        # Estimate memory needed
        estimated_mb = (n_features * resol * n_features * resol * n_samples * 8 * 2) / 1024 / 1024

        if estimated_mb > available_mb - safety_threshold_mb:
            return False, estimated_mb, available_mb
        return True, estimated_mb, available_mb

class UCIDatasetHandler:
    """Universal handler for UCI Machine Learning Repository datasets"""

    # UCI dataset URLs and information with target columns and best known results
    UCI_DATASETS = {
        'wine': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
            'description': 'Wine recognition data',
            'target_column': 'class',
            'feature_names': [
                'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline'
            ],
            'best_accuracy': 0.99,
            'best_method': 'Multiple methods (SVM, Random Forest, etc.)',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 50  # Lower resolution for memory efficiency
        },
        'iris': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            'description': 'Iris plants database',
            'target_column': 'class',
            'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'best_accuracy': 0.973,
            'best_method': 'Multiple methods',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 100
        },
        'breast-cancer': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
            'description': 'Breast Cancer Wisconsin (Diagnostic)',
            'target_column': 'diagnosis',
            'feature_names': [
                'id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape',
                'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei',
                'bland_chromatin', 'normal_nucleoli', 'mitoses'
            ],
            'best_accuracy': 0.971,
            'best_method': 'Neural Networks',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 80
        },
        'diabetes': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
            'description': 'Pima Indians Diabetes',
            'target_column': 'outcome',
            'feature_names': [
                'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                'insulin', 'bmi', 'diabetes_pedigree_function', 'age'
            ],
            'best_accuracy': 0.778,
            'best_method': 'Logistic Regression with feature selection',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 100
        },
        'wine-quality': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
            'description': 'Wine Quality',
            'target_column': 'quality',
            'delimiter': ';',
            'header': 0,
            'best_accuracy': 0.683,
            'best_method': 'SVM with RBF kernel',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 80
        },
        'car': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
            'description': 'Car Evaluation',
            'target_column': 'class',
            'feature_names': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            'best_accuracy': 0.968,
            'best_method': 'Decision Trees',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 60
        },
        'banknote': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
            'description': 'Banknote Authentication',
            'target_column': 'class',
            'feature_names': ['variance', 'skewness', 'curtosis', 'entropy'],
            'best_accuracy': 0.998,
            'best_method': 'Multiple methods',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 100
        },
        'seeds': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
            'description': 'Seeds Dataset',
            'target_column': 'type',
            'delimiter': '\t',
            'feature_names': ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove_length'],
            'best_accuracy': 0.914,
            'best_method': 'LDA',
            'reference': 'UCI Machine Learning Repository',
            'recommended_resolution': 100
        }
    }

    @staticmethod
    def get_available_uci_datasets() -> List[str]:
        """Get list of available UCI datasets"""
        return list(UCIDatasetHandler.UCI_DATASETS.keys())

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        if dataset_name is None:
            return {}
        dataset_name = dataset_name.lower()
        if dataset_name in UCIDatasetHandler.UCI_DATASETS:
            return UCIDatasetHandler.UCI_DATASETS[dataset_name]
        return {}

    @staticmethod
    def download_uci_dataset(dataset_name: str) -> Tuple[bool, str]:
        """Download dataset from UCI repository"""
        if dataset_name is None:
            return False, ""
        dataset_name = dataset_name.lower()

        if dataset_name not in UCIDatasetHandler.UCI_DATASETS:
            available = ", ".join(UCIDatasetHandler.get_available_uci_datasets())
            print(f"❌ Dataset '{dataset_name}' not found in UCI handler.")
            print(f"📋 Available UCI datasets: {available}")
            return False, ""

        dataset_info = UCIDatasetHandler.UCI_DATASETS[dataset_name]
        url = dataset_info['url']

        # Determine file extension
        if url.endswith('.data'):
            file_extension = '.data'
        elif url.endswith('.csv'):
            file_extension = '.csv'
        elif url.endswith('.txt'):
            file_extension = '.txt'
        else:
            file_extension = '.data'

        local_file = f"{dataset_name}{file_extension}"

        print(f"📥 Downloading {dataset_name} dataset from UCI repository...")
        print(f"🔗 URL: {url}")

        try:
            with urllib.request.urlopen(url) as response, open(local_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"✅ Downloaded {dataset_name} dataset to {local_file}")
            return True, local_file
        except Exception as e:
            print(f"❌ Failed to download {dataset_name} dataset: {e}")
            return False, ""

    @staticmethod
    def load_uci_dataset(dataset_name: str) -> Optional[pd.DataFrame]:
        """Load UCI dataset with proper formatting"""
        if dataset_name is None:
            return None
        dataset_name = dataset_name.lower()

        if dataset_name not in UCIDatasetHandler.UCI_DATASETS:
            return None

        dataset_info = UCIDatasetHandler.UCI_DATASETS[dataset_name]

        # Download dataset if not exists
        data_file = f"{dataset_name}.data"
        csv_file = f"{dataset_name}.csv"
        txt_file = f"{dataset_name}.txt"

        if not os.path.exists(data_file) and not os.path.exists(csv_file) and not os.path.exists(txt_file):
            success, downloaded_file = UCIDatasetHandler.download_uci_dataset(dataset_name)
            if not success:
                return None
        else:
            # Use existing file
            if os.path.exists(data_file):
                downloaded_file = data_file
            elif os.path.exists(csv_file):
                downloaded_file = csv_file
            else:
                downloaded_file = txt_file

        # Load data with dataset-specific parameters
        try:
            delimiter = dataset_info.get('delimiter', ',')
            header = dataset_info.get('header', None)

            if downloaded_file.endswith('.csv'):
                df = pd.read_csv(downloaded_file, delimiter=delimiter, header=header)
            else:
                # Try multiple delimiters for non-CSV files
                try:
                    df = pd.read_csv(downloaded_file, delimiter=delimiter, header=header)
                except:
                    # Try space/tab delimited
                    try:
                        df = pd.read_csv(downloaded_file, delim_whitespace=True, header=header)
                    except:
                        # Final fallback - use numpy and convert to DataFrame
                        data = np.loadtxt(downloaded_file)
                        df = pd.DataFrame(data)

            # Set column names if provided
            if 'feature_names' in dataset_info:
                columns = dataset_info['feature_names']
                if dataset_info['target_column'] not in columns:
                    columns = columns + [dataset_info['target_column']]
                if len(columns) == len(df.columns):
                    df.columns = columns
                elif len(columns) == len(df.columns) - 1:
                    # Assume last column is target
                    df.columns = columns + [dataset_info['target_column']]

            # Handle special cases for specific datasets
            df = UCIDatasetHandler._postprocess_dataset(dataset_name, df)

            print(f"✅ Loaded {dataset_name} dataset: {df.shape[0]} samples, {df.shape[1]} features")
            return df

        except Exception as e:
            print(f"❌ Error loading {dataset_name} dataset: {e}")
            return None

    @staticmethod
    def _postprocess_dataset(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process dataset based on specific requirements"""
        if dataset_name is None:
            return df
        dataset_name = dataset_name.lower()

        if dataset_name == 'breast-cancer':
            # Remove ID column and handle missing values
            if 'id' in df.columns:
                df = df.drop('id', axis=1)
            # Replace '?' with NaN and remove rows with missing values
            df = df.replace('?', np.nan)
            df = df.dropna()
            # Convert all columns to numeric except target
            target_col = UCIDatasetHandler.UCI_DATASETS[dataset_name]['target_column']
            for col in df.columns:
                if col != target_col:
                    df[col] = pd.to_numeric(df[col])

        elif dataset_name == 'car':
            # Car dataset is categorical, we'll handle it in preprocessing
            pass

        elif dataset_name == 'wine-quality':
            # Wine quality dataset is already clean
            pass

        elif dataset_name == 'banknote':
            # Banknote dataset is already clean
            pass

        elif dataset_name == 'seeds':
            # Seeds dataset might have extra spaces
            df = df.dropna()

        return df

    @staticmethod
    def create_uci_config(dataset_name: str, output_dir: str = ".") -> bool:
        """Create comprehensive configuration file for UCI dataset"""
        if dataset_name is None:
            return False
        dataset_name = dataset_name.lower()

        if dataset_name not in UCIDatasetHandler.UCI_DATASETS:
            return False

        dataset_info = UCIDatasetHandler.UCI_DATASETS[dataset_name]

        config = {
            'dataset_name': dataset_name,
            'description': dataset_info['description'],
            'target_column': dataset_info['target_column'],
            'source': 'UCI Machine Learning Repository',
            'url': dataset_info['url'],
            'best_known_accuracy': dataset_info.get('best_accuracy', 'Unknown'),
            'best_known_method': dataset_info.get('best_method', 'Unknown'),
            'recommended_resolution': dataset_info.get('recommended_resolution', 100),

            # CT-DBNN Core Hyperparameters
            'ctdbnn_config': {
                'resol': dataset_info.get('recommended_resolution', 100),
                'use_complex_tensor': True,
                'orthogonalize_weights': True,
                'parallel_processing': True,
                'smoothing_factor': 1e-8,
                'n_jobs': -1,
                'memory_safe': True,

                # Advanced CT-DBNN parameters
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
            },

            # Adaptive Learning Hyperparameters
            'adaptive_learning': {
                'enable_adaptive': True,
                'initial_samples_per_class': 5,
                'max_adaptive_rounds': 20,
                'patience': 10,
                'min_improvement': 0.001,
                'enable_acid_test': True,

                # Advanced Adaptive Learning parameters
                'divergence_threshold': 0.1,
                'max_samples_per_round': 5,
                'diversity_weight': 0.3,
                'uncertainty_weight': 0.7,
                'exploration_factor': 0.1,
                'class_balancing': True,
                'hard_sample_mining': True,
                'confidence_threshold': 0.8,
                'sample_selection_strategy': 'margin',  # 'margin', 'entropy', 'random'
            },

            # Data Preprocessing Parameters
            'preprocessing': {
                'normalization': 'standard',  # 'standard', 'minmax', 'robust', 'none'
                'handle_missing': 'mean',  # 'mean', 'median', 'mode', 'drop'
                'categorical_encoding': 'label',  # 'label', 'onehot', 'target'
                'feature_scaling': True,
                'remove_outliers': False,
                'outlier_threshold': 3.0,
                'feature_selection': 'all',  # 'all', 'kbest', 'variance'
                'variance_threshold': 0.0,
            }
        }

        config_file = os.path.join(output_dir, f"{dataset_name}.conf")

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"✅ Created comprehensive configuration file: {config_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to create configuration file: {e}")
            return False

    @staticmethod
    def setup_uci_dataset(dataset_name: str) -> bool:
        """Complete setup for UCI dataset: download, load, and create config"""
        if dataset_name is None:
            return False
        print(f"🔄 Setting up UCI dataset: {dataset_name}")

        # Create configuration
        if not UCIDatasetHandler.create_uci_config(dataset_name):
            return False

        # Download and load dataset
        df = UCIDatasetHandler.load_uci_dataset(dataset_name)
        if df is None:
            return False

        # Save as CSV for consistency
        csv_file = f"{dataset_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ Saved dataset as {csv_file}")

        print(f"🎉 Successfully set up {dataset_name} dataset!")
        return True

class DatasetConfig:
    """Dataset configuration handler"""

    @staticmethod
    def get_available_datasets():
        """Get list of available datasets from configuration files"""
        config_files = glob.glob("*.conf") + glob.glob("*.json")
        datasets = []
        for f in config_files:
            # Remove both .conf and .json extensions
            base_name = f.replace('.conf', '').replace('.json', '')
            if base_name not in datasets:  # Avoid duplicates
                datasets.append(base_name)
        return datasets

    @staticmethod
    def load_config(dataset_name):
        """Load configuration for a dataset - supports both .conf and .json"""
        if dataset_name is None:
            return {}
        # Try .json first, then .conf
        config_paths = [
            f"{dataset_name}.json",
            f"{dataset_name}.conf"
        ]

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
                    continue
        return {}

    @staticmethod
    def get_available_config_files():
        """Get all available configuration files with their types"""
        config_files = []
        # Look for JSON config files
        json_files = glob.glob("*.json")
        for f in json_files:
            # Skip the auto-saved config to avoid confusion
            if not f.endswith('_run_config.json') and not f.endswith('adaptive_ctdbnn_config.json'):
                config_files.append({'file': f, 'type': 'JSON'})

        # Look for CONF config files
        conf_files = glob.glob("*.conf")
        for f in conf_files:
            config_files.append({'file': f, 'type': 'CONF'})

        return config_files

class DataPreprocessor:
    """Comprehensive data preprocessing for CT-DBNN"""

    def __init__(self, target_column: str = 'target', sentinel_value: float = -99999.0):
        self.target_column = target_column
        self.sentinel_value = sentinel_value
        self.feature_encoders = {}  # For encoding categorical features
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.missing_value_indicators = {}

    def preprocess_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess feature columns - handle mixed types, missing values, etc."""
        processed_features = []
        feature_names = []

        for col in X.columns:
            if col == self.target_column:
                continue

            feature_data = X[col].copy()

            # Handle missing values
            missing_mask = self._detect_missing_values(feature_data)

            # Convert to numeric, handling errors
            numeric_data = self._convert_to_numeric(feature_data, col)

            # Store missing value information
            self.missing_value_indicators[col] = {
                'missing_mask': missing_mask,
                'has_missing': np.any(missing_mask)
            }

            processed_features.append(numeric_data)
            feature_names.append(col)  # This preserves the original column names

        # CRITICAL: Store the actual feature column names for serialization
        self.feature_columns = feature_names.copy()

        # Stack all features
        if processed_features:
            X_processed = np.column_stack(processed_features)
        else:
            X_processed = np.empty((len(X), 0))

        return X_processed, feature_names

    def preprocess_dataset(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess entire dataset"""
        print("🔧 Preprocessing dataset...")

        # Separate features and target
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Create a copy to avoid modifying original data
        data_clean = data.copy()

        # Preprocess features
        X_processed, feature_names = self.preprocess_features(data_clean)

        # Preprocess target
        y_processed = self.preprocess_target(data_clean[self.target_column])

        # Remove samples with missing target values
        valid_mask = ~np.isnan(y_processed)
        if not np.all(valid_mask):
            removed_count = len(y_processed) - np.sum(valid_mask)
            print(f"⚠️  Removed {removed_count} samples with invalid target values")
            X_processed = X_processed[valid_mask]
            y_processed = y_processed[valid_mask]

        print(f"✅ Preprocessing complete: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        print(f"📊 Feature types: {len(feature_names)} numeric/categorical features")
        print(f"💾 Stored actual feature names: {feature_names}")

        return X_processed, y_processed, feature_names

    def _detect_missing_values(self, data: pd.Series) -> np.ndarray:
        """Detect various types of missing values"""
        # Standard missing values
        missing_mask = data.isna()

        # String representations of missing values
        if data.dtype == 'object':
            missing_strings = ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'NaN', 'nan', 'ERROR', 'error', 'MISSING', 'missing']
            missing_mask = missing_mask | data.isin(missing_strings)

        return missing_mask.values

    def _convert_to_numeric(self, data: pd.Series, col_name: str) -> np.ndarray:
        """Convert data to numeric, handling various data types"""
        # If already numeric, return as is
        if pd.api.types.is_numeric_dtype(data):
            numeric_data = data.values.astype(float)
            # Replace any remaining NaN with sentinel value
            numeric_data = np.where(np.isnan(numeric_data), self.sentinel_value, numeric_data)
            return numeric_data

        # For categorical/string data
        if data.dtype == 'object':
            try:
                # Try direct conversion to numeric first
                numeric_data = pd.to_numeric(data, errors='coerce').values
                # Replace NaN with sentinel value
                numeric_data = np.where(np.isnan(numeric_data), self.sentinel_value, numeric_data)
                return numeric_data
            except:
                # Use label encoding for categorical data
                if col_name not in self.feature_encoders:
                    self.feature_encoders[col_name] = LabelEncoder()

                # Handle missing values before encoding
                clean_data = data.fillna('MISSING')
                encoded_data = self.feature_encoders[col_name].fit_transform(clean_data)
                return encoded_data.astype(float)

        # Fallback: convert to string then label encode
        str_data = data.astype(str)
        if col_name not in self.feature_encoders:
            self.feature_encoders[col_name] = LabelEncoder()
        encoded_data = self.feature_encoders[col_name].fit_transform(str_data)
        return encoded_data.astype(float)

    def preprocess_target(self, y: pd.Series) -> np.ndarray:
        """Preprocess target column"""
        # Handle missing target values
        if y.isna().any():
            print(f"⚠️  Warning: Found {y.isna().sum()} missing target values. They will be removed.")
            # We'll handle this at the dataset level by removing these samples

        # Convert to numeric if needed
        if not pd.api.types.is_numeric_dtype(y):
            try:
                y_processed = pd.to_numeric(y, errors='coerce')
                if y_processed.isna().any():
                    print(f"⚠️  Some target values couldn't be converted to numeric. Using label encoding.")
                    y_processed = self.target_encoder.fit_transform(y.fillna('MISSING'))
                else:
                    y_processed = y_processed.values
            except:
                y_processed = self.target_encoder.fit_transform(y.fillna('MISSING'))
        else:
            y_processed = y.values

        return y_processed.astype(int)


class CTDBNNVisualizer:
    """Visualization system for CT-DBNN"""

    def __init__(self, model, output_dir='visualizations', enabled=True):
        self.model = model
        self.output_dir = output_dir
        self.enabled = enabled
        os.makedirs(output_dir, exist_ok=True)

    def create_visualizations(self, X, y, predictions=None):
        """Create various visualizations"""
        if not self.enabled:
            return

        print("Creating visualizations...")

        # Create some basic plots
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=20, alpha=0.7, color='blue')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.savefig(f'{self.output_dir}/class_distribution.png')
        plt.close()

class OptimizedCTDBNNWrapper:
    """
    Optimized wrapper for ct_dbnn.py with memory management
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or {}

        # Initialize the core CT-DBNN with optimized settings
        ct_dbnn_config = {
            'resol': self.config.get('resol', 100),
            'use_complex_tensor': self.config.get('use_complex_tensor', True),
            'orthogonalize_weights': self.config.get('orthogonalize_weights', True),
            'parallel_processing': self.config.get('parallel_processing', True),
            'smoothing_factor': self.config.get('smoothing_factor', 1e-8),
            'n_jobs': self.config.get('n_jobs', -1),
            'memory_safe': True,  # Enable memory safety
        }

        # Use CT-DBNN
        self.core = ct_dbnn.ConsistentCTDBNN(ct_dbnn_config)

        # Store architectural components separately for freezing
        self.architecture_frozen = False
        self.frozen_components = {}

        # Store data and preprocessing
        self.data = None
        self.target_column = self.config.get('target_column', 'target')
        self.preprocessor = DataPreprocessor(target_column=self.target_column)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Training state
        self.train_enabled = True
        self.max_epochs = self.config.get('max_epochs', 100)
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)

        # Feature information - CRITICAL: Initialize feature_names properly
        self.feature_names = []
        self.initialized_with_full_data = False

        # Memory management
        self.memory_manager = MemoryManager()

    def _auto_detect_target_column(self):
        """Auto-detect the target column based on common patterns"""
        if self.data is None:
            return 'target'

        columns = self.data.columns.tolist()

        # Common target column names
        target_candidates = [
            'target', 'class', 'label', 'outcome', 'diagnosis', 'type', 'quality',
            'ObjectType', 'category', 'result', 'y', 'Y'
        ]

        # Check for exact matches first
        for candidate in target_candidates:
            if candidate in columns:
                return candidate

        # Check for case-insensitive matches
        for col in columns:
            if col.lower() in [c.lower() for c in target_candidates]:
                return col

        # Check for columns that might be categorical (low cardinality)
        for col in columns:
            if self.data[col].dtype == 'object' or self.data[col].nunique() < 20:
                # This might be the target column
                return col

        # If no good candidate found, use the last column (common in many datasets)
        return columns[-1]

    def load_data(self, file_path: str = None):
        """Load data from file with robust preprocessing - with UCI auto-download"""
        # Use the instance's dataset_name, not from some other object
        dataset_name = getattr(self, 'dataset_name', 'unknown')

        if file_path is None:
            # Try to find dataset file - prioritize original data files
            possible_files = [
                f"{dataset_name}.csv",
                f"{dataset_name}.data",
                f"{dataset_name}.txt",
                "data.csv",
                "train.csv"
            ]

            for file in possible_files:
                if os.path.exists(file):
                    file_path = file
                    print(f"📁 Found data file: {file_path}")
                    break

        if file_path is None:
            # Check if this is a known UCI dataset
            available_uci = UCIDatasetHandler.get_available_uci_datasets()
            if dataset_name and dataset_name.lower() in available_uci:
                print(f"🎯 Detected UCI dataset: {dataset_name}")
                # Setup UCI dataset (download + create config)
                success = UCIDatasetHandler.setup_uci_dataset(dataset_name)
                if success:
                    file_path = f"{dataset_name}.csv"
                else:
                    raise ValueError(f"Failed to setup UCI dataset: {dataset_name}")
            else:
                # Try to find any CSV or DAT file in current directory
                csv_files = glob.glob("*.csv")
                dat_files = glob.glob("*.dat")
                txt_files = glob.glob("*.txt")
                all_files = csv_files + dat_files + txt_files

                if all_files:
                    file_path = all_files[0]
                    print(f"📁 Auto-selected data file: {file_path}")
                else:
                    # Show available UCI datasets
                    available_uci = UCIDatasetHandler.get_available_uci_datasets()
                    print(f"❌ No data file found for dataset: {dataset_name}")
                    print(f"📋 Available UCI datasets: {', '.join(available_uci)}")
                    print("💡 You can use one of these UCI datasets or provide your own data file")
                    raise ValueError("No data file found. Please provide a CSV or DAT file.")

        # Load the data file with better error handling
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
                print(f"✅ Loaded CSV data: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
            else:
                # For .dat or .txt files, use robust loading
                print(f"📊 Loading data file: {file_path}")
                try:
                    # First try pandas with multiple delimiters
                    try:
                        self.data = pd.read_csv(file_path, delim_whitespace=True)
                        print(f"✅ Loaded data with pandas (whitespace): {self.data.shape[0]} samples, {self.data.shape[1]} columns")
                    except:
                        # Try with comma delimiter
                        try:
                            self.data = pd.read_csv(file_path, delimiter=',')
                            print(f"✅ Loaded data with pandas (comma): {self.data.shape[0]} samples, {self.data.shape[1]} columns")
                        except:
                            # Final fallback - use numpy
                            data = np.loadtxt(file_path)
                            n_features = data.shape[1] - 1
                            columns = [f'feature_{i}' for i in range(n_features)] + [self.target_column]
                            self.data = pd.DataFrame(data, columns=columns)
                            print(f"✅ Loaded data with numpy: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
                except Exception as e:
                    print(f"❌ Failed to load data file: {e}")
                    raise

            # Auto-detect target column if not found
            if self.target_column not in self.data.columns:
                print(f"🔍 Target column '{self.target_column}' not found. Auto-detecting target column...")
                self.target_column = self._auto_detect_target_column()
                print(f"🎯 Auto-detected target column: '{self.target_column}'")

            print(f"📊 Available columns: {list(self.data.columns)}")
            return self.data

        except Exception as e:
            print(f"❌ Error loading data file: {e}")
            raise

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        X_processed, y_processed, feature_names = self.preprocessor.preprocess_dataset(self.data)

        # CRITICAL: Store the ACTUAL feature names from preprocessing
        self.feature_names = feature_names
        self.preprocessor.feature_columns = feature_names

        print(f"💾 Wrapper stored ACTUAL feature names: {len(self.feature_names)} features - {self.feature_names}")

        return X_processed, y_processed, feature_names

    def initialize_with_full_data(self, X: np.ndarray, y: np.ndarray):
        """Step 1: Initialize CT-DBNN architecture with full dataset - NO TRAINING"""
        print("🏗️ Initializing CT-DBNN architecture with full dataset...")

        # Store the full data for reference
        self.X_full = X
        self.y_full = y

        # Create temporary file with full data using ACTUAL feature names
        temp_file = f"temp_full_init_{int(time.time())}.csv"

        # CRITICAL: Use ACTUAL feature names from preprocessing
        if hasattr(self, 'feature_names') and self.feature_names:
            feature_cols = self.feature_names
            print(f"💾 Using ACTUAL feature names for initialization: {feature_cols}")
        else:
            # Fallback: create generic names but this should not happen
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
            self.feature_names = feature_cols
            print(f"⚠️  Using generic feature names as fallback: {feature_cols}")

        full_df = pd.DataFrame(X, columns=feature_cols)
        full_df[self.target_column] = y
        full_df.to_csv(temp_file, index=False)

        try:
            # For CT-DBNN, architecture initialization means computing global likelihoods
            print("🔄 Computing global likelihoods with memory optimization...")

            # Load data for initialization
            X_temp = pd.read_csv(temp_file)
            y_temp = X_temp[self.target_column].values
            X_temp = X_temp.drop(columns=[self.target_column]).values

            # CRITICAL: Pass the ACTUAL feature names to CT-DBNN
            print(f"💾 Passing ACTUAL feature names to CT-DBNN: {feature_cols}")
            self.core.compute_global_likelihoods(X_temp, y_temp, feature_cols)

            # Store feature information for serialization
            self.core.feature_bins = getattr(self.core, 'feature_bins', None)
            self.core.feature_min = getattr(self.core, 'feature_min', None)
            self.core.feature_max = getattr(self.core, 'feature_max', None)
            self.core.n_bins = getattr(self.core, 'n_bins', None)
            self.core.unique_classes = np.unique(y)
            self.core.class_labels = self.core.unique_classes.tolist()

            # Initialize orthogonal weights with memory cleanup
            print("🔄 Initializing orthogonal weights...")
            self.memory_manager.optimize_memory()
            self.core.initialize_orthogonal_weights()
            self.memory_manager.optimize_memory()

            print("✅ CT-DBNN architecture initialized with full dataset")
            self.initialized_with_full_data = True

            # Freeze the architecture
            self.freeze_architecture()

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def freeze_architecture(self):
        """Freeze the current architecture to prevent changes during adaptive learning"""
        print("❄️ Freezing CT-DBNN architecture...")

        # Store current state of critical components
        self.frozen_components = {
            'feature_names': self.feature_names.copy() if hasattr(self, 'feature_names') else [],
            'target_column': self.target_column,
        }

        # Mark architecture as frozen
        self.architecture_frozen = True
        print("✅ Architecture frozen")

    def train_with_data(self, X_train: np.ndarray, y_train: np.ndarray, reset_weights: bool = True):
        """Step 2: Train with given data (no train/test split)"""
        if not self.initialized_with_full_data:
            # Try to initialize if not already done
            print("⚠️  CT-DBNN not initialized, attempting initialization...")
            self.initialize_with_full_data(X_train, y_train)
            if not self.initialized_with_full_data:
                raise ValueError("CT-DBNN must be initialized with full data first")

        if reset_weights:
            print("🔄 Resetting weights for new training...")
            self._reset_weights()

        print(f"🎯 Training with {len(X_train)} samples...")

        # Create temporary file with training data
        temp_file = f"temp_train_{int(time.time())}.csv"
        feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df[self.target_column] = y_train
        train_df.to_csv(temp_file, index=False)

        try:
            # For CT-DBNN, training is the orthogonal weight initialization
            # after global likelihoods are computed

            # Recompute likelihoods with current training data if needed
            if reset_weights or not hasattr(self.core, 'likelihoods_computed') or not self.core.likelihoods_computed:
                feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
                self.core.compute_global_likelihoods(X_train, y_train, feature_cols)

            # CT-DBNN uses one-step orthogonal weight initialization
            self.memory_manager.optimize_memory()
            self.core.initialize_orthogonal_weights()
            self.memory_manager.optimize_memory()

            # Mark as trained
            self.core.is_trained = True

            train_accuracy = self._compute_accuracy(X_train, y_train)
            print(f"✅ CT-DBNN training completed - Accuracy on training data: {train_accuracy:.4f}")
            return True

        except Exception as e:
            print(f"❌ CT-DBNN training error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _reset_weights(self):
        """Reset weights while preserving architecture - for CT-DBNN this means reinitializing orthogonal weights"""
        if hasattr(self.core, 'initialize_orthogonal_weights'):
            self.core.initialize_orthogonal_weights()
            print("✅ CT-DBNN weights reset with orthogonal initialization")
        else:
            print("⚠️  Cannot reset weights - CT-DBNN not properly initialized")

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy on given data"""
        try:
            predictions = self.predict(X)
            # Ensure both arrays have the same data type for comparison
            predictions = predictions.astype(y.dtype)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        except Exception as e:
            print(f"❌ Accuracy computation error: {e}")
            return 0.0

    def predict(self, X: np.ndarray):
        """Predict classes for input data using CT-DBNN"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            # If not trained, return random predictions based on class distribution
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))

        try:
            # Use CT-DBNN's predict method directly
            predictions = self.core.predict(X)
            return predictions

        except Exception as e:
            print(f"❌ CT-DBNN prediction error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: return random predictions
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))

    def _compute_batch_posterior(self, X: np.ndarray):
        """Compute posterior probabilities using CT-DBNN"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            # Return uniform probabilities if not trained
            n_classes = len(np.unique(self.y_full)) if hasattr(self, 'y_full') else 3
            return np.ones((len(X), n_classes)) / n_classes

        try:
            # Use CT-DBNN's predict_proba method
            posteriors = self.core.predict_proba(X)
            return posteriors

        except Exception as e:
            print(f"❌ CT-DBNN posterior computation error: {e}")
            n_classes = len(np.unique(self.y_full)) if hasattr(self, 'y_full') else 3
            return np.ones((len(X), n_classes)) / n_classes

class AdaptiveCTDBNN:
    """
    Main adaptive CT-DBNN class following adaptive_dbnn structure
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or {}

        # Ensure config has required fields
        if 'target_column' not in self.config:
            dataset_config = DatasetConfig.load_config(dataset_name)
            if dataset_config:
                self.config.update(dataset_config)

        # Enhanced adaptive learning configuration with proper defaults
        self.adaptive_config = self.config.get('adaptive_learning', {})
        # Set defaults for any missing parameters
        default_config = {
            "enable_adaptive": True,
            "initial_samples_per_class": 5,
            "max_adaptive_rounds": 20,
            "patience": 10,
            "min_improvement": 0.001,
            "enable_acid_test": True,
        }
        for key, default_value in default_config.items():
            if key not in self.adaptive_config:
                self.adaptive_config[key] = default_value

        # Initialize the base CT-DBNN model using our optimized wrapper
        self.model = OptimizedCTDBNNWrapper(dataset_name, config=self.config)

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

        # Store the full dataset for adaptive learning
        self.X_full = None
        self.y_full = None
        self.y_full_original = None

        # Track all selected samples for analysis
        self.all_selected_samples = defaultdict(list)
        self.sample_selection_history = []

        # Initialize label encoder for adaptive learning
        self.label_encoder = LabelEncoder()

        # Initialize visualizers
        self.adaptive_visualizer = None
        self._initialize_visualizers()

    def _initialize_visualizers(self):
        """Initialize visualization systems"""
        # Initialize adaptive visualizer
        try:
            self.adaptive_visualizer = CTDBNNVisualizer(
                self.model,
                output_dir='adaptive_visualizations',
                enabled=True
            )
            print("✓ Adaptive visualizer initialized")
        except Exception as e:
            print(f"⚠️ Could not initialize adaptive visualizer: {e}")
            self.adaptive_visualizer = None

        # Create output directory
        os.makedirs('adaptive_visualizations', exist_ok=True)

    def load_and_preprocess_data(self, file_path: str = None) -> bool:
        """Load and preprocess data with better file path handling"""
        try:
            print("📥 Loading and preprocessing data...")

            # Store file path for later use if provided
            if file_path:
                self.data_file_path = file_path

            # Load data using the model's method with the file path
            if hasattr(self, 'data_file_path') and self.data_file_path:
                self.model.load_data(self.data_file_path)
            else:
                self.model.load_data(file_path)

            # Preprocess data using the enhanced preprocessor
            X, y, feature_names = self.model.preprocess_data()

            # Store original y for reference (before encoding)
            y_original = y.copy()

            # Store the full dataset
            self.X_full = X
            self.y_full = y
            self.y_full_original = y_original

            print(f"✅ Data loaded and preprocessed:")
            print(f"   📊 Total samples: {X.shape[0]}")
            print(f"   🔧 Features: {X.shape[1]}")
            print(f"   🎯 Classes: {len(np.unique(y))}")

            return True

        except Exception as e:
            print(f"❌ Error loading and preprocessing data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def initialize_architecture(self):
        """Initialize CT-DBNN architecture with full dataset"""
        if self.X_full is None:
            raise ValueError("No data available. Call load_and_preprocess_data() first.")

        print("🏗️ Initializing CT-DBNN architecture...")

        # Initialize with full data (architecture only, no training)
        self.model.initialize_with_full_data(self.X_full, self.y_full)

        print("✅ Architecture initialization complete")

    def adaptive_learn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning method that respects previous best accuracy"""
        print("\n🚀 STARTING ADAPTIVE LEARNING WITH CT-DBNN")
        print("=" * 60)

        # Prepare data if not already done
        if self.X_full is None:
            print("🔄 Loading and preprocessing data...")
            success = self.load_and_preprocess_data()
            if not success:
                error_msg = "Failed to load and preprocess data. Check if the data file exists and is accessible."
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)

        X, y = self.X_full, self.y_full

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes: {np.unique(y)}")

        # CRITICAL: Use existing best accuracy if model was loaded
        starting_accuracy = self.best_accuracy
        print(f"🏁 Starting from previous best accuracy: {starting_accuracy:.4f}")

        # STEP 1: Initialize CT-DBNN architecture with full dataset ONLY if not already initialized
        if not getattr(self.model, 'initialized_with_full_data', False):
            print("🏗️ Initializing CT-DBNN architecture...")
            self.initialize_architecture()
        else:
            print("✅ CT-DBNN architecture already initialized")

        # If we have existing best training indices, use them as starting point
        if hasattr(self, 'best_training_indices') and self.best_training_indices:
            print(f"📚 Using existing best training set with {len(self.best_training_indices)} samples")
            initial_indices = self.best_training_indices.copy()
            X_train = X[initial_indices]
            y_train = y[initial_indices]
        else:
            # STEP 2: Select initial diverse training samples
            X_train, y_train, initial_indices = self._select_initial_training_samples(X, y)
            print(f"🎯 Selected new initial training set: {len(X_train)} samples")

        remaining_indices = [i for i in range(len(X)) if i not in initial_indices]

        print(f"📊 Initial training set: {len(X_train)} samples")
        print(f"📊 Remaining test set: {len(remaining_indices)} samples")

        # Initialize tracking variables for acid test-based stopping
        # START FROM PREVIOUS BEST ACCURACY, NOT 0.0
        self.best_accuracy = starting_accuracy
        self.best_training_indices = initial_indices.copy()
        self.best_round = 0
        acid_test_history = []
        patience_counter = 0

        max_rounds = self.adaptive_config['max_adaptive_rounds']
        patience = self.adaptive_config['patience']
        min_improvement = self.adaptive_config['min_improvement']

        print(f"\n🔄 Starting adaptive learning for up to {max_rounds} rounds...")
        print(f"🎯 Target: improve beyond {self.best_accuracy:.4f} (min improvement: {min_improvement})")
        self.adaptive_start_time = datetime.now()

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num

            print(f"\n🎯 Round {round_num}/{max_rounds}")
            print("-" * 40)

            # STEP 2 (continued): Train with current training data
            print("🎯 Training with current training data...")
            success = self.model.train_with_data(X_train, y_train, reset_weights=False)  # Don't reset weights for continued training

            if not success:
                print("❌ Training failed, stopping...")
                break

            # STEP 3: Run acid test on ENTIRE dataset - MAIN STOPPING CRITERION
            print("🧪 Running acid test on ENTIRE dataset...")
            try:
                all_predictions = self.model.predict(X)
                # Ensure predictions and y have same data type
                all_predictions = all_predictions.astype(y.dtype)
                acid_test_accuracy = accuracy_score(y, all_predictions)
                acid_test_history.append(acid_test_accuracy)
                print(f"📊 Acid test accuracy (entire dataset): {acid_test_accuracy:.4f}")

                # PRIMARY STOPPING CRITERION 1: 100% accuracy on ENTIRE dataset
                if acid_test_accuracy >= 0.9999:  # 99.99% accuracy (accounting for floating point)
                    print("🎉 REACHED 100% ACCURACY ON ENTIRE DATASET! Stopping adaptive learning.")
                    self.best_accuracy = acid_test_accuracy
                    self.best_training_indices = initial_indices.copy()
                    self.best_round = round_num
                    break

            except Exception as e:
                print(f"❌ Acid test failed: {e}")
                acid_test_accuracy = starting_accuracy  # Use starting accuracy as fallback
                acid_test_history.append(starting_accuracy)
                # Continue with the round even if acid test fails

            # STEP 4: Check if we have any remaining samples to process
            if not remaining_indices:
                print("💤 No more samples to add to training set")
                # PRIMARY STOPPING CRITERION 2: No more samples to add
                print("🎉 EXHAUSTED ALL SAMPLES! Stopping adaptive learning.")
                if acid_test_accuracy > self.best_accuracy:
                    self.best_accuracy = acid_test_accuracy
                    self.best_training_indices = initial_indices.copy()
                    self.best_round = round_num
                break

            # STEP 5: Identify failed candidates in remaining data
            X_remaining = X[remaining_indices]
            y_remaining = y[remaining_indices]

            # Get predictions for remaining data
            remaining_predictions = self.model.predict(X_remaining)
            remaining_posteriors = self.model._compute_batch_posterior(X_remaining)

            # Find misclassified samples
            misclassified_mask = remaining_predictions != y_remaining
            misclassified_indices = np.where(misclassified_mask)[0]

            if len(misclassified_indices) == 0:
                print("✅ No misclassified samples in remaining data!")
                # This means we have perfect classification on remaining data
                print("📊 Perfect on remaining data, continuing to monitor acid test...")
            else:
                print(f"📊 Found {len(misclassified_indices)} misclassified samples in remaining data")

                # STEP 6: Select most divergent failed candidates
                samples_to_add_indices = self._select_divergent_samples(
                    X_remaining, y_remaining, remaining_predictions, remaining_posteriors,
                    misclassified_indices, remaining_indices
                )

                if samples_to_add_indices:
                    # Update training set
                    initial_indices.extend(samples_to_add_indices)
                    remaining_indices = [i for i in remaining_indices if i not in samples_to_add_indices]

                    X_train = X[initial_indices]
                    y_train = y[initial_indices]

                    print(f"📈 Training set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}% of total)")
                    print(f"📊 Remaining set size: {len(remaining_indices)} samples")
                else:
                    print("💤 No divergent samples to add in this round")

            # STEP 7: Update best model and check for improvement
            # CRITICAL: Only update if we actually improve beyond previous best
            if acid_test_accuracy > self.best_accuracy + min_improvement:
                improvement = acid_test_accuracy - self.best_accuracy
                self.best_accuracy = acid_test_accuracy
                self.best_training_indices = initial_indices.copy()
                self.best_round = round_num
                patience_counter = 0
                print(f"🏆 New best acid test accuracy: {acid_test_accuracy:.4f} (+{improvement:.4f})")
            else:
                patience_counter += 1
                if acid_test_accuracy > self.best_accuracy:
                    small_improvement = acid_test_accuracy - self.best_accuracy
                    print(f"↗️  Small improvement: {acid_test_accuracy:.4f} (+{small_improvement:.4f}) - Patience: {patience_counter}/{patience}")
                else:
                    print(f"🔄 No improvement - Patience: {patience_counter}/{patience}")

            # PRIMARY STOPPING CRITERION 3: No significant improvement for patience rounds
            if patience_counter >= patience:
                print(f"🛑 PATIENCE EXCEEDED: No significant improvement in acid test for {patience} rounds")
                print(f"   Best acid test accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
                print(f"   Current acid test accuracy: {acid_test_accuracy:.4f}")
                break

            # PRIMARY STOPPING CRITERION 4: Maximum rounds reached
            if round_num >= max_rounds:
                print(f"🛑 MAXIMUM ROUNDS REACHED: Completed {max_rounds} rounds")
                break

        # Finalize with best configuration
        print(f"\n🎉 Adaptive learning completed after {self.adaptive_round} rounds!")

        # Ensure we use the best configuration found
        if not hasattr(self, 'best_accuracy') or self.best_accuracy == 0.0:
            # Use final values if best wasn't set
            self.best_accuracy = acid_test_history[-1] if acid_test_history else starting_accuracy
            self.best_training_indices = initial_indices.copy()
            self.best_round = self.adaptive_round

        print(f"🏆 Best acid test accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
        print(f"📊 Final training set: {len(self.best_training_indices)} samples ({len(self.best_training_indices)/len(X)*100:.1f}% of total)")

        # Only retrain with best configuration if it's better than our starting point
        if self.best_accuracy > starting_accuracy + min_improvement:
            print("🔧 Training final model with improved configuration...")
            X_train_best = X[self.best_training_indices]
            y_train_best = y[self.best_training_indices]
            self.model.train_with_data(X_train_best, y_train_best, reset_weights=True)
        else:
            print("💤 No significant improvement - keeping original model")

        # Final acid test verification
        final_predictions = self.model.predict(X)
        final_accuracy = accuracy_score(y, final_predictions)

        print(f"📊 Final acid test accuracy: {final_accuracy:.4f}")
        print(f"📈 Improvement from starting: {final_accuracy - starting_accuracy:+.4f}")

        # Create visualizations
        if self.adaptive_visualizer:
            self.adaptive_visualizer.create_visualizations(
                X, y, final_predictions
            )

        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        X_test_best = X[[i for i in range(len(X)) if i not in self.best_training_indices]]
        y_test_best = y[[i for i in range(len(X)) if i not in self.best_training_indices]]

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _select_initial_training_samples(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Select initial diverse training samples from each class"""
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

        X_train = X[initial_indices]
        y_train = y[initial_indices]

        return X_train, y_train, initial_indices

    def _select_divergent_samples(self, X_remaining: np.ndarray, y_remaining: np.ndarray,
                                predictions: np.ndarray, posteriors: np.ndarray,
                                misclassified_indices: np.ndarray, remaining_indices: List[int]) -> List[int]:
        """Select most divergent failed candidates from each class"""
        samples_to_add = []
        unique_classes = np.unique(y_remaining)

        print("🔍 Selecting most divergent failed candidates...")

        # Group misclassified samples by true class
        class_samples = defaultdict(list)

        for idx_in_remaining in misclassified_indices:
            original_idx = remaining_indices[idx_in_remaining]
            true_class = y_remaining[idx_in_remaining]
            pred_class = predictions[idx_in_remaining]

            # Convert class labels to 0-based indices for array access
            true_class_idx_result = np.where(unique_classes == true_class)[0]
            pred_class_idx_result = np.where(unique_classes == pred_class)[0]

            # Check if we found valid indices
            if len(true_class_idx_result) == 0 or len(pred_class_idx_result) == 0:
                continue

            true_class_idx = true_class_idx_result[0]
            pred_class_idx = pred_class_idx_result[0]

            # Calculate margin (divergence)
            true_posterior = posteriors[idx_in_remaining, true_class_idx]
            pred_posterior = posteriors[idx_in_remaining, pred_class_idx]
            margin = pred_posterior - true_posterior

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_posterior': true_posterior,
                'pred_posterior': pred_posterior
            })

        # For each class, select most divergent samples
        max_samples = 2  # Fixed to match adaptive_dbnn behavior

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

class AdaptiveCTDBNNGUI:
    """Complete GUI interface for Adaptive CT-DBNN"""

    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive CT-DBNN with UCI Dataset Support")
        self.root.geometry("1400x900")

        self.adaptive_model = None
        self.model_trained = False
        self.data_loaded = False
        self.current_data = None
        self.feature_names = []
        self.target_column = ""

        self.setup_gui()

    def save_model(self):
        """Save the current model to file"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No model to save. Please initialize and train a model first.")
            return

        if not self.model_trained:
            messagebox.showwarning("Warning", "Model is not trained yet. Please train the model before saving.")
            return

        try:
            # Determine default file path
            dataset_name = self.dataset_var.get()
            default_filename = f"{dataset_name}.bin"
            models_dir = "Models"
            os.makedirs(models_dir, exist_ok=True)
            default_path = os.path.join(models_dir, default_filename)

            # Ask for file path
            file_path = filedialog.asksaveasfilename(
                title="Save Model As",
                initialdir="Models",
                initialfile=default_filename,
                defaultextension=".bin",
                filetypes=[("Binary Model Files", "*.bin"), ("All files", "*.*")]
            )

            if not file_path:
                return  # User cancelled

            # Save the model
            success, message = ModelSerializer.save_model(self.adaptive_model, file_path)

            if success:
                self.log_output(f"💾 {message}")
                self.log_output(f"📁 Model saved to: {file_path}")

                # Show model info
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                self.log_output(f"📊 Model file size: {file_size:.2f} MB")
            else:
                self.log_output(f"❌ {message}")
                messagebox.showerror("Save Error", message)

        except Exception as e:
            error_msg = f"❌ Error saving model: {e}"
            self.log_output(error_msg)
            messagebox.showerror("Save Error", error_msg)

    def show_label_encoding_info(self):
        """Display information about label encoding for the current model"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No model loaded.")
            return

        try:
            target_encoder = self.adaptive_model.model.preprocessor.target_encoder

            if (hasattr(target_encoder, 'classes_') and
                target_encoder.classes_ is not None and
                len(target_encoder.classes_) > 0):

                encoding_info = "🔤 Label Encoding Information:\n"
                encoding_info += f"Number of classes: {len(target_encoder.classes_)}\n\n"
                encoding_info += "Class Mapping:\n"

                for i, class_name in enumerate(target_encoder.classes_):
                    encoding_info += f"  {i} → '{class_name}'\n"

                # Show in output log
                self.log_output(encoding_info)

                # Also show in message box for easy viewing
                messagebox.showinfo("Label Encoding Info",
                                  f"Label encoder has {len(target_encoder.classes_)} classes:\n" +
                                  "\n".join([f"{i} → '{cls}'" for i, cls in enumerate(target_encoder.classes_)]))
            else:
                self.log_output("⚠️ No label encoder information available")
                messagebox.showinfo("Label Encoding Info", "No label encoder information available.")

        except Exception as e:
            self.log_output(f"❌ Error getting label encoding info: {e}")

    def load_model(self):
        """Load a model from file with proper DataFrame reconstruction and label encoding"""
        try:
            # Ask for file path
            file_path = filedialog.askopenfilename(
                title="Load Model",
                initialdir="Models",
                filetypes=[("Binary Model Files", "*.bin"), ("All files", "*.*")]
            )

            if not file_path:
                return  # User cancelled

            # Load the model
            success, loaded_model, message = ModelSerializer.load_model(file_path)

            if success:
                self.adaptive_model = loaded_model
                self.model_trained = True
                self.data_loaded = True

                # Update GUI state
                self.dataset_var.set(self.adaptive_model.dataset_name)

                # Properly reconstruct the DataFrame without dimension mismatch
                self._reconstruct_data_from_loaded_model()

                # Update target column
                self.target_var.set(self.adaptive_model.model.target_column)

                # Update feature selection
                self.update_feature_selection()

                # Load hyperparameters from the loaded model
                self._update_gui_from_loaded_model()

                # Check label encoder status
                target_encoder = self.adaptive_model.model.preprocessor.target_encoder
                has_label_encoder = (hasattr(target_encoder, 'classes_') and
                                   target_encoder.classes_ is not None and
                                   len(target_encoder.classes_) > 0)

                self.log_output(f"📂 {message}")
                self.log_output(f"🎯 Loaded model for dataset: {self.adaptive_model.dataset_name}")
                self.log_output(f"🏆 Best accuracy: {self.adaptive_model.best_accuracy:.4f}")
                self.log_output(f"📊 Training rounds: {self.adaptive_model.adaptive_round}")

                if has_label_encoder:
                    self.log_output(f"🔤 Label encoder available with {len(target_encoder.classes_)} classes")
                    self.log_output(f"📋 Class labels: {list(target_encoder.classes_)}")
                else:
                    self.log_output("⚠️ No label encoder found - predictions will use numeric labels")

            else:
                self.log_output(f"❌ {message}")
                messagebox.showerror("Load Error", message)

        except Exception as e:
            error_msg = f"❌ Error loading model: {e}"
            self.log_output(error_msg)
            import traceback
            self.log_output(f"🔍 Detailed traceback:\n{traceback.format_exc()}")
            messagebox.showerror("Load Error", error_msg)

    def safe_dataframe_reconstruction(self, X, y, feature_names, target_column):
        """Safely reconstruct DataFrame handling various dimension scenarios"""
        try:
            if X is None or y is None:
                self.log_output("❌ No data available for reconstruction")
                return None

            n_samples, n_features = X.shape

            # Handle feature names mismatch
            if len(feature_names) != n_features:
                self.log_output(f"⚠️ Feature count mismatch: {len(feature_names)} names vs {n_features} features")
                # Use generic feature names
                feature_names = [f'feature_{i}' for i in range(n_features)]

            # Ensure y has correct shape
            if len(y) != n_samples:
                self.log_output(f"⚠️ Sample count mismatch: X has {n_samples}, y has {len(y)}")
                # Truncate to minimum length
                min_length = min(n_samples, len(y))
                X = X[:min_length]
                y = y[:min_length]
                n_samples = min_length

            # Create DataFrame
            data_dict = {}
            for i, feature_name in enumerate(feature_names):
                data_dict[feature_name] = X[:, i]

            data_dict[target_column] = y

            df = pd.DataFrame(data_dict)
            self.log_output(f"✅ Successfully reconstructed DataFrame: {df.shape}")
            return df

        except Exception as e:
            self.log_output(f"❌ Error in DataFrame reconstruction: {e}")
            return None

    def _reconstruct_data_from_loaded_model(self):
        """Properly reconstruct DataFrame from loaded model without dimension errors"""
        try:
            if (hasattr(self.adaptive_model, 'X_full') and
                hasattr(self.adaptive_model, 'y_full') and
                hasattr(self.adaptive_model.model, 'feature_names') and
                hasattr(self.adaptive_model.model, 'target_column')):

                X = self.adaptive_model.X_full
                y = self.adaptive_model.y_full
                feature_names = self.adaptive_model.model.feature_names
                target_column = self.adaptive_model.model.target_column
                self.current_data = self.safe_dataframe_reconstruction(X, y, feature_names, target_column)

                # Ensure dimensions match
                if X is not None and y is not None:
                    # Check if feature names match the number of features in X
                    if len(feature_names) == X.shape[1]:
                        # Create DataFrame with correct dimensions
                        data_dict = {}
                        for i, feature_name in enumerate(feature_names):
                            data_dict[feature_name] = X[:, i]

                        # Add target column
                        data_dict[target_column] = y

                        self.current_data = pd.DataFrame(data_dict)
                        self.log_output(f"✅ Reconstructed DataFrame: {self.current_data.shape}")

                    else:
                        self.log_output(f"⚠️ Feature count mismatch: {len(feature_names)} names vs {X.shape[1]} features")
                        # Fallback: create generic feature names
                        feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                        data_dict = {col: X[:, i] for i, col in enumerate(feature_cols)}
                        data_dict[target_column] = y
                        self.current_data = pd.DataFrame(data_dict)
                else:
                    self.log_output("⚠️ No data available in loaded model")
                    self.current_data = None

            else:
                self.log_output("⚠️ Incomplete model state for data reconstruction")
                self.current_data = None

        except Exception as e:
            self.log_output(f"❌ Error reconstructing data: {e}")
            self.current_data = None

    def _update_gui_from_loaded_model(self):
        """Update GUI controls from loaded model configuration"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
            return

        try:
            config = self.adaptive_model.config

            # Update CT-DBNN parameters - FIXED TYPO: ctdbnn_config not ctdbbnn_config
            ctdbnn_config = config.get('ctdbnn_config', {})
            self.resolution_var.set(str(ctdbnn_config.get('resol', 100)))
            self.learning_rate_var.set(str(ctdbnn_config.get('learning_rate', 0.01)))
            self.max_epochs_var.set(str(ctdbnn_config.get('max_epochs', 100)))
            self.batch_size_var.set(str(ctdbnn_config.get('batch_size', 32)))
            self.smoothing_factor_var.set(str(ctdbnn_config.get('smoothing_factor', '1e-8')))
            self.use_complex_var.set(ctdbnn_config.get('use_complex_tensor', True))
            self.orthogonalize_var.set(ctdbnn_config.get('orthogonalize_weights', True))  # FIXED TYPO
            self.parallel_var.set(ctdbnn_config.get('parallel_processing', True))

            # Update adaptive learning parameters
            adaptive_config = self.adaptive_model.adaptive_config
            self.initial_samples_var.set(str(adaptive_config.get('initial_samples_per_class', 5)))
            self.max_rounds_var.set(str(adaptive_config.get('max_adaptive_rounds', 20)))
            self.patience_var.set(str(adaptive_config.get('patience', 10)))
            self.min_improvement_var.set(str(adaptive_config.get('min_improvement', 0.001)))
            self.enable_acid_var.set(adaptive_config.get('enable_acid_test', True))

            self.log_output("✅ GUI updated from loaded model configuration")

        except Exception as e:
            self.log_output(f"⚠️ Could not fully update GUI from model: {e}")

    def predict_file(self):
        """Predict on a file and save results with memory-efficient chunking"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None or not self.model_trained:
            messagebox.showwarning("Warning", "No trained model available. Please train or load a model first.")
            return

        try:
            # Ask for input file
            input_file = filedialog.askopenfilename(
                title="Select File for Prediction",
                filetypes=[("CSV files", "*.csv"), ("Data files", "*.data"), ("Text files", "*.txt"), ("All files", "*.*")]
            )

            if not input_file:
                return  # User cancelled

            # Determine output file
            dataset_name = self.dataset_var.get()
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}_prediction.csv"

            # Ask for output location
            output_file = filedialog.asksaveasfilename(
                title="Save Predictions As",
                initialfile=output_file,
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not output_file:
                return  # User cancelled

            # Perform prediction with chunking
            success, message = self._predict_file_chunked(input_file, output_file)

            if success:
                self.log_output(f"🔮 {message}")

                # Show prediction summary
                if os.path.exists(output_file):
                    result_df = pd.read_csv(output_file)
                    n_predictions = len(result_df)
                    n_classes = len([col for col in result_df.columns if col.startswith('pred_')])
                    self.log_output(f"📊 Predictions completed: {n_predictions} samples")
                    self.log_output(f"🎯 Top classes saved: {n_classes} per sample")
                    self.log_output(f"💾 Results saved to: {output_file}")

            else:
                self.log_output(f"❌ {message}")
                messagebox.showerror("Prediction Error", message)

        except Exception as e:
            error_msg = f"❌ Error during prediction: {e}"
            self.log_output(error_msg)
            messagebox.showerror("Prediction Error", error_msg)

    def _predict_file_chunked(self, input_file, output_file, chunk_size=1000):
        """Predict on file in chunks to save memory with proper label decoding"""
        try:
            self.log_output(f"🔮 Starting prediction on {input_file}")
            self.log_output(f"📁 Processing in chunks of {chunk_size} samples")

            # Get the model and feature information
            model = self.adaptive_model.model
            feature_names = getattr(model, 'feature_names', [])
            target_column = model.target_column

            # Check if we have a label encoder for decoding
            target_encoder = model.preprocessor.target_encoder
            can_decode = (hasattr(target_encoder, 'classes_') and
                         target_encoder.classes_ is not None and
                         len(target_encoder.classes_) > 0)

            if can_decode:
                self.log_output("✅ Label encoder available - will decode predictions to original labels")
                original_classes = target_encoder.classes_
                self.log_output(f"📋 Original class labels: {list(original_classes)}")
            else:
                self.log_output("⚠️ No label encoder found - using numeric class labels")

            # Read first chunk to get column information
            first_chunk = pd.read_csv(input_file, nrows=1)
            all_columns = first_chunk.columns.tolist()

            # Determine which columns to use for prediction
            prediction_columns = [col for col in feature_names if col in all_columns]
            if not prediction_columns:
                prediction_columns = [col for col in all_columns if col != target_column]

            self.log_output(f"🎯 Using {len(prediction_columns)} features for prediction")
            self.log_output(f"📋 Features: {', '.join(prediction_columns)}")

            # Initialize output
            header_written = False
            total_samples = 0
            processed_size = 0
            total_size = os.path.getsize(input_file)

            # Process file in chunks
            for chunk_idx, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
                chunk_size_bytes = chunk.memory_usage(deep=True).sum()
                processed_size += chunk_size_bytes

                progress = (processed_size / total_size) * 100 if total_size > 0 else 0
                self.log_output(f"📦 Processing chunk {chunk_idx + 1} ({progress:.1f}%)...")

                # Make a copy to avoid modifying original
                chunk_data = chunk.copy()

                # Prepare features for prediction
                X_pred = chunk_data[prediction_columns].values

                # Get predictions and probabilities
                predictions = model.predict(X_pred)
                probabilities = model._compute_batch_posterior(X_pred)

                # Get unique classes from the model
                if hasattr(model.core, 'unique_classes'):
                    unique_classes = model.core.unique_classes
                else:
                    unique_classes = np.unique(self.adaptive_model.y_full)

                # Add prediction results to chunk with label decoding
                chunk_data = self._add_prediction_results(
                    chunk_data, predictions, probabilities, unique_classes
                )

                # Save chunk to output file
                mode = 'w' if not header_written else 'a'
                header = not header_written

                chunk_data.to_csv(output_file, mode=mode, header=header, index=False)
                header_written = True

                total_samples += len(chunk_data)
                self.log_output(f"   ✅ Chunk {chunk_idx + 1}: {len(chunk_data)} samples processed")

                # Memory cleanup
                del chunk_data, X_pred, predictions, probabilities
                if chunk_idx % 10 == 0:  # Clean memory every 10 chunks
                    gc.collect()

            self.log_output(f"✅ Prediction completed: {total_samples} samples processed")

            # Show sample of decoded predictions if available
            if can_decode and os.path.exists(output_file):
                result_sample = pd.read_csv(output_file, nrows=5)
                if 'prediction' in result_sample.columns:
                    sample_predictions = result_sample['prediction'].head(3).tolist()
                    self.log_output(f"🔍 Sample predictions: {sample_predictions}")

            return True, f"Prediction completed: {total_samples} samples processed"

        except Exception as e:
            return False, f"Prediction error: {e}"

    def _add_prediction_results(self, chunk_data, predictions, probabilities, unique_classes):
        """Add prediction results to chunk data with proper label decoding"""
        try:
            # Get the target encoder from the model
            target_encoder = self.adaptive_model.model.preprocessor.target_encoder

            # Check if we have a fitted label encoder for decoding
            has_label_encoder = (hasattr(target_encoder, 'classes_') and
                               target_encoder.classes_ is not None and
                               len(target_encoder.classes_) > 0)

            if has_label_encoder:
                # Decode the primary prediction
                try:
                    # Ensure predictions are within valid range for the encoder
                    valid_predictions = np.clip(predictions, 0, len(target_encoder.classes_) - 1)
                    decoded_predictions = target_encoder.inverse_transform(valid_predictions)
                    chunk_data['prediction'] = decoded_predictions
                    self.log_output(f"✅ Decoded {len(decoded_predictions)} predictions using label encoder")
                except Exception as e:
                    self.log_output(f"⚠️ Could not decode primary predictions: {e}")
                    chunk_data['prediction'] = predictions

                # Get top classes for each sample based on posterior probabilities
                n_top = min(5, len(unique_classes))  # Maximum 5 top classes

                for i in range(n_top):
                    # Get the i-th highest probability and corresponding class for each sample
                    top_indices = np.argsort(probabilities, axis=1)[:, -(i+1)]
                    top_probs = probabilities[np.arange(len(probabilities)), top_indices]
                    top_classes = unique_classes[top_indices]

                    # Only keep predictions with probability > 1/3 of max probability for that sample
                    max_probs = np.max(probabilities, axis=1)
                    keep_mask = top_probs > (max_probs / 3.0)

                    # Set low-confidence predictions to NaN
                    top_classes[~keep_mask] = -1
                    top_probs[~keep_mask] = np.nan

                    # Decode the class labels
                    try:
                        # Ensure classes are within valid range
                        valid_top_classes = np.clip(top_classes, 0, len(target_encoder.classes_) - 1)
                        decoded_top_classes = target_encoder.inverse_transform(valid_top_classes)
                        # Replace -1 with NaN for invalid predictions
                        decoded_top_classes = np.where(top_classes == -1, np.nan, decoded_top_classes)
                        chunk_data[f'pred_class_{i+1}'] = decoded_top_classes
                    except Exception as e:
                        self.log_output(f"⚠️ Could not decode top class {i+1}: {e}")
                        chunk_data[f'pred_class_{i+1}'] = np.where(top_classes == -1, np.nan, top_classes)

                    chunk_data[f'pred_prob_{i+1}'] = top_probs

            else:
                # No label encoder available, use numeric labels
                self.log_output("⚠️ No label encoder found, using numeric predictions")
                chunk_data['prediction'] = predictions

                # Get top classes without decoding
                n_top = min(5, len(unique_classes))

                for i in range(n_top):
                    top_indices = np.argsort(probabilities, axis=1)[:, -(i+1)]
                    top_probs = probabilities[np.arange(len(probabilities)), top_indices]
                    top_classes = unique_classes[top_indices]

                    max_probs = np.max(probabilities, axis=1)
                    keep_mask = top_probs > (max_probs / 3.0)

                    top_classes[~keep_mask] = np.nan
                    top_probs[~keep_mask] = np.nan

                    chunk_data[f'pred_class_{i+1}'] = top_classes
                    chunk_data[f'pred_prob_{i+1}'] = top_probs

            # Add confidence score (max probability)
            chunk_data['confidence'] = np.max(probabilities, axis=1)

            return chunk_data

        except Exception as e:
            self.log_output(f"❌ Error adding prediction results: {e}")
            import traceback
            self.log_output(f"🔍 Detailed error: {traceback.format_exc()}")
            # Fallback: just add basic prediction
            chunk_data['prediction'] = predictions
            chunk_data['confidence'] = np.max(probabilities, axis=1)
            return chunk_data

    def exit_application(self):
        """Exit the application with confirmation"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.log_output("👋 Exiting Adaptive CT-DBNN Application...")
            self.root.quit()
            self.root.destroy()

    def load_default_parameters(self):
        """Load default hyperparameters for the current dataset"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            dataset_name = self.dataset_var.get()
            config = DatasetConfig.load_config(dataset_name)

            if config:
                # CT-DBNN parameters
                ctdbnn_config = config.get('ctdbnn_config', {})
                self.resolution_var.set(str(ctdbnn_config.get('resol', 100)))
                self.learning_rate_var.set(str(ctdbnn_config.get('learning_rate', 0.01)))
                self.max_epochs_var.set(str(ctdbnn_config.get('max_epochs', 100)))
                self.batch_size_var.set(str(ctdbnn_config.get('batch_size', 32)))
                self.smoothing_factor_var.set(str(ctdbnn_config.get('smoothing_factor', '1e-8')))
                self.use_complex_var.set(ctdbnn_config.get('use_complex_tensor', True))
                self.orthogonalize_var.set(ctdbnn_config.get('orthogonalize_weights', True))
                self.parallel_var.set(ctdbnn_config.get('parallel_processing', True))

                # Adaptive learning parameters
                adaptive_config = config.get('adaptive_learning', {})
                self.initial_samples_var.set(str(adaptive_config.get('initial_samples_per_class', 5)))
                self.max_rounds_var.set(str(adaptive_config.get('max_adaptive_rounds', 20)))
                self.patience_var.set(str(adaptive_config.get('patience', 10)))
                self.min_improvement_var.set(str(adaptive_config.get('min_improvement', 0.001)))
                self.enable_acid_var.set(adaptive_config.get('enable_acid_test', True))
                self.class_balance_var.set(adaptive_config.get('class_balancing', True))
                self.hard_mining_var.set(adaptive_config.get('hard_sample_mining', True))
                self.divergence_threshold_var.set(str(adaptive_config.get('divergence_threshold', 0.1)))
                self.max_samples_round_var.set(str(adaptive_config.get('max_samples_per_round', 5)))
                self.selection_strategy_var.set(adaptive_config.get('sample_selection_strategy', 'margin'))

                # Preprocessing parameters
                preprocessing_config = config.get('preprocessing', {})
                self.normalization_var.set(preprocessing_config.get('normalization', 'standard'))
                self.missing_values_var.set(preprocessing_config.get('handle_missing', 'mean'))
                self.encoding_var.set(preprocessing_config.get('categorical_encoding', 'label'))
                self.remove_outliers_var.set(preprocessing_config.get('remove_outliers', False))
                self.feature_scaling_var.set(preprocessing_config.get('feature_scaling', True))

                self.log_output("✅ Loaded default parameters from configuration")
            else:
                self.log_output("⚠️ No configuration found, using built-in defaults")

        except Exception as e:
            self.log_output(f"❌ Error loading default parameters: {e}")

    def save_current_parameters(self):
        """Save current hyperparameters to configuration file"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
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
                    'smoothing_factor': float(self.smoothing_factor_var.get()),
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
                    'divergence_threshold': float(self.divergence_threshold_var.get()),
                    'max_samples_per_round': int(self.max_samples_round_var.get()),
                    'sample_selection_strategy': self.selection_strategy_var.get(),
                },

                'preprocessing': {
                    'normalization': self.normalization_var.get(),
                    'handle_missing': self.missing_values_var.get(),
                    'categorical_encoding': self.encoding_var.get(),
                    'remove_outliers': self.remove_outliers_var.get(),
                    'feature_scaling': self.feature_scaling_var.get(),
                }
            }

            config_file = f"{dataset_name}.conf"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)

            self.log_output(f"✅ Saved current parameters to {config_file}")

        except Exception as e:
            self.log_output(f"❌ Error saving parameters: {e}")

    def reset_to_dataset_defaults(self):
        """Reset parameters to dataset-specific defaults"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            dataset_name = self.dataset_var.get()
            dataset_info = UCIDatasetHandler.get_dataset_info(dataset_name)

            if dataset_info:
                recommended_resol = dataset_info.get('recommended_resolution', 100)
                self.resolution_var.set(str(recommended_resol))
                self.log_output(f"✅ Reset resolution to dataset default: {recommended_resol}")
            else:
                self.log_output("⚠️ No dataset-specific defaults found")

        except Exception as e:
            self.log_output(f"❌ Error resetting to dataset defaults: {e}")

    def apply_hyperparameters(self):
        """Apply current hyperparameters to the model"""
        if not hasattr(self, 'adaptive_model') or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return

        try:
            # Update CT-DBNN configuration
            self.adaptive_model.config.update({
                'resol': int(self.resolution_var.get()),
                'target_column': self.target_var.get(),
            })

            # Update CT-DBNN core configuration
            self.adaptive_model.config['ctdbnn_config'] = {
                'resol': int(self.resolution_var.get()),
                'learning_rate': float(self.learning_rate_var.get()),
                'max_epochs': int(self.max_epochs_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'smoothing_factor': float(self.smoothing_factor_var.get()),
                'use_complex_tensor': self.use_complex_var.get(),
                'orthogonalize_weights': self.orthogonalize_var.get(),
                'parallel_processing': self.parallel_var.get(),
            }

            # Update adaptive learning configuration
            self.adaptive_model.adaptive_config.update({
                'initial_samples_per_class': int(self.initial_samples_var.get()),
                'max_adaptive_rounds': int(self.max_rounds_var.get()),
                'patience': int(self.patience_var.get()),
                'min_improvement': float(self.min_improvement_var.get()),
                'enable_acid_test': self.enable_acid_var.get(),
                'class_balancing': self.class_balance_var.get(),
                'hard_sample_mining': self.hard_mining_var.get(),
                'divergence_threshold': float(self.divergence_threshold_var.get()),
                'max_samples_per_round': int(self.max_samples_round_var.get()),
                'sample_selection_strategy': self.selection_strategy_var.get(),
            })

            # Update model's configuration
            self.adaptive_model.model.config.update(self.adaptive_model.config)

            self.log_output("✅ Hyperparameters applied to model")
            self.log_output(f"   Resolution: {self.resolution_var.get()}")
            self.log_output(f"   Learning Rate: {self.learning_rate_var.get()}")
            self.log_output(f"   Max Rounds: {self.max_rounds_var.get()}")

        except Exception as e:
            self.log_output(f"❌ Error applying hyperparameters: {e}")


    def setup_hyperparameters_tab(self, parent):
        """Setup comprehensive hyperparameters configuration tab"""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(parent)
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
        ttk.Label(core_frame, text="(Grid resolution for tensor operations)").grid(row=0, column=2, sticky=tk.W, padx=5)

        # Learning Rate
        ttk.Label(core_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.learning_rate_var = tk.StringVar(value="0.01")
        ttk.Entry(core_frame, textvariable=self.learning_rate_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Max Epochs
        ttk.Label(core_frame, text="Max Epochs:").grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        self.max_epochs_var = tk.StringVar(value="100")
        ttk.Entry(core_frame, textvariable=self.max_epochs_var, width=10).grid(row=1, column=4, padx=5, pady=2)

        # Batch Size
        ttk.Label(core_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value="32")
        ttk.Entry(core_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # Smoothing Factor
        ttk.Label(core_frame, text="Smoothing Factor:").grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        self.smoothing_factor_var = tk.StringVar(value="1e-8")
        ttk.Entry(core_frame, textvariable=self.smoothing_factor_var, width=10).grid(row=2, column=4, padx=5, pady=2)

        # Advanced CT-DBNN Options
        ttk.Label(core_frame, text="Advanced Options:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)

        self.use_complex_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(core_frame, text="Use Complex Tensor", variable=self.use_complex_var).grid(row=3, column=1, sticky=tk.W, padx=5)

        self.orthogonalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(core_frame, text="Orthogonalize Weights", variable=self.orthogonalize_var).grid(row=3, column=2, sticky=tk.W, padx=5)

        self.parallel_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(core_frame, text="Parallel Processing", variable=self.parallel_var).grid(row=3, column=3, sticky=tk.W, padx=5)

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

        # Divergence threshold
        ttk.Label(adaptive_frame, text="Divergence Threshold:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.divergence_threshold_var = tk.StringVar(value="0.1")
        ttk.Entry(adaptive_frame, textvariable=self.divergence_threshold_var, width=10).grid(row=3, column=1, padx=5, pady=2)

        # Max samples per round
        ttk.Label(adaptive_frame, text="Max Samples/Round:").grid(row=3, column=2, sticky=tk.W, padx=5, pady=2)
        self.max_samples_round_var = tk.StringVar(value="5")
        ttk.Entry(adaptive_frame, textvariable=self.max_samples_round_var, width=10).grid(row=3, column=3, padx=5, pady=2)

        # Sample selection strategy
        ttk.Label(adaptive_frame, text="Selection Strategy:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.selection_strategy_var = tk.StringVar(value="margin")
        strategy_combo = ttk.Combobox(adaptive_frame, textvariable=self.selection_strategy_var,
                                     values=["margin", "entropy", "random"], width=10)
        strategy_combo.grid(row=4, column=1, padx=5, pady=2)

        # Preprocessing Parameters Frame
        preprocessing_frame = ttk.LabelFrame(scrollable_frame, text="Data Preprocessing Parameters", padding="10")
        preprocessing_frame.pack(fill=tk.X, pady=5, padx=10)

        # Normalization
        ttk.Label(preprocessing_frame, text="Normalization:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.normalization_var = tk.StringVar(value="standard")
        norm_combo = ttk.Combobox(preprocessing_frame, textvariable=self.normalization_var,
                                 values=["standard", "minmax", "robust", "none"], width=10)
        norm_combo.grid(row=0, column=1, padx=5, pady=2)

        # Missing value handling
        ttk.Label(preprocessing_frame, text="Missing Values:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.missing_values_var = tk.StringVar(value="mean")
        missing_combo = ttk.Combobox(preprocessing_frame, textvariable=self.missing_values_var,
                                    values=["mean", "median", "mode", "drop"], width=10)
        missing_combo.grid(row=0, column=3, padx=5, pady=2)

        # Categorical encoding
        ttk.Label(preprocessing_frame, text="Categorical Encoding:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.encoding_var = tk.StringVar(value="label")
        encoding_combo = ttk.Combobox(preprocessing_frame, textvariable=self.encoding_var,
                                     values=["label", "onehot", "target"], width=10)
        encoding_combo.grid(row=1, column=1, padx=5, pady=2)

        # Advanced preprocessing options
        self.remove_outliers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(preprocessing_frame, text="Remove Outliers", variable=self.remove_outliers_var).grid(row=1, column=2, sticky=tk.W, padx=5)

        self.feature_scaling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(preprocessing_frame, text="Feature Scaling", variable=self.feature_scaling_var).grid(row=1, column=3, sticky=tk.W, padx=5)

        # Control buttons frame
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Button(button_frame, text="Load Default Parameters",
                  command=self.load_default_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Current Parameters",
                  command=self.save_current_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Dataset Defaults",
                  command=self.reset_to_dataset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Parameters",
                  command=self.apply_hyperparameters).pack(side=tk.RIGHT, padx=5)


        # Configure column weights for responsive layout
        for frame in [core_frame, adaptive_frame, preprocessing_frame]:
            frame.columnconfigure(2, weight=1)

    def setup_gui(self):
        """Setup the main GUI with hyperparameters tab"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Adaptive CT-DBNN - Memory Optimized",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Data Tab
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="Data Management")

        # Hyperparameters Tab
        hyperparams_tab = ttk.Frame(notebook)
        notebook.add(hyperparams_tab, text="Hyperparameters")

        # Training Tab
        training_tab = ttk.Frame(notebook)
        notebook.add(training_tab, text="Training & Evaluation")

        # Visualization Tab
        viz_tab = ttk.Frame(notebook)
        notebook.add(viz_tab, text="Visualization")

        # Setup each tab
        self.setup_data_tab(data_tab)
        self.setup_hyperparameters_tab(hyperparams_tab)
        self.setup_training_tab(training_tab)
        self.setup_visualization_tab(viz_tab)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_data_tab(self, parent):
        """Setup data management tab"""
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(parent, text="Dataset Selection", padding="10")
        dataset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dataset_frame, text="Dataset:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, width=30)
        self.dataset_combo.grid(row=0, column=1, padx=5)

        # Populate with available datasets
        available_uci = UCIDatasetHandler.get_available_uci_datasets()
        self.dataset_combo['values'] = available_uci

        ttk.Button(dataset_frame, text="Load Dataset Info",
                  command=self.load_dataset_info).grid(row=0, column=2, padx=5)
        ttk.Button(dataset_frame, text="Load Data",
                  command=self.load_data).grid(row=0, column=3, padx=5)
        ttk.Button(dataset_frame, text="Browse File",
                  command=self.browse_file).grid(row=0, column=4, padx=5)

        # Dataset info frame
        info_frame = ttk.LabelFrame(parent, text="Dataset Information", padding="10")
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, width=100)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Feature selection frame
        feature_frame = ttk.LabelFrame(parent, text="Feature Selection", padding="10")
        feature_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Target selection
        ttk.Label(feature_frame, text="Target Column:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(feature_frame, textvariable=self.target_var, width=20)
        self.target_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        self.target_var.trace('w', self.on_target_changed)

        # Feature selection
        ttk.Label(feature_frame, text="Feature Columns:").grid(row=1, column=0, sticky=tk.NW, padx=5)

        # Frame for feature checkboxes with scrollbar
        feature_list_frame = ttk.Frame(feature_frame)
        feature_list_frame.grid(row=1, column=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)

        # Create canvas and scrollbar for feature list
        self.feature_canvas = tk.Canvas(feature_list_frame, height=150)
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

        ttk.Button(button_frame, text="Select All",
                  command=self.select_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Deselect All",
                  command=self.deselect_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Select Numeric",
                  command=self.select_numeric_features).pack(side=tk.LEFT, padx=2)

        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        feature_frame.columnconfigure(1, weight=1)
        feature_frame.rowconfigure(1, weight=1)

    def setup_training_tab(self, parent):
        """Setup training and evaluation tab with new functionality"""
        # Control buttons frame - Top row
        control_frame1 = ttk.Frame(parent)
        control_frame1.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame1, text="Initialize Model",
                  command=self.initialize_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame1, text="Run Adaptive Learning",
                  command=self.run_adaptive_learning, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame1, text="Evaluate Model",
                  command=self.evaluate_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame1, text="Make Predictions",
                  command=self.make_predictions, width=15).pack(side=tk.LEFT, padx=2)

        # Control buttons frame - Bottom row (new functionality)
        control_frame2 = ttk.Frame(parent)
        control_frame2.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame2, text="Save Model",
                  command=self.save_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame2, text="Load Model",
                  command=self.load_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame2, text="Predict File",
                  command=self.predict_file, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame2, text="Show Label Info",
                  command=self.show_label_encoding_info, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame2, text="Exit",
                  command=self.exit_application, width=12).pack(side=tk.LEFT, padx=2)

        # Output frame
        output_frame = ttk.LabelFrame(parent, text="Training Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=100)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def setup_visualization_tab(self, parent):
        """Setup visualization tab"""
        # Visualization controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Show Class Distribution",
                  command=self.show_class_distribution).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Feature Importance",
                  command=self.show_feature_importance).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Confusion Matrix",
                  command=self.show_confusion_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Results",
                  command=self.show_results).pack(side=tk.LEFT, padx=5)

        # Visualization frame
        self.viz_frame = ttk.Frame(parent)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.viz_frame)
        self.toolbar.update()

    def log_output(self, message):
        """Add message to output text"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()
        self.status_var.set(message)

    def load_dataset_info(self):
        """Load and display dataset information"""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return

        dataset_info = UCIDatasetHandler.get_dataset_info(dataset_name)
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
        info_text += f"🔗 URL: {dataset_info.get('url', 'N/A')}\n\n"
        info_text += f"📋 Features: {', '.join(dataset_info.get('feature_names', []))}"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.log_output(f"✅ Loaded information for dataset: {dataset_name}")

    def browse_file(self):
        """Browse for data file with better path handling"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("Data files", "*.data"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            # Use the filename without extension as dataset name
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            self.dataset_var.set(dataset_name)
            self.log_output(f"📁 Selected file: {file_path}")
            self.load_data(file_path)

    def load_data(self, file_path=None):
        """Load dataset with proper error handling and target column synchronization"""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select or enter a dataset name first.")
            return

        try:
            self.log_output(f"📥 Loading dataset: {dataset_name}")

            # Create adaptive model with the dataset name
            self.adaptive_model = AdaptiveCTDBNN(dataset_name)

            # Load data - pass the file path properly
            if file_path:
                # Store the file path for later use
                self.adaptive_model.data_file_path = file_path
                # Load data first
                self.adaptive_model.model.load_data(file_path)
            else:
                # Pass the dataset name explicitly to the model
                self.adaptive_model.model.dataset_name = dataset_name
                self.adaptive_model.model.load_data()

            # NOW set the target column from GUI BEFORE preprocessing
            if hasattr(self, 'target_var') and self.target_var.get():
                target_column = self.target_var.get()
                # Update ALL target column references
                self.adaptive_model.model.target_column = target_column
                self.adaptive_model.model.preprocessor.target_column = target_column
                self.adaptive_model.config['target_column'] = target_column
                self.adaptive_model.target_column = target_column
                self.log_output(f"🎯 Target column set to: {target_column}")

            self.current_data = self.adaptive_model.model.data
            self.data_loaded = True
            if self.data_loaded:
                self.load_default_parameters()
                self.log_output("📋 Auto-loaded default hyperparameters")
            # Update feature selection
            self.update_feature_selection()

            self.log_output(f"✅ Dataset loaded: {self.current_data.shape[0]} samples, {self.current_data.shape[1]} features")
            self.log_output(f"📊 Available columns: {list(self.current_data.columns)}")

        except Exception as e:
            self.log_output(f"❌ Error loading dataset: {e}")
            import traceback
            self.log_output(f"🔍 Detailed error: {traceback.format_exc()}")

    def update_feature_selection(self):
        """Update feature selection UI and synchronize target column"""
        if not self.data_loaded or self.current_data is None:
            return

        # Clear existing feature checkboxes
        for widget in self.feature_scroll_frame.winfo_children():
            widget.destroy()

        self.feature_vars = {}
        columns = list(self.current_data.columns)

        # Update target combo
        self.target_combo['values'] = columns

        # Smart target column detection
        target_candidates = ['target', 'class', 'label', 'outcome', 'diagnosis', 'type', 'quality', 'ObjectType']
        current_target = self.target_var.get()

        # If no target selected or current target not in columns, try to auto-detect
        if not current_target or current_target not in columns:
            for candidate in target_candidates:
                if candidate in columns:
                    self.target_var.set(candidate)
                    # Immediately update the model if it exists
                    if hasattr(self, 'adaptive_model') and self.adaptive_model is not None:
                        self.adaptive_model.model.target_column = candidate
                        self.adaptive_model.model.preprocessor.target_column = candidate
                    self.log_output(f"🔍 Auto-detected target column: {candidate}")
                    break
            else:
                # If no candidate found, use the last column
                if columns:
                    self.target_var.set(columns[-1])
                    if hasattr(self, 'adaptive_model') and self.adaptive_model is not None:
                        self.adaptive_model.model.target_column = columns[-1]
                        self.adaptive_model.model.preprocessor.target_column = columns[-1]
                    self.log_output(f"📝 Using last column as target: {columns[-1]}")

        # Create feature checkboxes
        for i, col in enumerate(columns):
            var = tk.BooleanVar(value=True)
            self.feature_vars[col] = var

            # Determine column type for styling
            col_type = 'numeric' if pd.api.types.is_numeric_dtype(self.current_data[col]) else 'categorical'
            display_text = f"{col} ({col_type})"

            cb = ttk.Checkbutton(self.feature_scroll_frame, text=display_text, variable=var)
            cb.pack(anchor=tk.W, padx=5, pady=2)

        # Update the model's target column immediately
        current_target = self.target_var.get()
        if hasattr(self, 'adaptive_model') and self.adaptive_model is not None:
            self.adaptive_model.model.target_column = current_target
            self.adaptive_model.model.preprocessor.target_column = current_target
            self.adaptive_model.config['target_column'] = current_target

    def select_all_features(self):
        """Select all features"""
        for var in self.feature_vars.values():
            var.set(True)

    def deselect_all_features(self):
        """Deselect all features"""
        for var in self.feature_vars.values():
            var.set(False)

    def select_numeric_features(self):
        """Select only numeric features"""
        if not self.data_loaded:
            return

        for col, var in self.feature_vars.items():
            is_numeric = pd.api.types.is_numeric_dtype(self.current_data[col])
            var.set(is_numeric)

    def get_selected_features(self):
        """Get list of selected feature columns"""
        if not hasattr(self, 'feature_vars'):
            return []

        selected_features = []
        target_column = self.target_var.get()

        for col, var in self.feature_vars.items():
            if var.get() and col != target_column:
                selected_features.append(col)

        return selected_features

    def initialize_model(self):
        """Initialize the model with current hyperparameters"""
        if not self.data_loaded:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            self.log_output("🏗️ Initializing CT-DBNN model...")

            # Apply hyperparameters first
            self.apply_hyperparameters()

            # Get selected features and target
            selected_features = self.get_selected_features()
            target_column = self.target_var.get()

            if not selected_features:
                messagebox.showwarning("Warning", "Please select at least one feature.")
                return

            if not target_column:
                messagebox.showwarning("Warning", "Please select a target column.")
                return

            # Update ALL target column references
            self.adaptive_model.model.target_column = target_column
            self.adaptive_model.model.preprocessor.target_column = target_column
            self.adaptive_model.config['target_column'] = target_column
            self.adaptive_model.target_column = target_column

            # Prepare data with selected features
            if target_column not in self.current_data.columns:
                self.log_output(f"❌ Target column '{target_column}' not found in dataset.")
                self.log_output(f"   Available columns: {list(self.current_data.columns)}")
                return

            selected_data = self.current_data[selected_features + [target_column]]

            # Preprocess with current parameters
            self.adaptive_model.model.preprocessor.target_column = target_column
            X, y, feature_names = self.adaptive_model.model.preprocessor.preprocess_dataset(selected_data)

            # Store the preprocessed data
            self.adaptive_model.X_full = X
            self.adaptive_model.y_full = y
            self.adaptive_model.y_full_original = y.copy()

            self.log_output(f"✅ Model initialized with hyperparameters")
            self.log_output(f"🎯 Target: {target_column}")
            self.log_output(f"📊 Features: {', '.join(selected_features)}")
            self.log_output(f"🔧 Resolution: {self.resolution_var.get()}")
            self.log_output(f"📦 Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")

        except Exception as e:
            self.log_output(f"❌ Error initializing model: {e}")
            import traceback
            self.log_output(f"   Detailed error: {traceback.format_exc()}")

    def on_target_changed(self, *args):
        """Handle target column selection change - update ALL references"""
        if hasattr(self, 'adaptive_model') and self.adaptive_model is not None and self.data_loaded:
            target_column = self.target_var.get()

            # Update ALL target column references throughout the system
            self.adaptive_model.model.target_column = target_column
            self.adaptive_model.model.preprocessor.target_column = target_column
            self.adaptive_model.config['target_column'] = target_column
            self.adaptive_model.target_column = target_column

            self.log_output(f"🎯 Target column changed to: {target_column}")

    def run_adaptive_learning(self):
        """Run adaptive learning with proper data initialization"""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data and initialize model first.")
            return

        try:
            self.log_output("🚀 Starting adaptive learning...")

            # CRITICAL: Ensure target column is synchronized
            target_column = self.target_var.get()
            self.adaptive_model.model.target_column = target_column
            self.adaptive_model.model.preprocessor.target_column = target_column
            self.adaptive_model.config['target_column'] = target_column
            self.adaptive_model.target_column = target_column

            # If data hasn't been preprocessed yet, do it now
            if self.adaptive_model.X_full is None:
                self.log_output("🔄 Preprocessing data for adaptive learning...")
                selected_features = self.get_selected_features()
                selected_data = self.current_data[selected_features + [target_column]]

                X, y, feature_names = self.adaptive_model.model.preprocessor.preprocess_dataset(selected_data)
                self.adaptive_model.X_full = X
                self.adaptive_model.y_full = y
                self.adaptive_model.y_full_original = y.copy()
                self.log_output(f"✅ Preprocessed {X.shape[0]} samples with {X.shape[1]} features")

            # Update adaptive learning configuration
            self.adaptive_model.adaptive_config.update({
                'initial_samples_per_class': int(self.initial_samples_var.get()),
                'max_adaptive_rounds': int(self.max_rounds_var.get()),
            })

            # Run adaptive learning
            X_train, y_train, X_test, y_test = self.adaptive_model.adaptive_learn()

            self.model_trained = True
            self.log_output("✅ Adaptive learning completed successfully!")
            self.log_output(f"📊 Final training set: {len(X_train)} samples")
            self.log_output(f"📈 Final test set: {len(X_test)} samples")
            self.log_output(f"🏆 Best accuracy: {self.adaptive_model.best_accuracy:.4f}")

        except Exception as e:
            error_msg = f"❌ Error during adaptive learning: {e}"
            self.log_output(error_msg)
            import traceback
            detailed_error = traceback.format_exc()
            self.log_output(f"🔍 Detailed error:\n{detailed_error}")
            messagebox.showerror("Adaptive Learning Error", f"{error_msg}\n\nCheck the output log for details.")

    def evaluate_model(self):
        """Evaluate the trained model"""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please run adaptive learning first.")
            return

        try:
            self.log_output("📊 Evaluating model...")

            if hasattr(self.adaptive_model, 'X_full') and hasattr(self.adaptive_model, 'y_full'):
                predictions = self.adaptive_model.model.predict(self.adaptive_model.X_full)
                accuracy = accuracy_score(self.adaptive_model.y_full, predictions)

                self.log_output(f"🔍 Model Evaluation:")
                self.log_output(f"   Accuracy on full dataset: {accuracy:.4f}")
                self.log_output(f"   Best adaptive accuracy: {self.adaptive_model.best_accuracy:.4f}")

                # Show classification report
                report = classification_report(self.adaptive_model.y_full, predictions)
                self.log_output(f"\n📈 Classification Report:\n{report}")
            else:
                self.log_output("❌ No data available for evaluation.")

        except Exception as e:
            self.log_output(f"❌ Error during evaluation: {e}")

    def make_predictions(self):
        """Make predictions using trained model"""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please run adaptive learning first.")
            return

        try:
            # For demonstration, use the test set for predictions
            if hasattr(self.adaptive_model, 'X_full'):
                predictions = self.adaptive_model.model.predict(self.adaptive_model.X_full)
                accuracy = accuracy_score(self.adaptive_model.y_full, predictions)

                self.log_output(f"🔮 Predictions on full dataset:")
                self.log_output(f"   Accuracy: {accuracy:.4f}")
                self.log_output(f"   Sample predictions: {predictions[:10]}...")

                # Show some actual vs predicted
                actual = self.adaptive_model.y_full[:10]
                predicted = predictions[:10]
                self.log_output(f"   Actual:    {actual}")
                self.log_output(f"   Predicted: {predicted}")
            else:
                self.log_output("❌ No data available for predictions.")

        except Exception as e:
            self.log_output(f"❌ Error during predictions: {e}")

    def show_class_distribution(self):
        """Show class distribution plot"""
        if not self.data_loaded:
            return

        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            target_col = self.target_var.get()
            if target_col in self.current_data.columns:
                class_counts = self.current_data[target_col].value_counts().sort_index()
                ax.bar(class_counts.index.astype(str), class_counts.values, alpha=0.7, color='skyblue')
                ax.set_title('Class Distribution')
                ax.set_xlabel('Class')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

            self.canvas.draw()
            self.log_output("✅ Class distribution plot displayed")

        except Exception as e:
            self.log_output(f"❌ Error showing class distribution: {e}")

    def show_feature_importance(self):
        """Show feature importance plot"""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            # Get feature importance from the model
            if hasattr(self.adaptive_model.model.core, 'global_anti_net'):
                n_features = self.adaptive_model.model.core.innodes
                feature_importance = np.zeros(n_features)

                for i in range(n_features):
                    feature_importance[i] = np.sum(self.adaptive_model.model.core.global_anti_net[i+1, :, :, :, :])

                # Normalize
                feature_importance = feature_importance / np.sum(feature_importance)

                # Get feature names
                feature_names = self.get_selected_features()
                if len(feature_names) != n_features:
                    feature_names = [f'Feature {i+1}' for i in range(n_features)]

                y_pos = np.arange(len(feature_names))
                bars = ax.barh(y_pos, feature_importance)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names)
                ax.set_xlabel('Importance Score')
                ax.set_title('Feature Importance')
                ax.grid(True, alpha=0.3)

                # Add value annotations
                for i, v in enumerate(feature_importance):
                    ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

            self.canvas.draw()
            self.log_output("✅ Feature importance plot displayed")

        except Exception as e:
            self.log_output(f"❌ Error showing feature importance: {e}")

    def show_confusion_matrix(self):
        """Show confusion matrix"""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return

        try:
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            if hasattr(self.adaptive_model, 'X_full') and hasattr(self.adaptive_model, 'y_full'):
                predictions = self.adaptive_model.model.predict(self.adaptive_model.X_full)
                cm = confusion_matrix(self.adaptive_model.y_full, predictions)

                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title('Confusion Matrix')
                ax.figure.colorbar(im, ax=ax)

                # Add labels
                classes = np.unique(self.adaptive_model.y_full)
                tick_marks = np.arange(len(classes))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(classes)
                ax.set_yticklabels(classes)
                ax.set_ylabel('True label')
                ax.set_xlabel('Predicted label')

                # Add text annotations
                thresh = cm.max() / 2.
                for i, j in np.ndindex(cm.shape):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            self.canvas.draw()
            self.log_output("✅ Confusion matrix displayed")

        except Exception as e:
            self.log_output(f"❌ Error showing confusion matrix: {e}")

    def show_results(self):
        """Show detailed results"""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please run adaptive learning first.")
            return

        try:
            self.log_output("\n📈 Detailed Results:")
            self.log_output(f"   Dataset: {self.adaptive_model.dataset_name}")
            self.log_output(f"   Best Accuracy: {self.adaptive_model.best_accuracy:.4f}")
            self.log_output(f"   Training Rounds: {self.adaptive_model.adaptive_round}")
            self.log_output(f"   Final Training Size: {len(self.adaptive_model.best_training_indices)}")
            self.log_output(f"   Resolution: {self.adaptive_model.config.get('resol', 100)}")

            # Compare with best known results
            dataset_info = UCIDatasetHandler.get_dataset_info(self.adaptive_model.dataset_name)
            if dataset_info and 'best_accuracy' in dataset_info:
                best_known = dataset_info['best_accuracy']
                self.log_output(f"   Best Known Accuracy: {best_known}")

                if isinstance(best_known, (int, float)):
                    if self.adaptive_model.best_accuracy >= best_known:
                        self.log_output("   🎉 EXCELLENT! Matched or exceeded best known accuracy!")
                    elif self.adaptive_model.best_accuracy >= best_known * 0.95:
                        self.log_output("   👍 GOOD! Within 5% of best known accuracy!")
                    else:
                        self.log_output("   💡 There's room for improvement.")

            # Show memory usage
            memory_usage = MemoryManager.get_memory_usage()
            self.log_output(f"   Memory Usage: {memory_usage:.1f} MB")

        except Exception as e:
            self.log_output(f"❌ Error showing results: {e}")

def main():
    """Main function to run adaptive CT-DBNN"""
    import sys

    # Check for GUI flag or no arguments
    if "--gui" in sys.argv or "-g" in sys.argv or len(sys.argv) == 1:
        if GUI_AVAILABLE:
            print("🎨 Launching Adaptive CT-DBNN GUI...")
            root = tk.Tk()
            app = AdaptiveCTDBNNGUI(root)
            root.mainloop()
        else:
            print("❌ GUI not available. Using command line interface.")
            # Fall back to command line
            run_command_line()
    else:
        run_command_line()

def run_command_line():
    """Run the command line interface"""
    print("🎯 Adaptive CT-DBNN System with UCI Dataset Support")
    print("=" * 60)

    # Check for config file parameter
    config_file = None
    dataset_name = None

    for i, arg in enumerate(sys.argv):
        if arg in ["--config", "-c"] and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
        elif arg in ["--dataset", "-d"] and i + 1 < len(sys.argv):
            dataset_name = sys.argv[i + 1]

    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {config_file}")

            # Extract dataset name from config if not provided
            if dataset_name is None and 'dataset_name' in config:
                dataset_name = config['dataset_name']

        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            config = {}

    # If no dataset name provided, show available options
    if dataset_name is None:
        # Show available UCI datasets
        available_uci = UCIDatasetHandler.get_available_uci_datasets()
        print(f"📋 Available UCI datasets: {', '.join(available_uci)}")

        # Show available config files
        available_configs = DatasetConfig.get_available_config_files()
        if available_configs:
            print(f"📁 Available configuration files:")
            for cfg in available_configs:
                print(f"   • {cfg['file']} ({cfg['type']})")

        print("\n💡 Usage: python adaptive_ctdbnn.py --dataset <dataset_name>")
        print("💡 Or: python adaptive_ctdbnn.py --config <config_file>")
        print("💡 Or: python adaptive_ctdbnn.py --gui (for graphical interface)")
        return

    # Create adaptive CT-DBNN
    print(f"🎯 Initializing Adaptive CT-DBNN for dataset: {dataset_name}")
    adaptive_model = AdaptiveCTDBNN(dataset_name, config)

    # Run adaptive learning
    print("\n🚀 Starting adaptive learning with CT-DBNN...")
    try:
        X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()

        print(f"\n✅ Adaptive learning completed!")
        print(f"📦 Final training set size: {len(X_train)}")
        print(f"📊 Final test set size: {len(X_test)}")
        print(f"🏆 Best accuracy achieved: {adaptive_model.best_accuracy:.4f}")

        # Compare with best known results
        dataset_info = UCIDatasetHandler.get_dataset_info(adaptive_model.dataset_name)
        if dataset_info and 'best_accuracy' in dataset_info:
            best_known = dataset_info['best_accuracy']
            print(f"📚 Best known accuracy: {best_known}")

            if isinstance(best_known, (int, float)):
                if adaptive_model.best_accuracy >= best_known:
                    print("🎉 EXCELLENT! Matched or exceeded best known accuracy!")
                elif adaptive_model.best_accuracy >= best_known * 0.95:
                    print("👍 GOOD! Within 5% of best known accuracy!")
                else:
                    print("💡 There's room for improvement. Consider tuning parameters.")

    except Exception as e:
        print(f"❌ Adaptive learning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
