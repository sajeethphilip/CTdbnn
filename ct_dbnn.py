#!/usr/bin/env python3
"""
ENHANCED CT-DBNN MODULE - Complete Prediction System with Feature Selection
BEAUTIFIED INTERFACE WITH COMPREHENSIVE PREDICTION CAPABILITIES
"""

import numpy as np
import time
import os
import sys
import argparse
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
import pandas as pd
import urllib.request
import zipfile
import io
import re

# Parallel processing imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

# Optional imports for GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from tkinter import simpledialog
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.use('TkAgg')
    GUI_AVAILABLE = True
    from ct_dbnn_gui import EnhancedGUI
except ImportError:
    GUI_AVAILABLE = False


# UCI Dataset Repository with metadata
UCI_DATASETS = {
    "iris": {
        "name": "Iris",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "description": "Classic iris flower dataset with 3 classes",
        "features": 4,
        "samples": 150,
        "best_accuracy": 0.973,
        "reference": "Fisher, R.A. (1936)"
    },
    "wine": {
        "name": "Wine",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "description": "Wine recognition data from chemical analysis",
        "features": 13,
        "samples": 178,
        "best_accuracy": 1.000,
        "reference": "Aeberhard, S. et al. (1992)"
    },
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        "description": "Breast cancer diagnosis data",
        "features": 30,
        "samples": 569,
        "best_accuracy": 0.971,
        "reference": "Wolberg, W.H. et al. (1995)"
    },
    "diabetes": {
        "name": "Pima Indians Diabetes",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
        "description": "Diabetes diagnosis in Pima Indian population",
        "features": 8,
        "samples": 768,
        "best_accuracy": 0.772,
        "reference": "Smith, J.W. et al. (1988)"
    }
}


class UCIDatasetLoader:
    """Loader for fetching any UCI dataset by name"""

    @staticmethod
    def fetch_dataset_info(dataset_name):
        """Fetch dataset information from UCI repository"""
        try:
            # Try to find dataset in UCI repository
            base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            dataset_urls = [
                f"{base_url}{dataset_name}/{dataset_name}.data",
                f"{base_url}{dataset_name.replace('-', '_')}/{dataset_name.replace('-', '_')}.data",
                f"{base_url}{dataset_name}/data.csv",
                f"{base_url}{dataset_name.replace('-', '_')}/data.csv"
            ]

            for url in dataset_urls:
                try:
                    response = urllib.request.urlopen(url)
                    if response.status == 200:
                        return {
                            "name": dataset_name.replace('_', ' ').title(),
                            "url": url,
                            "description": f"UCI {dataset_name} dataset",
                            "features": "Unknown",
                            "samples": "Unknown",
                            "best_accuracy": "Unknown",
                            "reference": "UCI Repository"
                        }
                except:
                    continue

            return None
        except Exception as e:
            print(f"Error fetching dataset info: {e}")
            return None

    @staticmethod
    def load_any_uci_dataset(dataset_name):
        """Load any UCI dataset by name"""
        try:
            # First try predefined datasets
            if dataset_name in UCI_DATASETS:
                return UCI_DATASETS[dataset_name]

            # Try to fetch from UCI repository
            info = UCIDatasetLoader.fetch_dataset_info(dataset_name)
            if info:
                return info
            else:
                # Try common variations
                variations = [
                    dataset_name,
                    dataset_name.replace('-', ''),
                    dataset_name.replace('_', ''),
                    dataset_name.replace('-', '_'),
                    dataset_name.replace('_', '-')
                ]

                for var in variations:
                    info = UCIDatasetLoader.fetch_dataset_info(var)
                    if info:
                        return info

            return None

        except Exception as e:
            print(f"Error loading UCI dataset: {e}")
            return None

    @staticmethod
    def download_uci_data(dataset_info):
        """Download and parse UCI dataset data"""
        try:
            url = dataset_info['url']
            response = urllib.request.urlopen(url)
            content = response.read().decode('utf-8')

            # Try to parse as CSV
            try:
                # Try different delimiters
                for delimiter in [',', '\t', ';', ' ']:
                    try:
                        from io import StringIO
                        df = pd.read_csv(StringIO(content), delimiter=delimiter, header=None)
                        if df.shape[1] > 1:  # Valid dataset with multiple columns
                            return df
                    except:
                        continue
            except:
                pass

            # If CSV parsing fails, try space-separated
            lines = content.strip().split('\n')
            data = []
            for line in lines:
                if line.strip() and not line.startswith('@') and not line.startswith('#'):
                    # Clean the line and split by whitespace
                    cleaned_line = re.sub(r'\s+', ' ', line.strip())
                    row = cleaned_line.split(' ')
                    # Filter out empty strings
                    row = [x for x in row if x]
                    if row:
                        data.append(row)

            if data:
                # Find maximum columns
                max_cols = max(len(row) for row in data)
                # Pad rows with fewer columns
                for i, row in enumerate(data):
                    if len(row) < max_cols:
                        data[i] = row + [np.nan] * (max_cols - len(row))

                return pd.DataFrame(data)
            else:
                return None

        except Exception as e:
            print(f"Error downloading UCI data: {e}")
            return None


class DataPreprocessor:
    """
    Comprehensive data preprocessing with CONSISTENT label encoding
    ALWAYS preserves and uses original feature names - NO DUMMY NAMES
    """
    def __init__(self, missing_value_placeholder=-99999):
        self.missing_value_placeholder = missing_value_placeholder
        self.feature_encoders = {}  # encoders for each feature column
        self.target_encoder = None  # encoder for target column
        self.target_decoder = None  # decoder for target column
        self.feature_columns = None  # Store original feature names
        self.target_column = None
        self.original_dtypes = {}
        self.is_fitted = False

    def fit_transform_features(self, features, feature_names=None):
        """Fit and transform features with label encoding and missing value handling"""
        if isinstance(features, pd.DataFrame):
            # ALWAYS use DataFrame column names - PRESERVE ORIGINAL NAMES
            self.feature_columns = list(features.columns)
            print(f"🔧 Using DataFrame column names: {self.feature_columns}")
            result = self._fit_transform_dataframe(features)
        elif isinstance(features, np.ndarray):
            # FORCE use of provided feature names - NO DUMMY NAMES
            if feature_names is not None:
                self.feature_columns = feature_names
                print(f"🔧 Using provided feature names: {self.feature_columns}")
            else:
                # If no names provided, try to extract from the data object
                n_features = features.shape[1]
                if hasattr(features, 'columns'):
                    self.feature_columns = list(features.columns)
                    print(f"🔧 Using data.columns: {self.feature_columns}")
                elif hasattr(features, 'feature_names'):
                    self.feature_columns = features.feature_names
                    print(f"🔧 Using data.feature_names: {self.feature_columns}")
                elif hasattr(features, 'feature_names_in_'):
                    # For newer sklearn datasets
                    self.feature_columns = list(features.feature_names_in_)
                    print(f"🔧 Using data.feature_names_in_: {self.feature_columns}")
                else:
                    # LAST RESORT: Use descriptive names that indicate source
                    self.feature_columns = [f'Feature_{i+1}' for i in range(n_features)]
                    print(f"⚠️  No feature names found. Using: {self.feature_columns}")
            result = self._fit_transform_array(features)
        else:
            # Convert to numpy array and process
            features_array = np.array(features)
            if feature_names is not None:
                self.feature_columns = feature_names
                print(f"🔧 Using provided feature names: {self.feature_columns}")
            else:
                n_features = features_array.shape[1]
                self.feature_columns = [f'Feature_{i+1}' for i in range(n_features)]
                print(f"⚠️  No feature names provided. Using: {self.feature_columns}")
            result = self._fit_transform_array(features_array)

        self.is_fitted = True
        return result

    def _fit_transform_dataframe(self, df):
        """Fit and transform DataFrame features - PRESERVE ORIGINAL COLUMN NAMES"""
        self.feature_columns = list(df.columns)
        self.original_dtypes = df.dtypes.to_dict()

        processed_data = df.copy()
        self.feature_encoders = {}

        for col in df.columns:
            # Handle missing values
            if df[col].isna().any():
                processed_data[col] = df[col].fillna(self.missing_value_placeholder)

            # Convert to numeric if possible, otherwise use label encoding
            if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
                # Label encoding for categorical data
                unique_vals = processed_data[col].unique()
                encoder = {val: idx for idx, val in enumerate(unique_vals)}
                self.feature_encoders[col] = encoder
                processed_data[col] = processed_data[col].map(encoder)
            else:
                # Keep numeric data as is, but store identity encoder
                self.feature_encoders[col] = 'numeric'

        return processed_data.values.astype(np.float64)

    def _fit_transform_array(self, array):
        """Fit and transform numpy array features - USE ORIGINAL FEATURE NAMES"""
        processed_data = array.copy().astype(object)
        n_samples, n_features = array.shape

        # CRITICAL: Use stored feature columns - NO DUMMY NAMES
        if self.feature_columns is None:
            self.feature_columns = [f'Column_{i+1}' for i in range(n_features)]
            print(f"⚠️  No feature columns set. Using: {self.feature_columns}")

        self.feature_encoders = {}

        for col_idx in range(n_features):
            col_name = self.feature_columns[col_idx]
            col_data = array[:, col_idx]

            # Handle missing values
            missing_mask = pd.isna(col_data) if hasattr(pd, 'isna') else (
                (col_data == None) | ((isinstance(col_data, str)) & ((col_data == 'NaN') | (col_data == 'NA') | (col_data == '')))
            )
            if np.any(missing_mask):
                processed_data[missing_mask, col_idx] = self.missing_value_placeholder

            # Check if column needs encoding (non-numeric)
            try:
                # Try to convert to numeric
                numeric_data = pd.to_numeric(processed_data[:, col_idx], errors='coerce')
                non_numeric_mask = pd.isna(numeric_data)
                if np.any(non_numeric_mask):
                    # Contains non-numeric values, need encoding
                    unique_vals = np.unique(processed_data[:, col_idx])
                    encoder = {val: idx for idx, val in enumerate(unique_vals)}
                    self.feature_encoders[col_name] = encoder  # Use column name as key
                    for i, val in enumerate(processed_data[:, col_idx]):
                        processed_data[i, col_idx] = encoder.get(val, 0)
                else:
                    # All numeric
                    processed_data[:, col_idx] = numeric_data
                    self.feature_encoders[col_name] = 'numeric'  # Use column name as key
            except:
                # Fallback: treat as categorical
                unique_vals = np.unique(processed_data[:, col_idx])
                encoder = {val: idx for idx, val in enumerate(unique_vals)}
                self.feature_encoders[col_name] = encoder  # Use column name as key
                for i, val in enumerate(processed_data[:, col_idx]):
                    processed_data[i, col_idx] = encoder.get(val, 0)

        return processed_data.astype(np.float64)

    def transform_features(self, features):
        """Transform features using fitted encoders - HANDLE COLUMN NAMES PROPERLY"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        if isinstance(features, pd.DataFrame):
            return self._transform_dataframe(features)
        elif isinstance(features, np.ndarray):
            return self._transform_array(features)
        else:
            features_array = np.array(features)
            return self._transform_array(features_array)

    def _transform_dataframe(self, df):
        """Transform DataFrame features using fitted encoders - USE EXACT TRAINING COLUMN NAMES AND ORDER"""
        processed_data = df.copy()

        # CRITICAL: Check if all required columns are present
        missing_columns = []
        for fitted_col in self.feature_columns:  # Use stored feature columns in training order
            if fitted_col not in df.columns:
                missing_columns.append(fitted_col)

        if missing_columns:
            raise ValueError(f"Missing columns in prediction data: {missing_columns}. "
                           f"Available columns: {list(df.columns)}. "
                           f"Expected columns (in training order): {self.feature_columns}")

        # Process only the columns that were fitted, in the exact training order
        for col in self.feature_columns:
            # Handle missing values
            if df[col].isna().any():
                processed_data[col] = df[col].fillna(self.missing_value_placeholder)

            # Apply encoding if needed
            if self.feature_encoders[col] != 'numeric':
                encoder = self.feature_encoders[col]
                processed_data[col] = processed_data[col].map(
                    lambda x: encoder.get(x, 0) if x in encoder else 0
                )

        # Ensure we return only the columns that were fitted, in the CORRECT TRAINING ORDER
        processed_data = processed_data[self.feature_columns]
        return processed_data.values.astype(np.float64)

    def _transform_array(self, array):
        """Transform numpy array features using fitted encoders"""
        processed_data = array.copy().astype(object)
        n_samples, n_features = array.shape

        # Check if number of features matches
        if n_features != len(self.feature_columns):
            raise ValueError(f"Number of features ({n_features}) doesn't match fitted features ({len(self.feature_columns)})")

        for col_idx in range(n_features):
            col_name = self.feature_columns[col_idx]

            if col_name not in self.feature_encoders:
                raise ValueError(f"Column '{col_name}' not seen during fitting. Available columns: {list(self.feature_encoders.keys())}")

            col_data = array[:, col_idx]

            # Handle missing values
            missing_mask = pd.isna(col_data) if hasattr(pd, 'isna') else (
                (col_data == None) | ((isinstance(col_data, str)) & ((col_data == 'NaN') | (col_data == 'NA') | (col_data == '')))
            )
            if np.any(missing_mask):
                processed_data[missing_mask, col_idx] = self.missing_value_placeholder

            # Apply encoding if needed
            if self.feature_encoders[col_name] != 'numeric':
                encoder = self.feature_encoders[col_name]
                for i, val in enumerate(processed_data[:, col_idx]):
                    processed_data[i, col_idx] = encoder.get(val, 0)
            else:
                # Convert to numeric
                try:
                    processed_data[:, col_idx] = pd.to_numeric(processed_data[:, col_idx], errors='coerce')
                    processed_data[pd.isna(processed_data[:, col_idx]), col_idx] = self.missing_value_placeholder
                except:
                    processed_data[:, col_idx] = self.missing_value_placeholder

        return processed_data.astype(np.float64)

    def fit_transform_targets(self, targets):
        """Fit and transform targets with label encoding - STORE BOTH ENCODER AND DECODER"""
        targets_data = self._extract_targets(targets)
        self.target_encoder = {}
        self.target_decoder = {}

        unique_vals = np.unique(targets_data)
        for idx, val in enumerate(unique_vals):
            encoded_val = float(idx + 1)
            self.target_encoder[val] = encoded_val
            self.target_decoder[encoded_val] = val

        encoded = np.array([self.target_encoder[val] for val in targets_data])
        return encoded.astype(np.float64)

    def transform_targets(self, targets):
        """Transform targets using fitted encoder"""
        if self.target_encoder is None:
            raise ValueError("Target encoder not fitted")

        targets_data = self._extract_targets(targets)
        encoded = np.array([self.target_encoder.get(val, 0.0) for val in targets_data])
        return encoded.astype(np.float64)

    def inverse_transform_targets(self, encoded_targets):
        """Convert encoded targets back to original labels"""
        if self.target_decoder is None:
            raise ValueError("Target decoder not fitted")

        if isinstance(encoded_targets, (pd.Series, pd.DataFrame)):
            original = encoded_targets.map(lambda x: self.target_decoder.get(x, "Unknown"))
        else:
            original = np.array([self.target_decoder.get(x, "Unknown") for x in encoded_targets])

        return original

    def _extract_targets(self, targets):
        """Extract targets array from various input types"""
        if isinstance(targets, np.ndarray):
            return targets
        elif isinstance(targets, pd.Series):
            return targets.values
        elif isinstance(targets, pd.DataFrame):
            return targets.values.flatten()
        else:
            try:
                return np.array(targets)
            except:
                raise ValueError(f"Unsupported targets type: {type(targets)}")

    def get_feature_info(self):
        """Get information about feature encoding"""
        info = {
            'missing_value_placeholder': self.missing_value_placeholder,
            'feature_encoders': {},
            'target_encoder': self.target_encoder,
            'target_decoder': self.target_decoder,
            'is_fitted': self.is_fitted
        }

        for col, encoder in self.feature_encoders.items():
            if encoder == 'numeric':
                info['feature_encoders'][col] = 'numeric'
            else:
                info['feature_encoders'][col] = {
                    'type': 'categorical',
                    'mapping': encoder,
                    'num_categories': len(encoder)
                }

        return info

    def get_feature_names(self):
        """Get original feature names"""
        return self.feature_columns

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore state from pickling"""
        self.__dict__.update(state)


class ParallelCTDBNN:
    """
    Optimized CT-DBNN with CONSISTENT label handling across all operations
    ALWAYS uses original feature names - NO DUMMY NAMES
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'resol': 8,
            'use_complex_tensor': True,
            'orthogonalize_weights': True,
            'smoothing_factor': 1e-8,
            'parallel_processing': True,
            'n_jobs': -1,
            'batch_size': 1000,
            'missing_value_placeholder': -99999,
        }
        if config:
            self.config.update(config)

        # Data preprocessor
        self.preprocessor = DataPreprocessor(
            missing_value_placeholder=self.config['missing_value_placeholder']
        )

        # Determine number of workers
        if self.config['n_jobs'] == -1:
            self.n_jobs = mp.cpu_count()
        else:
            self.n_jobs = min(self.config['n_jobs'], mp.cpu_count())

        print(f"🔄 Parallel processing: {self.n_jobs} workers")

        # Global data structures - COMPUTED ONCE DURING TRAINING
        self.global_anti_net = None
        self.binloc = None
        self.max_val = None
        self.min_val = None
        self.dmyclass = None
        self.resolution_arr = None

        # Weight structures - COMPUTED ONCE DURING TRAINING
        self.anti_wts = None
        self.complex_weights = None

        # State
        self.innodes = 0
        self.outnodes = 0
        self.class_to_encoded = None
        self.encoded_to_class = None
        self.is_trained = False
        self.likelihoods_computed = False

        # Feature names for proper output
        self.feature_names = None
        self.target_name = None

        # Training history
        self.training_history = {
            'training_time': 0,
            'train_accuracy': 0,
            'test_accuracy': 0,
            'config': self.config.copy()
        }

        # Store normalized training features for consistent processing
        self.training_features_norm = None
        self.training_targets_encoded = None

        # Model metadata
        self.model_metadata = {
            'creation_time': time.time(),
            'version': '1.0.0',
            'data_name': 'unknown_dataset'
        }

        # Feature selection
        self.selected_features = None
        self.target_column_name = None

    def compute_global_likelihoods(self, features, targets, feature_names=None):
        """
        Compute global likelihoods ONCE on entire dataset with comprehensive preprocessing
        This is called ONLY during training - PRESERVES ACTUAL FEATURE NAMES
        """
        if self.likelihoods_computed:
            print("⚠️  Likelihoods already computed! Using existing global likelihoods.")
            return self.training_features_norm

        print("Computing GLOBAL likelihoods on entire dataset...")
        print("🔧 Applying comprehensive data preprocessing...")

        # CRITICAL FIX: Extract actual feature names from the data
        actual_feature_names = None

        if hasattr(features, 'columns'):
            # DataFrame with column names - USE EXACT COLUMN NAMES
            actual_feature_names = list(features.columns)
            print(f"🔧 Using DataFrame column names: {actual_feature_names}")
        elif feature_names is not None:
            # Use provided feature names
            actual_feature_names = feature_names
            print(f"🔧 Using provided feature names: {actual_feature_names}")
        elif hasattr(features, 'feature_names'):
            # Data object with feature_names attribute
            actual_feature_names = features.feature_names
            print(f"🔧 Using data.feature_names: {actual_feature_names}")
        elif hasattr(features, 'feature_names_in_'):
            # For newer sklearn datasets
            actual_feature_names = list(features.feature_names_in_)
            print(f"🔧 Using data.feature_names_in_: {actual_feature_names}")
        else:
            # Try to extract from the data structure
            try:
                # For sklearn datasets, they often have feature_names attribute
                if hasattr(features, 'feature_names'):
                    actual_feature_names = features.feature_names
                    print(f"🔧 Using dataset.feature_names: {actual_feature_names}")
                else:
                    # Last resort: check if we can infer from the data structure
                    n_features = features.shape[1] if hasattr(features, 'shape') else len(features[0])
                    actual_feature_names = [f'Feature_{i+1}' for i in range(n_features)]
                    print(f"⚠️  No feature names found. Using: {actual_feature_names}")
            except:
                n_features = features.shape[1] if hasattr(features, 'shape') else len(features[0])
                actual_feature_names = [f'Feature_{i+1}' for i in range(n_features)]
                print(f"⚠️  Could not extract feature names. Using: {actual_feature_names}")

        # Preprocess features and targets - PASS ACTUAL FEATURE NAMES
        features_processed = self.preprocessor.fit_transform_features(features, actual_feature_names)
        self.training_targets_encoded = self.preprocessor.fit_transform_targets(targets)

        # Store feature names from preprocessor - USE ACTUAL NAMES
        self.feature_names = self.preprocessor.get_feature_names()

        # Update model metadata with dataset info - USE ACTUAL NAME
        if hasattr(features, 'name'):
            self.model_metadata['data_name'] = features.name
        elif hasattr(targets, 'name'):
            self.model_metadata['data_name'] = targets.name
        elif hasattr(self, 'model_metadata') and 'data_name' in self.model_metadata:
            # Keep existing name if already set
            pass
        else:
            # Use a descriptive name
            self.model_metadata['data_name'] = f"dataset_{int(time.time())}"

        n_samples, n_features = features_processed.shape
        self.innodes = n_features
        resol = self.config['resol']

        print(f"Data shape: {features_processed.shape}, Resolution: {resol}")
        print(f"✅ ACTUAL Feature names: {self.feature_names}")
        print(f"Missing values replaced with: {self.config['missing_value_placeholder']}")

        # Step 1: Fit encoder - USE THE SAME ENCODING AS PREPROCESSOR
        self._fit_encoder_consistent()

        # Initialize global arrays with proper dimensions
        self.max_val = np.zeros(n_features + 2, dtype=np.float64)
        self.min_val = np.zeros(n_features + 2, dtype=np.float64)
        self.resolution_arr = np.zeros(n_features + 8, dtype=np.int32)

        # Initialize binloc with proper dimensions
        self.binloc = np.zeros((n_features + 2, resol + 8), dtype=np.float64)

        # Compute global min/max (ignoring missing values) - STORE FOR CONSISTENT NORMALIZATION
        for i in range(n_features):
            feature_idx = i + 1
            feature_data = features_processed[:, i]

            # Filter out missing values for min/max calculation
            valid_mask = feature_data != self.config['missing_value_placeholder']
            valid_data = feature_data[valid_mask]

            if len(valid_data) > 0:
                self.max_val[feature_idx] = np.max(valid_data)
                self.min_val[feature_idx] = np.min(valid_data)
            else:
                self.max_val[feature_idx] = 1.0
                self.min_val[feature_idx] = 0.0

            self.resolution_arr[feature_idx] = resol

            # Initialize bin locations
            for j in range(1, resol + 1):
                self.binloc[feature_idx][j] = (j - 1) * 1.0

        # Normalize features (missing values remain as placeholder during normalization)
        # STORE THE NORMALIZED FEATURES FOR CONSISTENT PROCESSING
        self.training_features_norm = self._normalize_features(features_processed)

        # Initialize global network counts with proper dimensions
        self.global_anti_net = np.zeros(
            (n_features + 2, resol + 2, n_features + 2, resol + 2, self.outnodes + 2),
            dtype=np.float64
        )

        # Build global likelihoods (skip samples with too many missing values)
        total_samples = len(self.training_features_norm)
        print(f"Building global likelihoods from {total_samples} samples...")

        valid_samples = 0
        for sample_idx in range(total_samples):
            # Skip samples that are entirely missing values
            sample_data = self.training_features_norm[sample_idx, :]
            missing_count = np.sum(sample_data == self.config['missing_value_placeholder'])

            if missing_count >= n_features:  # Skip if all features are missing
                continue

            valid_samples += 1
            bins = self._find_closest_bins(sample_data)

            for i in range(n_features):
                feature_i = i + 1
                bin_i = bins[i] + 1

                # Skip if this feature is missing
                if self.training_features_norm[sample_idx, i] == self.config['missing_value_placeholder']:
                    continue

                for l in range(n_features):
                    feature_l = l + 1
                    bin_l = bins[l] + 1

                    # Skip if this feature is missing
                    if self.training_features_norm[sample_idx, l] == self.config['missing_value_placeholder']:
                        continue

                    # Find correct class for this sample - USE ENCODED TARGETS
                    k_class = 1
                    while (k_class <= self.outnodes and
                           abs(self.training_targets_encoded[sample_idx] - self.dmyclass[k_class]) > self.dmyclass[0]):
                        k_class += 1

                    if k_class <= self.outnodes:
                        self.global_anti_net[feature_i, bin_i, feature_l, bin_l, k_class] += 1
                        self.global_anti_net[feature_i, bin_i, feature_l, bin_l, 0] += 1

            if sample_idx > 0 and sample_idx % 50 == 0:
                print(f"  Processed {sample_idx}/{total_samples} samples...")

        print(f"✅ Used {valid_samples} valid samples (excluding entirely missing samples)")

        # Apply smoothing
        smoothing = self.config['smoothing_factor']
        for i in range(1, n_features + 1):
            for j in range(1, resol + 1):
                for l in range(1, n_features + 1):
                    for m in range(1, resol + 1):
                        for k in range(1, self.outnodes + 1):
                            self.global_anti_net[i, j, l, m, k] += smoothing
                        self.global_anti_net[i, j, l, m, 0] += smoothing * self.outnodes

        self.likelihoods_computed = True
        print("✅ Global likelihoods computed and fixed")

        return self.training_features_norm

    def _fit_encoder_consistent(self):
        """Fit class encoder - CONSISTENT with preprocessor encoding"""
        if self.preprocessor.target_encoder is None:
            raise ValueError("Preprocessor target encoder not fitted")

        # USE THE SAME ENCODING AS THE PREPROCESSOR
        self.class_to_encoded = self.preprocessor.target_encoder
        self.encoded_to_class = self.preprocessor.target_decoder

        unique_classes = list(self.class_to_encoded.keys())
        self.outnodes = len(unique_classes)

        self.dmyclass = np.zeros(self.outnodes + 2, dtype=np.float64)
        self.dmyclass[0] = 0.2

        for i, original_class in enumerate(unique_classes, 1):
            self.dmyclass[i] = self.class_to_encoded[original_class]

        print(f"Encoded {self.outnodes} classes: {unique_classes}")
        print(f"Encoding mapping: {self.class_to_encoded}")

    def initialize_orthogonal_weights(self):
        """
        ONE-STEP ORTHOGONALIZATION in Complex Tensor Space
        Called ONLY during training
        """
        if not self.likelihoods_computed:
            raise ValueError("Must compute global likelihoods first!")

        resol = self.config['resol']
        n_features = self.innodes

        if self.config['use_complex_tensor'] and self.config['orthogonalize_weights']:
            print("🎯 Performing ONE-STEP orthogonalization in complex space...")

            # Initialize complex weights with orthogonal phases
            self.complex_weights = np.ones(
                (n_features + 2, resol + 2, n_features + 2, resol + 2, self.outnodes + 2),
                dtype=np.complex128
            )

            # Create orthogonal phases: θ_k = (k-1) * 2π / K
            phases = np.array([(k-1) * 2 * np.pi / self.outnodes
                             for k in range(1, self.outnodes + 1)])
            complex_phases = np.exp(1j * phases)

            print(f"Orthogonal phases: {[f'{p*180/np.pi:.1f}°' for p in phases]}")

            # Apply orthogonal phases to all weight positions
            for i in range(1, n_features + 1):
                for j in range(1, resol + 1):
                    for l in range(1, n_features + 1):
                        for m in range(1, resol + 1):
                            for k in range(1, self.outnodes + 1):
                                self.complex_weights[i, j, l, m, k] = complex_phases[k-1]

            # Convert to real weights (magnitude) - STORE FOR CONSISTENT PREDICTION
            self.anti_wts = np.abs(self.complex_weights)

        else:
            # Fallback: standard initialization
            self.anti_wts = np.ones(
                (n_features + 2, resol + 2, n_features + 2, resol + 2, self.outnodes + 2),
                dtype=np.float64
            )

        print("✅ Orthogonal weights initialized")

    def compute_class_probabilities(self, features_norm, sample_idx):
        """
        FIXED: Enhanced probability computation with proper normalization
        Uses precomputed likelihoods and orthogonal weights
        """
        classval = np.ones(self.outnodes + 2)
        bins = self._find_closest_bins(features_norm[sample_idx, :])

        classval[0] = 0.0
        n_features = self.innodes

        # Use log-space to avoid underflow
        log_probs = np.zeros(self.outnodes + 2)

        for i in range(n_features):
            feature_i = i + 1
            bin_i = bins[i] + 1

            # Skip if this feature is missing
            if features_norm[sample_idx, i] == self.config['missing_value_placeholder']:
                continue

            for l in range(n_features):
                feature_l = l + 1
                bin_l = bins[l] + 1

                # Skip if this feature is missing
                if features_norm[sample_idx, l] == self.config['missing_value_placeholder']:
                    continue

                for k in range(1, self.outnodes + 1):
                    if self.global_anti_net[feature_i, bin_i, feature_l, bin_l, 0] > 0:
                        # Use precomputed likelihoods from training
                        likelihood = (self.global_anti_net[feature_i, bin_i, feature_l, bin_l, k] /
                                    self.global_anti_net[feature_i, bin_i, feature_l, bin_l, 0])
                    else:
                        likelihood = 1.0 / self.outnodes

                    # Use precomputed orthogonal weights from training
                    weight = self.anti_wts[feature_i, bin_i, feature_l, bin_l, k]

                    # Use log to prevent underflow
                    if likelihood > 0 and weight > 0:
                        log_probs[k] += np.log(likelihood) + np.log(weight)

        # Convert back from log space with numerical stability
        max_log = np.max(log_probs[1:self.outnodes+1])
        for k in range(1, self.outnodes + 1):
            classval[k] = np.exp(log_probs[k] - max_log)

        # Normalize to proper probabilities
        total = np.sum(classval[1:self.outnodes+1])
        if total > 0:
            for k in range(1, self.outnodes + 1):
                classval[k] /= total

        classval[0] = 0.0

        return classval

    def train(self, features_train, targets_train):
        """
        ONE-STEP TRAINING with orthogonal weight initialization
        Computes and stores ALL model parameters
        """
        if not self.likelihoods_computed:
            raise ValueError("Must compute global likelihoods first!")

        print("🚀 ONE-STEP training with orthogonal weights...")
        start_time = time.time()

        # Use preprocessed features from likelihood computation
        if self.training_features_norm is None:
            raise ValueError("Training features not available. Call compute_global_likelihoods first.")

        # One-step orthogonal weight initialization - STORES WEIGHTS
        self.initialize_orthogonal_weights()

        # Evaluate training accuracy using precomputed likelihoods and weights
        n_samples = len(self.training_features_norm)
        correct_predictions = 0
        probabilities = []

        for sample_idx in range(n_samples):
            classval = self.compute_class_probabilities(self.training_features_norm, sample_idx)
            probabilities.append(classval[1:self.outnodes+1])

            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            # Use the SAME encoded targets from likelihood computation
            actual = self.training_targets_encoded[sample_idx]
            predicted = self.dmyclass[kmax]

            # FIXED: Use proper comparison for arrays
            if np.abs(actual - predicted) <= self.dmyclass[0]:
                correct_predictions += 1

        accuracy = (correct_predictions / n_samples) * 100

        self.is_trained = True
        training_time = time.time() - start_time

        # Store training history
        self.training_history['training_time'] = training_time
        self.training_history['train_accuracy'] = accuracy / 100

        # Report confidence statistics
        probabilities_array = np.array(probabilities)
        max_probs = np.max(probabilities_array, axis=1)
        print(f"✅ Training completed in {training_time:.3f}s")
        print(f"🎯 Accuracy with orthogonal weights: {accuracy:.2f}% ({correct_predictions}/{n_samples})")
        print(f"📊 Confidence statistics - Min: {np.min(max_probs):.4f}, "
              f"Max: {np.max(max_probs):.4f}, Mean: {np.mean(max_probs):.4f}")

        return training_time

    def _normalize_features(self, features):
        """Normalize features using stored global min/max from training"""
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        if not hasattr(self, 'max_val') or self.max_val is None:
            raise ValueError("Model not trained. Min/max values not available.")

        n_features = self.innodes
        features_norm = np.zeros_like(features, dtype=np.float64)

        for i in range(n_features):
            feature_idx = i + 1
            feature_data = features[:, i]

            # Preserve missing values
            missing_mask = feature_data == self.config['missing_value_placeholder']

            feature_range = self.max_val[feature_idx] - self.min_val[feature_idx]
            if feature_range > 0:
                normalized = (feature_data - self.min_val[feature_idx]) / feature_range
                normalized = np.clip(normalized, 0, 1)
                normalized = normalized * (self.resolution_arr[feature_idx] - 1)

                # Restore missing values
                normalized[missing_mask] = self.config['missing_value_placeholder']
                features_norm[:, i] = normalized
            else:
                features_norm[:, i] = self.config['missing_value_placeholder'] if np.any(missing_mask) else 0

        return features_norm

    def _find_closest_bins(self, feature_vector):
        """Find closest bins for a feature vector, handling missing values"""
        n_features = self.innodes
        bins = np.zeros(n_features, dtype=np.int32)

        for i in range(n_features):
            feature_idx = i + 1
            value = feature_vector[i]

            # If value is missing, assign to bin 0
            if value == self.config['missing_value_placeholder']:
                bins[i] = 0
                continue

            resolution_val = self.resolution_arr[feature_idx]

            min_dist = 2.0 * resolution_val
            best_bin = 0

            for j in range(1, resolution_val + 1):
                dist = abs(value - self.binloc[feature_idx][j])
                if dist < min_dist:
                    min_dist = dist
                    best_bin = j

            if best_bin > 0:
                best_bin -= 1

            bins[i] = best_bin

        return bins

    def predict_proba(self, features):
        """Predict class probabilities using precomputed model parameters"""
        if not self.likelihoods_computed or not self.is_trained:
            raise ValueError("Model must be trained first!")

        # Preprocess features using the SAME preprocessor from training
        features_processed = self.preprocessor.transform_features(features)

        # Normalize using the SAME min/max from training
        features_norm = self._normalize_features(features_processed)
        n_samples = len(features_norm)

        probabilities = np.zeros((n_samples, self.outnodes))

        for sample_idx in range(n_samples):
            # Use precomputed likelihoods and orthogonal weights
            classval = self.compute_class_probabilities(features_norm, sample_idx)
            for k in range(1, self.outnodes + 1):
                probabilities[sample_idx, k-1] = classval[k]

        return probabilities

    def predict(self, features):
        """Predict class labels using precomputed model parameters"""
        probabilities = self.predict_proba(features)
        predictions_encoded = np.argmax(probabilities, axis=1) + 1

        # Convert encoded predictions back to original class labels using SAME decoder
        predictions = []
        for pred_enc in predictions_encoded:
            original_class = self.encoded_to_class.get(float(pred_enc), "Unknown")
            predictions.append(original_class)

        max_probs = np.max(probabilities, axis=1)
        print(f"📊 Prediction confidence - Min: {np.min(max_probs):.4f}, "
              f"Max: {np.max(max_probs):.4f}, Mean: {np.mean(max_probs):.4f}")

        return np.array(predictions)

    def predict_with_confidence(self, features, top_n=3):
        """
        Predict with confidence scores and top N predictions
        Returns: (predictions, probabilities, top_predictions)
        """
        probabilities = self.predict_proba(features)
        n_samples = probabilities.shape[0]

        # Get top predictions
        top_predictions = []
        top_confidences = []

        for i in range(n_samples):
            sample_probs = probabilities[i]
            # Get indices sorted by probability (descending)
            sorted_indices = np.argsort(sample_probs)[::-1]
            # Get top N predictions
            top_n_indices = sorted_indices[:top_n]

            predictions_list = []
            confidences_list = []

            for idx in top_n_indices:
                if sample_probs[idx] > 0:  # Only include meaningful predictions
                    class_label = self.encoded_to_class.get(float(idx + 1), "Unknown")
                    predictions_list.append(class_label)
                    confidences_list.append(sample_probs[idx])

            top_predictions.append(predictions_list)
            top_confidences.append(confidences_list)

        # Get primary prediction (highest probability)
        primary_predictions = self.predict(features)

        return primary_predictions, probabilities, top_predictions, top_confidences

    def evaluate(self, features, targets):
        """Evaluate model accuracy using CONSISTENT label handling"""
        # Get predictions (already in original labels)
        predictions = self.predict(features)

        # Convert targets to original labels for comparison
        targets_original = self.preprocessor.inverse_transform_targets(
            self.preprocessor.transform_targets(targets)
        )

        # FIXED: Ensure both are 1D arrays for comparison
        predictions = np.array(predictions).flatten()
        targets_original = np.array(targets_original).flatten()

        accuracy = accuracy_score(targets_original, predictions)
        self.training_history['test_accuracy'] = accuracy

        print(f"🔍 Evaluation Details:")
        print(f"   Predictions sample: {predictions[:10]}")
        print(f"   Actual targets sample: {targets_original[:10]}")
        print(f"   Match count: {np.sum(predictions == targets_original)}/{len(predictions)}")

        return accuracy

    def plot_feature_importance(self):
        """Plot feature importance using original feature names"""
        if not self.is_trained:
            print("Model not trained yet")
            return

        try:
            n_features = self.innodes
            feature_importance = np.zeros(n_features)

            for i in range(n_features):
                # Simple measure: sum of likelihoods for each feature
                feature_importance[i] = np.sum(self.global_anti_net[i+1, :, :, :, :])

            # Normalize
            feature_importance = feature_importance / np.sum(feature_importance)

            # Use original feature names if available
            if self.feature_names is not None and len(self.feature_names) == n_features:
                feature_labels = self.feature_names
            else:
                feature_labels = [f'Feature {i+1}' for i in range(n_features)]

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(feature_labels))

            bars = ax.barh(y_pos, feature_importance)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_labels)
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance (Based on Global Likelihoods)')

            # Add value annotations
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{width:.3f}', ha='left', va='center')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting feature importance: {e}")

    def save_model(self, filepath):
        """Save the complete model state including hyperparameters and encoders"""
        model_state = {
            'config': self.config,
            'preprocessor': self.preprocessor,
            'global_anti_net': self.global_anti_net,
            'binloc': self.binloc,
            'max_val': self.max_val,
            'min_val': self.min_val,
            'dmyclass': self.dmyclass,
            'resolution_arr': self.resolution_arr,
            'anti_wts': self.anti_wts,
            'complex_weights': self.complex_weights,
            'innodes': self.innodes,
            'outnodes': self.outnodes,
            'class_to_encoded': self.class_to_encoded,
            'encoded_to_class': self.encoded_to_class,
            'is_trained': self.is_trained,
            'likelihoods_computed': self.likelihoods_computed,
            'feature_names': self.feature_names,  # Store actual feature names
            'target_name': self.target_name,
            'training_features_norm': self.training_features_norm,
            'training_targets_encoded': self.training_targets_encoded,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'selected_features': self.selected_features,
            'target_column_name': self.target_column_name
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        # Also save hyperparameters separately for easy access
        hp_filepath = filepath.replace('.pkl', '_hp.json').replace('.bin', '_hp.json')
        with open(hp_filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ Model saved to {filepath}")
        print(f"✅ Feature names: {self.feature_names}")

    def load_model(self, filepath):
        """Load complete model state including hyperparameters and encoders"""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)

        # Restore all attributes
        for key, value in model_state.items():
            setattr(self, key, value)

        print(f"✅ Model loaded from {filepath}")
        print(f"📊 Model info: {self.model_metadata['data_name']}, "
              f"Trained: {self.is_trained}, Features: {self.innodes}, Classes: {self.outnodes}")
        print(f"📊 Feature names: {self.feature_names}")

    def save_hyperparameters(self, filepath=None):
        """Save hyperparameters to JSON file"""
        if filepath is None:
            data_name = self.model_metadata.get('data_name', 'unknown')
            filepath = f"{data_name}_hp.json"

        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ Hyperparameters saved to {filepath}")
        return filepath

    def load_hyperparameters(self, filepath):
        """Load hyperparameters from JSON file"""
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)

        self.config.update(loaded_config)
        print(f"✅ Hyperparameters loaded from {filepath}")
        return self.config

    def predict_on_file(self, filepath, output_file=None, features_to_use=None, has_target=False, target_column=None):
        """
        Comprehensive prediction on file with feature selection
        USES EXACT FEATURE NAMES AND ORDERING FROM TRAINING
        """
        try:
            # Load data
            df = pd.read_csv(filepath)
            original_columns = df.columns.tolist()

            print(f"📊 Prediction file columns: {original_columns}")
            print(f"📊 Model feature names: {self.feature_names}")

            # Determine features to use - ALWAYS USE MODEL'S FEATURE NAMES IN TRAINING ORDER
            if features_to_use is None:
                features_to_use = self.feature_names  # Use exact feature names from training

            # Remove target column if specified
            if target_column and target_column in features_to_use:
                features_to_use = [f for f in features_to_use if f != target_column]

            # STRICT FEATURE MATCHING: Use exact feature names and ordering from training
            available_features = []
            missing_features = []

            for feature in features_to_use:
                if feature in df.columns:
                    available_features.append(feature)
                else:
                    missing_features.append(feature)

            if missing_features:
                print(f"❌ Missing features in prediction file: {missing_features}")
                print(f"   Available features: {df.columns.tolist()}")
                raise ValueError(f"Missing features in prediction file: {missing_features}")

            if len(available_features) != len(features_to_use):
                print(f"❌ Feature count mismatch. Expected: {len(features_to_use)}, Found: {len(available_features)}")
                raise ValueError(f"Feature count mismatch. Expected: {len(features_to_use)}, Found: {len(available_features)}")

            print(f"✅ All {len(available_features)} features matched by exact names")
            print(f"🔧 Using features in training order: {available_features}")

            # Extract features for prediction - IN EXACT TRAINING ORDER
            prediction_features = df[available_features]

            print(f"🔮 Making predictions using {len(available_features)} features")
            print(f"   Features used (in training order): {available_features}")

            # Make predictions
            primary_predictions, probabilities, top_predictions, top_confidences = self.predict_with_confidence(prediction_features, top_n=3)

            # Create results dataframe
            results_df = df.copy()

            # Add prediction columns
            results_df['Prediction'] = primary_predictions
            results_df['Confidence'] = [max(probs) for probs in probabilities]

            # Add top 3 predictions with confidences
            for i in range(3):
                pred_col = f'Top_{i+1}_Prediction'
                conf_col = f'Top_{i+1}_Confidence'

                pred_values = []
                conf_values = []

                for j in range(len(top_predictions)):
                    if i < len(top_predictions[j]):
                        pred_values.append(top_predictions[j][i])
                        conf_values.append(top_confidences[j][i])
                    else:
                        pred_values.append(None)
                        conf_values.append(None)

                results_df[pred_col] = pred_values
                results_df[conf_col] = conf_values

            # Sort by confidence (descending)
            results_df = results_df.sort_values('Confidence', ascending=False)

            # Save results
            if output_file is None:
                base_name = os.path.splitext(filepath)[0]
                output_file = f"{base_name}_predictions.csv"

            results_df.to_csv(output_file, index=False)
            print(f"✅ Predictions saved to {output_file}")
            print(f"📊 Results include: {len(available_features)} features, {len(results_df)} samples")
            print(f"🎯 Confidence range: {results_df['Confidence'].min():.3f} - {results_df['Confidence'].max():.3f}")

            return results_df

        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            print(f"🔍 Debug info:")
            print(f"   File: {filepath}")
            print(f"   Model features: {self.feature_names}")
            if 'df' in locals():
                print(f"   File columns: {df.columns.tolist()}")
            raise


def main():
    """Main function with command-line interface - UPDATED with optional split"""
    parser = argparse.ArgumentParser(
        description="Enhanced CT-DBNN Classifier with Comprehensive Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with GUI
  python ct_dbnn_enhanced.py --gui

  # Train on Iris dataset with no test split (use all data for training)
  python ct_dbnn_enhanced.py --dataset iris --train

  # Train with 80% training data, 20% testing
  python ct_dbnn_enhanced.py --dataset iris --train --train-split 80

  # Train with custom parameters and no test split
  python ct_dbnn_enhanced.py --csv data.csv --features col1,col2,col3 --target outcome --train --no-split

  # Load and evaluate model
  python ct_dbnn_enhanced.py --load-model model.pkl --evaluate

  # Predict on new data
  python ct_dbnn_enhanced.py --load-model model.pkl --predict new_data.csv
        """
    )

    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--dataset', help='UCI dataset to use (name from repository or custom)')
    data_group.add_argument('--csv', help='Custom CSV file path')
    data_group.add_argument('--features', help='Comma-separated list of features to use for training/prediction')
    data_group.add_argument('--target', help='Target column name')

    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--resol', type=int, default=8,
                           help='Resolution for feature discretization (default: 8)')
    model_group.add_argument('--no-complex-tensor', action='store_true',
                           help='Disable complex tensor operations')
    model_group.add_argument('--no-orthogonal-weights', action='store_true',
                           help='Disable orthogonal weight initialization')
    model_group.add_argument('--smoothing-factor', type=float, default=1e-8,
                           help='Smoothing factor for probabilities (default: 1e-8)')
    model_group.add_argument('--no-parallel', action='store_true',
                           help='Disable parallel processing')
    model_group.add_argument('--n-jobs', type=int, default=-1,
                           help='Number of parallel jobs (default: -1 for all)')

    # Training options
    training_group = parser.add_argument_group('Training Options')
    training_group.add_argument('--train', action='store_true', help='Train model on dataset')
    training_group.add_argument('--no-split', action='store_true',
                               help='Use all data for training (no test split)')
    training_group.add_argument('--train-split', type=float, default=80,
                               help='Percentage of data to use for training (default: 80)')
    training_group.add_argument('--evaluate', action='store_true',
                               help='Evaluate model on test set (requires train/test split)')

    # Operation modes
    operation_group = parser.add_argument_group('Operation Modes')
    operation_group.add_argument('--gui', action='store_true',
                               help='Launch GUI interface')
    operation_group.add_argument('--load-model', help='Load model from file')
    operation_group.add_argument('--save-model', help='Save model to file')
    operation_group.add_argument('--save-hp', help='Save hyperparameters to JSON file')
    operation_group.add_argument('--load-hp', help='Load hyperparameters from JSON file')

    # Prediction options
    pred_group = parser.add_argument_group('Prediction Options')
    pred_group.add_argument('--predict', help='Run prediction on specified file')
    pred_group.add_argument('--has-target', action='store_true',
                          help='Input file has target column (will be ignored for prediction)')
    pred_group.add_argument('--target-col', help='Target column name in prediction file')

    # Additional options
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')

    args = parser.parse_args()

    # GUI mode
    if args.gui:
        if not GUI_AVAILABLE:
            print("GUI libraries not available. Please install tkinter, matplotlib, and pandas.")
            return 1

        app = EnhancedGUI()
        app.run()
        return 0

    # Command-line mode - Prediction
    if args.predict:
        if not args.load_model:
            print("❌ Please specify --load-model for prediction")
            return 1

        model = ParallelCTDBNN()
        model.load_model(args.load_model)

        features_to_use = None
        if args.features:
            features_to_use = [f.strip() for f in args.features.split(',')]

        try:
            results_df = model.predict_on_file(
                filepath=args.predict,
                features_to_use=features_to_use,
                has_target=args.has_target,
                target_column=args.target_col
            )
            print(f"✅ Prediction completed. Results saved to: {args.predict.replace('.csv', '_predictions.csv')}")
            return 0
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return 1

    # Command-line mode - Load existing model
    if args.load_model:
        model = ParallelCTDBNN()
        model.load_model(args.load_model)

        if args.evaluate:
            if hasattr(model, 'X_test') and model.X_test is not None:
                accuracy = model.evaluate(model.X_test, model.y_test)
                print(f"Test accuracy: {accuracy:.3f}")
            else:
                print("❌ Cannot evaluate: No test data available in loaded model")

        if args.save_model:
            model.save_model(args.save_model)

        return 0

    # Training mode
    if args.dataset or args.csv:
        config = {
            'resol': args.resol,
            'use_complex_tensor': not args.no_complex_tensor,
            'orthogonalize_weights': not args.no_orthogonal_weights,
            'smoothing_factor': args.smoothing_factor,
            'parallel_processing': not args.no_parallel,
            'n_jobs': args.n_jobs
        }

        # Load data
        if args.dataset:
            print(f"Loading dataset: {args.dataset}")
            if args.dataset == "iris":
                data = load_iris()
            elif args.dataset == "wine":
                data = load_wine()
            elif args.dataset == "breast_cancer":
                data = load_breast_cancer()
            elif args.dataset == "diabetes":
                info = UCI_DATASETS["diabetes"]
                df = UCIDatasetLoader.download_uci_data(info)
                data = type('DiabetesData', (), {})()
                data.data = df.iloc[:, :-1].values
                data.target = df.iloc[:, -1].values
            else:
                info = UCIDatasetLoader.load_any_uci_dataset(args.dataset)
                if info:
                    df = UCIDatasetLoader.download_uci_data(info)
                    if df is not None:
                        data = type('CustomData', (), {})()
                        data.data = df.iloc[:, :-1].values
                        data.target = df.iloc[:, -1].values
                    else:
                        print(f"Could not download dataset: {args.dataset}")
                        return 1
                else:
                    print(f"Dataset {args.dataset} not found")
                    return 1

            features = data.data
            targets = data.target

        elif args.csv:
            if not args.target:
                print("Please specify --target for CSV files")
                return 1

            print(f"Loading CSV: {args.csv}")
            df = pd.read_csv(args.csv)

            if args.features:
                features_to_use = [f.strip() for f in args.features.split(',')]
                missing_features = [f for f in features_to_use if f not in df.columns]
                if missing_features:
                    print(f"❌ Missing features: {missing_features}")
                    return 1
                features = df[features_to_use].values
            else:
                features_to_use = [col for col in df.columns if col != args.target]
                features = df[features_to_use].values

            targets = df[args.target].values
            feature_names = features_to_use if args.features else [col for col in df.columns if col != args.target]

        # Handle train/test split based on arguments
        if args.no_split:
            X_train = features
            y_train = targets
            X_test = None
            y_test = None
            print("✅ Using 100% of data for training (no test split)")
            print(f"   Training samples: {len(X_train)}")
        else:
            test_size = (100 - args.train_split) / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets,
                test_size=test_size,
                random_state=args.random_state,
                stratify=targets
            )
            print(f"✅ Data split: {args.train_split}% training, {100-args.train_split}% testing")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Testing samples: {len(X_test)}")

        # Initialize and train model
        model = ParallelCTDBNN(config)

        if 'feature_names' in locals():
            model.feature_names = feature_names
            model.selected_features = feature_names
            model.target_column_name = args.target

        print("Computing global likelihoods...")
        model.compute_global_likelihoods(X_train, y_train, feature_names if 'feature_names' in locals() else None)

        if args.train:
            print("Training model...")
            model.train(X_train, y_train)

            if args.evaluate:
                if X_test is not None and y_test is not None:
                    accuracy = model.evaluate(X_test, y_test)
                    print(f"Test accuracy: {accuracy:.3f}")
                else:
                    print("❌ Cannot evaluate: No test data available (use --train-split instead of --no-split)")

        if args.save_model:
            model.save_model(args.save_model)

        if args.save_hp:
            model.save_hyperparameters(args.save_hp)

    elif args.load_hp:
        model = ParallelCTDBNN()
        model.load_hyperparameters(args.load_hp)
        print(f"Loaded hyperparameters: {model.config}")

    else:
        parser.print_help()

    return 0


class EnhancedGUI:
    """
    Enhanced GUI with UCI dataset support and hyperparameter management
    """
    def __init__(self):
        if not GUI_AVAILABLE:
            print("GUI libraries not available. Running in command-line mode.")
            return

        self.root = tk.Tk()
        self.root.title("Enhanced CT-DBNN Classifier")
        self.root.geometry("1200x800")

        # Configure style
        self.style = ttk.Style()
        self.style.configure('TNotebook.Tab', font=('Arial', 10, 'bold'))
        self.style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Arial', 10, 'bold'))

        # Model instance
        self.model = None
        self.current_dataset = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_selection = {}

        self.create_widgets()

    def on_dataset_selected(self, event):
        """Handle dataset selection from combo box"""
        dataset_name = self.dataset_var.get()
        if dataset_name in UCI_DATASETS:
            info = UCI_DATASETS[dataset_name]
            self.display_dataset_info(info)

    def on_custom_dataset_entered(self, event):
        """Handle custom dataset entry"""
        self.fetch_custom_dataset()

    def load_uci_dataset(self):
        """Load selected UCI dataset"""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset")
            return

        try:
            self.update_status(f"Loading {dataset_name} dataset...")
            self.log_message(f"📥 Loading UCI dataset: {dataset_name}")

            # Reset everything for fresh start
            self.reset_everything()

            # Load the dataset
            if dataset_name == "iris":
                data = load_iris()
            elif dataset_name == "wine":
                data = load_wine()
            elif dataset_name == "breast_cancer":
                data = load_breast_cancer()
            elif dataset_name == "diabetes":
                info = UCI_DATASETS["diabetes"]
                df = UCIDatasetLoader.download_uci_data(info)
                data = type('DiabetesData', (), {})()
                data.data = df.iloc[:, :-1].values
                data.target = df.iloc[:, -1].values
                data.feature_names = df.columns[:-1].tolist()
            else:
                # Try custom UCI dataset
                info = UCIDatasetLoader.load_any_uci_dataset(dataset_name)
                if info:
                    df = UCIDatasetLoader.download_uci_data(info)
                    if df is not None:
                        data = type('CustomData', (), {})()
                        data.data = df.iloc[:, :-1].values
                        data.target = df.iloc[:, -1].values
                        data.feature_names = df.columns[:-1].tolist()
                    else:
                        messagebox.showerror("Error", f"Could not download dataset: {dataset_name}")
                        return
                else:
                    messagebox.showerror("Error", f"Dataset {dataset_name} not found")
                    return

            self.current_dataset = data

            # Update available features list
            self.available_listbox.delete(0, tk.END)
            if hasattr(data, 'feature_names'):
                for feature in data.feature_names:
                    self.available_listbox.insert(tk.END, feature)
            elif hasattr(data, 'data'):
                n_features = data.data.shape[1]
                for i in range(n_features):
                    self.available_listbox.insert(tk.END, f'Feature_{i+1}')

            # Update target combos
            if hasattr(data, 'feature_names'):
                self.target_combo['values'] = data.feature_names
                self.pred_target_combo['values'] = data.feature_names

            # Update data preview
            self.update_data_preview(data.data, data.target if hasattr(data, 'target') else None)

            self.log_message(f"✅ Dataset loaded successfully: {dataset_name}")
            if hasattr(data, 'data'):
                self.log_message(f"   Samples: {data.data.shape[0]}, Features: {data.data.shape[1]}")

            # Try to auto-load hyperparameters
            self.auto_load_parameters()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.log_message(f"❌ Error loading dataset: {str(e)}")

    def fetch_custom_dataset(self):
        """Fetch custom UCI dataset by name"""
        dataset_name = self.custom_dataset_var.get().strip()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please enter a dataset name")
            return

        try:
            self.update_status(f"Fetching {dataset_name} from UCI...")
            self.log_message(f"🌐 Fetching UCI dataset: {dataset_name}")

            info = UCIDatasetLoader.load_any_uci_dataset(dataset_name)
            if info:
                self.display_dataset_info(info)
                self.dataset_var.set('')  # Clear the predefined selection
                self.log_message(f"✅ Dataset info fetched: {info['name']}")
            else:
                messagebox.showwarning("Warning", f"Dataset '{dataset_name}' not found in UCI repository")
                self.log_message(f"❌ Dataset not found: {dataset_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch dataset info: {str(e)}")
            self.log_message(f"❌ Error fetching dataset: {str(e)}")

    def display_dataset_info(self, info):
        """Display dataset information in the info box"""
        self.dataset_info.config(state='normal')
        self.dataset_info.delete('1.0', tk.END)

        info_text = f"""Dataset: {info['name']}
    Description: {info['description']}
    Features: {info['features']}
    Samples: {info['samples']}
    Best Known Accuracy: {info['best_accuracy']}
    Reference: {info['reference']}
    URL: {info['url']}
    """
        self.dataset_info.insert('1.0', info_text)
        self.dataset_info.config(state='disabled')

    def save_model_file(self):
        """Save model to file with intelligent filename recommendations"""
        if not self.model or not self.model.is_trained:
            messagebox.showwarning("Warning", "No trained model to save")
            return

        # Generate intelligent default filename
        default_filename = self._generate_model_filename("save")

        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if filepath:
            try:
                self.model.save_model(filepath)
                self.log_message(f"✅ Model saved to: {filepath}")

                # Also save hyperparameters with matching name
                hp_filepath = filepath.replace('.pkl', '_hp.json')
                self.model.save_hyperparameters(hp_filepath)
                self.log_message(f"✅ Hyperparameters saved to: {hp_filepath}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                self.log_message(f"❌ Error saving model: {str(e)}")

    def load_model_file(self):
        """Load model from file with intelligent filename suggestions"""
        # Generate suggested filename pattern for file dialog
        suggested_pattern = self._generate_model_filename("load") + "*.pkl"

        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[
                ("Model files", "*.pkl"),
                ("Hyperparameter files", "*_hp.json"),
                ("All files", "*.*")
            ],
            initialfile=suggested_pattern
        )

        if filepath:
            try:
                if filepath.endswith('_hp.json'):
                    # Load hyperparameters only
                    self.model = ParallelCTDBNN()
                    self.model.load_hyperparameters(filepath)
                    self.log_message(f"✅ Hyperparameters loaded from: {filepath}")
                    self.log_message(f"   Configuration: {self.model.config}")

                    # Try to find corresponding model file
                    model_filepath = filepath.replace('_hp.json', '.pkl')
                    if os.path.exists(model_filepath):
                        response = messagebox.askyesno(
                            "Load Model",
                            f"Found corresponding model file:\n{os.path.basename(model_filepath)}\n\nDo you want to load the model too?"
                        )
                        if response:
                            self.model.load_model(model_filepath)
                            self.log_message(f"✅ Model loaded from: {model_filepath}")
                            self._update_gui_after_model_load()
                else:
                    # Load complete model
                    self.model = ParallelCTDBNN()
                    self.model.load_model(filepath)
                    self.log_message(f"✅ Model loaded from: {filepath}")
                    self._update_gui_after_model_load()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.log_message(f"❌ Error loading model: {str(e)}")

    def _generate_model_filename(self, action_type="save"):
        """Generate intelligent filename based on dataset and model characteristics"""
        base_name = "untrained_model"

        # Try to get dataset name from various sources
        if hasattr(self, 'model') and self.model and hasattr(self.model, 'model_metadata'):
            dataset_name = self.model.model_metadata.get('data_name', 'unknown_dataset')
            base_name = dataset_name.replace(' ', '_').lower()
        elif hasattr(self, 'current_dataset') and self.current_dataset:
            if hasattr(self.current_dataset, 'name'):
                base_name = self.current_dataset.name.replace(' ', '_').lower()
            elif hasattr(self.current_dataset, '__class__'):
                base_name = self.current_dataset.__class__.__name__.lower()

        # Add model characteristics if available
        model_suffix = ""
        if hasattr(self, 'model') and self.model:
            if self.model.is_trained:
                accuracy = "untrained"
                if hasattr(self.model, 'training_history') and 'train_accuracy' in self.model.training_history:
                    acc = self.model.training_history['train_accuracy']
                    accuracy = f"acc{acc:.3f}".replace('0.', '').replace('.', '')
                elif hasattr(self.model, 'training_history') and 'test_accuracy' in self.model.training_history:
                    acc = self.model.training_history['test_accuracy']
                    accuracy = f"acc{acc:.3f}".replace('0.', '').replace('.', '')

                features = f"f{self.model.innodes}" if hasattr(self.model, 'innodes') else ""
                classes = f"c{self.model.outnodes}" if hasattr(self.model, 'outnodes') else ""

                model_suffix = f"_{features}_{classes}_{accuracy}"
            else:
                model_suffix = "_untrained"

        # Add timestamp for save operations to avoid overwrites
        if action_type == "save":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            return f"{base_name}{model_suffix}_{timestamp}.pkl"
        else:
            # For load operations, suggest pattern without timestamp
            return f"{base_name}{model_suffix}"

    def _update_gui_after_model_load(self):
        """Update GUI elements after loading a model"""
        if not self.model:
            return

        # Update feature selection if model has feature names
        if hasattr(self.model, 'feature_names') and self.model.feature_names:
            self.auto_select_features()
            self.log_message(f"📊 Auto-selected features: {self.model.feature_names}")

        # Update hyperparameter controls if model has config
        if hasattr(self.model, 'config'):
            self._update_parameter_controls()

        # Update status
        dataset_name = getattr(self.model.model_metadata, 'data_name', 'unknown') if hasattr(self.model, 'model_metadata') else 'unknown'
        self.log_message(f"🎯 Loaded model: {dataset_name}, Features: {self.model.innodes}, Classes: {self.model.outnodes}")

    def _update_parameter_controls(self):
        """Update parameter controls from loaded model configuration"""
        if not hasattr(self, 'model') or not self.model or not hasattr(self.model, 'config'):
            return

        try:
            for key, var in self.param_vars.items():
                if key in self.model.config:
                    var.set(str(self.model.config[key]))

            self.log_message("✅ Hyperparameters updated from loaded model")

        except Exception as e:
            self.log_message(f"⚠️  Could not update all parameters: {e}")


    def update_data_preview(self, features, targets):
        """Update data preview treeview"""
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        if features is None:
            return

        n_features = features.shape[1]

        for i in range(n_features):
            feature_data = features[:, i]

            # Calculate statistics
            feature_type = "Numeric" if np.issubdtype(feature_data.dtype, np.number) else "Categorical"
            missing_count = np.sum(pd.isna(feature_data)) if hasattr(pd, 'isna') else np.sum(feature_data == None)
            unique_count = len(np.unique(feature_data))

            if np.issubdtype(feature_data.dtype, np.number):
                data_range = f"{np.min(feature_data):.3f} - {np.max(feature_data):.3f}"
            else:
                data_range = f"{unique_count} categories"

            # Get feature name
            if hasattr(self.current_dataset, 'feature_names') and i < len(self.current_dataset.feature_names):
                feature_name = self.current_dataset.feature_names[i]
            else:
                feature_name = f"Feature_{i+1}"

            self.data_tree.insert("", "end", values=(feature_name, feature_type, missing_count, unique_count, data_range))

    def reset_everything(self):
        """Reset all GUI state"""
        self.model = None
        self.current_dataset = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.feature_selection = {}

        # Clear lists
        self.available_listbox.delete(0, tk.END)
        self.selected_listbox.delete(0, tk.END)
        self.target_var.set('')

        # Clear data preview
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)

        # Clear logs
        self.clear_logs()

    def clear_logs(self):
        """Clear all log text widgets"""
        for text_widget in [self.log_text, self.pred_results_text, self.results_text]:
            text_widget.config(state='normal')
            text_widget.delete('1.0', tk.END)
            text_widget.config(state='disabled')

    def log_message(self, message):
        """Add message to training log"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(message)
        self.root.update()

    def apply_parameters(self):
        """Apply hyperparameters from GUI"""
        try:
            self.model_config = {
                'resol': int(self.param_vars['resol'].get()),
                'use_complex_tensor': self.param_vars['use_complex_tensor'].get().lower() == 'true',
                'orthogonalize_weights': self.param_vars['orthogonalize_weights'].get().lower() == 'true',
                'smoothing_factor': float(self.param_vars['smoothing_factor'].get()),
                'parallel_processing': self.param_vars['parallel_processing'].get().lower() == 'true',
                'n_jobs': int(self.param_vars['n_jobs'].get()),
                'batch_size': int(self.param_vars['batch_size'].get()),
                'missing_value_placeholder': float(self.param_vars['missing_value_placeholder'].get())
            }
            self.log_message("✅ Hyperparameters applied successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameter values: {str(e)}")
            self.log_message(f"❌ Error applying parameters: {str(e)}")

    def reset_parameters(self):
        """Reset parameters to defaults"""
        defaults = {
            'resol': '8',
            'use_complex_tensor': 'True',
            'orthogonalize_weights': 'True',
            'smoothing_factor': '1e-8',
            'parallel_processing': 'True',
            'n_jobs': '-1',
            'batch_size': '1000',
            'missing_value_placeholder': '-99999'
        }

        for key, value in defaults.items():
            self.param_vars[key].set(value)

        self.log_message("✅ Parameters reset to defaults")

    def save_parameters(self):
        """Save parameters to file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Hyperparameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.apply_parameters()  # Ensure current values are applied
                with open(filepath, 'w') as f:
                    json.dump(self.model_config, f, indent=2)
                self.log_message(f"✅ Hyperparameters saved to: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save parameters: {str(e)}")

    def load_parameters(self):
        """Load parameters from file"""
        filepath = filedialog.askopenfilename(
            title="Load Hyperparameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    loaded_config = json.load(f)

                for key, value in loaded_config.items():
                    if key in self.param_vars:
                        self.param_vars[key].set(str(value))

                self.log_message(f"✅ Hyperparameters loaded from: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {str(e)}")

    def auto_load_parameters(self):
        """Auto-load parameters based on dataset"""
        if not hasattr(self, 'current_dataset'):
            return

        # Simple auto-config based on dataset size
        if hasattr(self.current_dataset, 'data'):
            n_samples, n_features = self.current_dataset.data.shape

            if n_features > 20:
                # High-dimensional data
                self.param_vars['resol'].set('6')
                self.param_vars['batch_size'].set('500')
            elif n_samples > 10000:
                # Large dataset
                self.param_vars['batch_size'].set('2000')

            self.log_message("✅ Auto-loaded parameters based on dataset characteristics")

    def compute_likelihoods(self):
        """Compute global likelihoods"""
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize model first")
            return

        try:
            self.log_message("🔧 Computing global likelihoods...")
            self.model.compute_global_likelihoods(self.X_train, self.y_train)
            self.log_message("✅ Global likelihoods computed successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute likelihoods: {str(e)}")
            self.log_message(f"❌ Error computing likelihoods: {str(e)}")

    def train_model(self):
        """Train the model"""
        if not self.model:
            messagebox.showwarning("Warning", "Please initialize model first")
            return

        try:
            self.log_message("🚀 Training model...")
            training_time = self.model.train(self.X_train, self.y_train)
            self.log_message(f"✅ Training completed in {training_time:.2f}s")
            self.log_message(f"🎯 Training accuracy: {self.model.training_history['train_accuracy']:.3f}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.log_message(f"❌ Error training model: {str(e)}")

    def evaluate_model(self):
        """Evaluate the model - UPDATED to handle no test split"""
        if not self.model or not self.model.is_trained:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        if self.X_test is None or self.y_test is None:
            messagebox.showwarning("Warning", "No test data available. Please enable train/test split during initialization.")
            return

        try:
            self.log_message("🔍 Evaluating model...")
            accuracy = self.model.evaluate(self.X_test, self.y_test)
            self.log_message(f"✅ Evaluation completed")
            self.log_message(f"🎯 Test accuracy: {accuracy:.3f}")

            self.results_text.config(state='normal')
            self.results_text.delete('1.0', tk.END)

            if hasattr(self.model, 'training_history') and 'train_accuracy' in self.model.training_history:
                train_acc = self.model.training_history['train_accuracy']
            else:
                train_acc = "N/A"

            results_summary = f"""MODEL EVALUATION RESULTS
    {'='*40}
    Dataset: {self.model.model_metadata['data_name']}
    Training Accuracy: {train_acc}
    Test Accuracy: {accuracy:.3f}
    Training Time: {self.model.training_history['training_time']:.2f}s
    Features: {self.model.innodes}
    Classes: {self.model.outnodes}
    Training Samples: {len(self.X_train)}
    Test Samples: {len(self.X_test)}
    """
            self.results_text.insert('1.0', results_summary)
            self.results_text.config(state='disabled')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to evaluate model: {str(e)}")
            self.log_message(f"❌ Error evaluating model: {str(e)}")

    def show_feature_importance(self):
        """Show feature importance plot"""
        if not self.model or not self.model.is_trained:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        try:
            self.model.plot_feature_importance()
            self.log_message("✅ Feature importance plot displayed")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show feature importance: {str(e)}")
            self.log_message(f"❌ Error showing feature importance: {str(e)}")

    def show_confusion_matrix(self):
        """Show confusion matrix"""
        if not self.model or not self.model.is_trained:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        try:
            # Make predictions
            predictions = self.model.predict(self.X_test)
            targets_original = self.model.preprocessor.inverse_transform_targets(
                self.model.preprocessor.transform_targets(self.y_test)
            )

            # Create confusion matrix
            cm = confusion_matrix(targets_original, predictions)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # Add labels
            classes = np.unique(np.concatenate([targets_original, predictions]))
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title='Confusion Matrix',
                   ylabel='True label',
                   xlabel='Predicted label')

            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.show()
            self.log_message("✅ Confusion matrix displayed")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show confusion matrix: {str(e)}")
            self.log_message(f"❌ Error showing confusion matrix: {str(e)}")

    def export_results(self):
        """Export results to file"""
        if not self.model or not self.model.is_trained:
            messagebox.showwarning("Warning", "No results to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    f.write(f"CT-DBNN Model Results\n")
                    f.write(f"====================\n\n")
                    f.write(f"Dataset: {self.model.model_metadata['data_name']}\n")
                    f.write(f"Training Accuracy: {self.model.training_history['train_accuracy']:.3f}\n")
                    if 'test_accuracy' in self.model.training_history:
                        f.write(f"Test Accuracy: {self.model.training_history['test_accuracy']:.3f}\n")
                    f.write(f"Training Time: {self.model.training_history['training_time']:.2f}s\n")
                    f.write(f"Features: {self.model.innodes}\n")
                    f.write(f"Classes: {self.model.outnodes}\n")
                    f.write(f"\nHyperparameters:\n")
                    for key, value in self.model.config.items():
                        f.write(f"  {key}: {value}\n")

                self.log_message(f"✅ Results exported to: {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                self.log_message(f"❌ Error exporting results: {str(e)}")
    #-------------For Command Line Inteface -----------
    def save_model(self, filepath=None):
        """Save the complete model state with intelligent default naming"""
        if filepath is None:
            # Generate intelligent default filename
            dataset_name = self.model_metadata.get('data_name', 'unknown_dataset').replace(' ', '_').lower()
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if self.is_trained:
                accuracy = "untrained"
                if 'train_accuracy' in self.training_history:
                    acc = self.training_history['train_accuracy']
                    accuracy = f"acc{acc:.3f}".replace('0.', '').replace('.', '')
                filepath = f"{dataset_name}_model_f{self.innodes}_c{self.outnodes}_{accuracy}_{timestamp}.pkl"
            else:
                filepath = f"{dataset_name}_untrained_{timestamp}.pkl"

        model_state = {
            'config': self.config,
            'preprocessor': self.preprocessor,
            'global_anti_net': self.global_anti_net,
            'binloc': self.binloc,
            'max_val': self.max_val,
            'min_val': self.min_val,
            'dmyclass': self.dmyclass,
            'resolution_arr': self.resolution_arr,
            'anti_wts': self.anti_wts,
            'complex_weights': self.complex_weights,
            'innodes': self.innodes,
            'outnodes': self.outnodes,
            'class_to_encoded': self.class_to_encoded,
            'encoded_to_class': self.encoded_to_class,
            'is_trained': self.is_trained,
            'likelihoods_computed': self.likelihoods_computed,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_features_norm': self.training_features_norm,
            'training_targets_encoded': self.training_targets_encoded,
            'training_history': self.training_history,
            'model_metadata': self.model_metadata,
            'selected_features': self.selected_features,
            'target_column_name': self.target_column_name
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)

        # Also save hyperparameters separately for easy access
        hp_filepath = filepath.replace('.pkl', '_hp.json')
        with open(hp_filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"✅ Model saved to {filepath}")
        print(f"✅ Hyperparameters saved to {hp_filepath}")
        print(f"✅ Feature names: {self.feature_names}")

        return filepath
    #-------------For Command Line Inteface -----------

    def create_widgets(self):
        """Create enhanced GUI with tabs"""
        # Main container frame to hold everything
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill='both', expand=True, pady=(0, 10))

        # Data Management Tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="📊 Data Management")
        self.create_data_tab(data_frame)

        # Feature Selection Tab
        feature_frame = ttk.Frame(notebook)
        notebook.add(feature_frame, text="🎯 Feature Selection")
        self.create_feature_tab(feature_frame)

        # Hyperparameters Tab
        param_frame = ttk.Frame(notebook)
        notebook.add(param_frame, text="⚙️ Hyperparameters")
        self.create_parameters_tab(param_frame)

        # Training Tab
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="🚀 Training")
        self.create_training_tab(train_frame)

        # Prediction Tab
        predict_frame = ttk.Frame(notebook)
        notebook.add(predict_frame, text="🔮 Prediction")
        self.create_prediction_tab(predict_frame)

        # Results Tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="📈 Results")
        self.create_results_tab(results_frame)

        # Bottom frame for status and exit button
        bottom_frame = ttk.Frame(main_container)
        bottom_frame.pack(fill='x', side='bottom')

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(bottom_frame, textvariable=self.status_var, relief='sunken')
        status_bar.pack(side='left', fill='x', expand=True)

        # Exit button
        exit_button = ttk.Button(bottom_frame, text="Exit", command=self.exit_application)
        exit_button.pack(side='right', padx=(10, 0))

    def exit_application(self):
        """Exit the application"""
        if messagebox.askokcancel("Exit", "Are you sure you want to exit?"):
            self.root.destroy()

    def create_data_tab(self, parent):
        """Create data management tab with UCI dataset support"""
        # Main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # UCI Dataset Selection
        uci_frame = ttk.LabelFrame(main_frame, text="🌐 UCI Dataset Repository", padding=10)
        uci_frame.pack(fill='x', pady=(0, 10))

        # Dataset selection combo
        ttk.Label(uci_frame, text="Select UCI Dataset:").grid(row=0, column=0, sticky='w', padx=(0, 10))
        self.dataset_var = tk.StringVar()
        dataset_combo = ttk.Combobox(uci_frame, textvariable=self.dataset_var,
                                   values=list(UCI_DATASETS.keys()), state='readonly')
        dataset_combo.grid(row=0, column=1, sticky='ew', padx=(0, 10))
        dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)

        # Custom UCI dataset entry
        ttk.Label(uci_frame, text="Or enter UCI dataset name:").grid(row=1, column=0, sticky='w', padx=(0, 10), pady=(10, 0))
        self.custom_dataset_var = tk.StringVar()
        custom_entry = ttk.Entry(uci_frame, textvariable=self.custom_dataset_var, width=20)
        custom_entry.grid(row=1, column=1, sticky='ew', padx=(0, 10), pady=(10, 0))
        custom_entry.bind('<Return>', self.on_custom_dataset_entered)

        # Load UCI button
        ttk.Button(uci_frame, text="Load Dataset",
                  command=self.load_uci_dataset).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(uci_frame, text="Fetch Custom Dataset",
                  command=self.fetch_custom_dataset).grid(row=1, column=2, padx=(0, 10), pady=(10, 0))

        # Dataset info display
        self.dataset_info = scrolledtext.ScrolledText(uci_frame, height=6, width=80)
        self.dataset_info.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(10, 0))
        self.dataset_info.insert('1.0', "Select a dataset to view information...")
        self.dataset_info.config(state='disabled')

        # Configure grid weights
        uci_frame.columnconfigure(1, weight=1)

        # Local File Loading
        local_frame = ttk.LabelFrame(main_frame, text="💾 Local Files", padding=10)
        local_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(local_frame, text="Load CSV File",
                  command=self.load_csv_file).pack(side='left', padx=(0, 10))
        ttk.Button(local_frame, text="Load Model",
                  command=self.load_model_file).pack(side='left', padx=(0, 10))
        ttk.Button(local_frame, text="Save Model",
                  command=self.save_model_file).pack(side='left')
        # In create_data_tab method, add this button:
        ttk.Button(local_frame, text="Set Feature Names",
                  command=self.set_feature_names).pack(side='left', padx=(10, 0))

        # Data Preview
        preview_frame = ttk.LabelFrame(main_frame, text="👀 Data Preview", padding=10)
        preview_frame.pack(fill='both', expand=True)

        # Create treeview for data display
        columns = ("Feature", "Type", "Missing", "Unique", "Range")
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=12)

        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120)

        # Scrollbars
        v_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.data_tree.yview)
        h_scroll = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.data_tree.pack(side='left', fill='both', expand=True)
        v_scroll.pack(side='right', fill='y')
        h_scroll.pack(side='bottom', fill='x')

    def create_feature_tab(self, parent):
        """Create feature selection tab"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Feature selection frame
        feature_frame = ttk.LabelFrame(main_frame, text="Feature Selection", padding=10)
        feature_frame.pack(fill='both', expand=True)

        # Instructions
        ttk.Label(feature_frame, text="Select features for training and specify target column:",
                 font=('Arial', 10, 'bold')).pack(anchor='w', pady=(0, 10))

        # Feature selection area
        selection_frame = ttk.Frame(feature_frame)
        selection_frame.pack(fill='both', expand=True)

        # Left side - Available features
        available_frame = ttk.LabelFrame(selection_frame, text="Available Features", padding=10)
        available_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        self.available_listbox = tk.Listbox(available_frame, selectmode=tk.MULTIPLE, height=15)
        available_scroll = ttk.Scrollbar(available_frame, orient='vertical', command=self.available_listbox.yview)
        self.available_listbox.configure(yscrollcommand=available_scroll.set)
        self.available_listbox.pack(side='left', fill='both', expand=True)
        available_scroll.pack(side='right', fill='y')

        # Right side - Selected features and target
        selected_frame = ttk.LabelFrame(selection_frame, text="Selection", padding=10)
        selected_frame.pack(side='right', fill='both', expand=True)

        # Selected features
        ttk.Label(selected_frame, text="Selected Features:").pack(anchor='w')
        self.selected_listbox = tk.Listbox(selected_frame, height=8)
        selected_scroll = ttk.Scrollbar(selected_frame, orient='vertical', command=self.selected_listbox.yview)
        self.selected_listbox.configure(yscrollcommand=selected_scroll.set)
        self.selected_listbox.pack(fill='both', expand=True, pady=(5, 10))
        selected_scroll.pack(side='right', fill='y')

        # Target selection
        target_frame = ttk.Frame(selected_frame)
        target_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(target_frame, text="Target Column:").pack(side='left', padx=(0, 10))
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, state='readonly')
        self.target_combo.pack(side='left', fill='x', expand=True)

        # Control buttons
        button_frame = ttk.Frame(feature_frame)
        button_frame.pack(fill='x', pady=10)

        ttk.Button(button_frame, text="Add Selected",
                  command=self.add_features).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Remove Selected",
                  command=self.remove_features).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Select All",
                  command=self.select_all_features).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Clear All",
                  command=self.clear_features).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Auto-Select from Model",
                  command=self.auto_select_features).pack(side='left')

        # Apply button
        ttk.Button(feature_frame, text="Apply Feature Selection",
                  command=self.apply_feature_selection, style='Accent.TButton').pack(pady=10)

    def create_parameters_tab(self, parent):
        """Create hyperparameter configuration tab"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Parameter frame
        param_frame = ttk.LabelFrame(main_frame, text="Model Hyperparameters", padding=10)
        param_frame.pack(fill='both', expand=True)

        # Parameter entries
        self.param_vars = {}
        parameters = [
            ('resol', 'Resolution', '8', 'Number of bins for feature discretization (4-16)'),
            ('use_complex_tensor', 'Use Complex Tensor', 'True', 'Enable complex tensor operations'),
            ('orthogonalize_weights', 'Orthogonal Weights', 'True', 'Use orthogonal weight initialization'),
            ('smoothing_factor', 'Smoothing Factor', '1e-8', 'Laplace smoothing for probabilities'),
            ('parallel_processing', 'Parallel Processing', 'True', 'Enable parallel computation'),
            ('n_jobs', 'Number of Jobs', '-1', 'Number of parallel workers (-1 for all)'),
            ('batch_size', 'Batch Size', '1000', 'Batch size for processing'),
            ('missing_value_placeholder', 'Missing Value', '-99999', 'Placeholder for missing values')
        ]

        for i, (key, label, default, help_text) in enumerate(parameters):
            ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky='w', padx=(0, 10), pady=5)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(param_frame, textvariable=var, width=15)
            entry.grid(row=i, column=1, sticky='w', padx=(0, 20), pady=5)
            ttk.Label(param_frame, text=help_text, foreground='gray').grid(row=i, column=2, sticky='w', pady=5)
            self.param_vars[key] = var

        # Button frame
        button_frame = ttk.Frame(param_frame)
        button_frame.grid(row=len(parameters), column=0, columnspan=3, pady=20)

        ttk.Button(button_frame, text="Apply Parameters",
                  command=self.apply_parameters).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Reset to Defaults",
                  command=self.reset_parameters).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Save Parameters",
                  command=self.save_parameters).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Load Parameters",
                  command=self.load_parameters).pack(side='left')
        ttk.Button(button_frame, text="Auto-load Parameters",
                  command=self.auto_load_parameters).pack(side='left', padx=(10, 0))

    def create_training_tab(self, parent):
        """Create training interface tab with configurable train/test split"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Training configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))

        # Train/test split configuration
        split_frame = ttk.Frame(config_frame)
        split_frame.pack(fill='x', pady=5)

        self.enable_split_var = tk.BooleanVar(value=False)
        self.split_checkbox = ttk.Checkbutton(split_frame, text="Enable Train/Test Split",
                                             variable=self.enable_split_var,
                                             command=self.toggle_split_config)
        self.split_checkbox.pack(side='left', padx=(0, 20))

        self.split_label = ttk.Label(split_frame, text="Training Percentage:")
        self.split_label.pack(side='left', padx=(0, 10))

        self.train_split_var = tk.DoubleVar(value=80.0)
        self.train_split_spin = ttk.Spinbox(split_frame, from_=10, to=100, increment=5,
                                           textvariable=self.train_split_var, width=5,
                                           state='disabled')
        self.train_split_spin.pack(side='left', padx=(0, 10))

        self.percent_label = ttk.Label(split_frame, text="%")
        self.percent_label.pack(side='left')

        # Training controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(control_frame, text="Initialize Model",
                  command=self.initialize_model).pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Compute Likelihoods",
                  command=self.compute_likelihoods).pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Train Model",
                  command=self.train_model).pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Evaluate Model",
                  command=self.evaluate_model).pack(side='left')

        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Training Progress", padding=10)
        progress_frame.pack(fill='x', pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill='x', pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Ready")
        self.progress_label.pack()

        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", padding=10)
        log_frame.pack(fill='both', expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.pack(fill='both', expand=True)
        self.log_text.config(state='disabled')

    def toggle_split_config(self):
        """Enable/disable train split configuration based on checkbox"""
        if self.enable_split_var.get():
            self.train_split_spin.config(state='normal')
            self.split_label.config(foreground='black')
            self.percent_label.config(foreground='black')
        else:
            self.train_split_spin.config(state='disabled')
            self.split_label.config(foreground='gray')
            self.percent_label.config(foreground='gray')


    def create_prediction_tab(self, parent):
        """Create prediction interface tab"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Prediction controls
        control_frame = ttk.LabelFrame(main_frame, text="Prediction Controls", padding=10)
        control_frame.pack(fill='x', pady=(0, 10))

        # File selection
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill='x', pady=5)

        ttk.Button(file_frame, text="Select Prediction File",
                  command=self.select_prediction_file).pack(side='left', padx=(0, 10))
        self.pred_file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.pred_file_var).pack(side='left')

        # Options frame
        options_frame = ttk.Frame(control_frame)
        options_frame.pack(fill='x', pady=5)

        ttk.Label(options_frame, text="Has Target Column:").pack(side='left', padx=(0, 10))
        self.has_target_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, variable=self.has_target_var,
                       command=self.toggle_target_selection).pack(side='left', padx=(0, 20))

        ttk.Label(options_frame, text="Target Column:").pack(side='left', padx=(0, 10))
        self.pred_target_var = tk.StringVar()
        self.pred_target_combo = ttk.Combobox(options_frame, textvariable=self.pred_target_var, state='disabled')
        self.pred_target_combo.pack(side='left', fill='x', expand=True, padx=(0, 20))

        # Prediction button
        ttk.Button(control_frame, text="Run Prediction",
                  command=self.run_prediction, style='Accent.TButton').pack(pady=10)

        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding=10)
        results_frame.pack(fill='both', expand=True)

        self.pred_results_text = scrolledtext.ScrolledText(results_frame, height=15)
        self.pred_results_text.pack(fill='both', expand=True)
        self.pred_results_text.config(state='disabled')

    def create_results_tab(self, parent):
        """Create results visualization tab"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Results controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=(0, 10))

        ttk.Button(control_frame, text="Show Feature Importance",
                  command=self.show_feature_importance).pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Show Confusion Matrix",
                  command=self.show_confusion_matrix).pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Export Results",
                  command=self.export_results).pack(side='left')

        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Model Results", padding=10)
        results_frame.pack(fill='both', expand=True)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=20)
        self.results_text.pack(fill='both', expand=True)
        self.results_text.config(state='disabled')

    # Feature selection methods
    def add_features(self):
        """Add selected features to training set"""
        selected = self.available_listbox.curselection()
        for idx in selected:
            feature = self.available_listbox.get(idx)
            if feature not in self.selected_listbox.get(0, tk.END):
                self.selected_listbox.insert(tk.END, feature)

    def remove_features(self):
        """Remove selected features from training set"""
        selected = self.selected_listbox.curselection()
        for idx in reversed(selected):
            self.selected_listbox.delete(idx)

    def select_all_features(self):
        """Select all available features"""
        self.selected_listbox.delete(0, tk.END)
        for i in range(self.available_listbox.size()):
            self.selected_listbox.insert(tk.END, self.available_listbox.get(i))

    def clear_features(self):
        """Clear all selected features"""
        self.selected_listbox.delete(0, tk.END)

    def auto_select_features(self):
        """Auto-select features from loaded model"""
        if self.model and hasattr(self.model, 'feature_names'):
            self.clear_features()
            for feature in self.model.feature_names:
                self.selected_listbox.insert(tk.END, feature)
            self.log_message("✅ Auto-selected features from loaded model")
        else:
            messagebox.showwarning("Warning", "No model loaded or model has no feature names")

    def apply_feature_selection(self):
        """Apply feature selection to current dataset"""
        selected_features = list(self.selected_listbox.get(0, tk.END))
        target_column = self.target_var.get()

        if not selected_features:
            messagebox.showwarning("Warning", "Please select at least one feature")
            return

        if not target_column:
            messagebox.showwarning("Warning", "Please select a target column")
            return

        # Store selection
        self.feature_selection = {
            'features': selected_features,
            'target': target_column
        }

        self.log_message(f"✅ Feature selection applied: {len(selected_features)} features, target: {target_column}")

        # Update prediction target combo
        self.pred_target_combo['values'] = selected_features
        if target_column in selected_features:
            self.pred_target_var.set(target_column)

    def toggle_target_selection(self):
        """Enable/disable target selection based on checkbox"""
        if self.has_target_var.get():
            self.pred_target_combo.config(state='readonly')
        else:
            self.pred_target_combo.config(state='disabled')
            self.pred_target_var.set('')

    def select_prediction_file(self):
        """Select file for prediction"""
        filepath = filedialog.askopenfilename(
            title="Select file for prediction",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.pred_file_var.set(filepath)
            # Load column names for target selection
            try:
                df = pd.read_csv(filepath, nrows=1)
                self.pred_target_combo['values'] = list(df.columns)
            except Exception as e:
                self.log_message(f"❌ Error reading file: {e}")

    def run_prediction(self):
        """Run prediction on selected file"""
        if not self.model or not self.model.is_trained:
            messagebox.showwarning("Warning", "Please train a model first")
            return

        filepath = self.pred_file_var.get()
        if not filepath or filepath == "No file selected":
            messagebox.showwarning("Warning", "Please select a prediction file")
            return

        try:
            self.log_message(f"🔮 Running prediction on: {os.path.basename(filepath)}")

            # Determine features to use
            features_to_use = None
            if self.model.selected_features:
                features_to_use = self.model.selected_features
            elif self.feature_selection:
                features_to_use = self.feature_selection['features']

            # Determine target column
            target_column = None
            if self.has_target_var.get():
                target_column = self.pred_target_var.get()

            # Run prediction
            results_df = self.model.predict_on_file(
                filepath=filepath,
                features_to_use=features_to_use,
                has_target=self.has_target_var.get(),
                target_column=target_column
            )

            # Display results summary
            self.pred_results_text.config(state='normal')
            self.pred_results_text.delete('1.0', tk.END)

            summary = f"""PREDICTION RESULTS
{'='*50}
File: {os.path.basename(filepath)}
Samples: {len(results_df)}
Features used: {len(features_to_use) if features_to_use else 'All available'}

Confidence Statistics:
- Minimum: {results_df['Confidence'].min():.3f}
- Maximum: {results_df['Confidence'].max():.3f}
- Average: {results_df['Confidence'].mean():.3f}

Top 5 Predictions (by confidence):
"""
            for i, (idx, row) in enumerate(results_df.head().iterrows()):
                summary += f"{i+1}. {row['Prediction']} (Confidence: {row['Confidence']:.3f})\n"

            self.pred_results_text.insert('1.0', summary)
            self.pred_results_text.config(state='disabled')

            self.log_message("✅ Prediction completed successfully")

        except Exception as e:
            self.log_message(f"❌ Prediction failed: {e}")
            messagebox.showerror("Error", f"Prediction failed: {e}")


    def load_csv_file(self):
        """Load custom CSV file with feature selection - FIXED to use actual column names"""
        try:
            filepath = filedialog.askopenfilename(
                title="Select CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not filepath:
                return

            self.update_status("Loading CSV file...")
            self.log_message(f"📥 Loading CSV file: {filepath}")

            # Reset everything for fresh start
            self.reset_everything()

            # Load CSV
            self.current_csv_path = filepath
            df = pd.read_csv(filepath)

            # CRITICAL FIX: Store the actual column names for model training
            self.actual_feature_names = [col for col in df.columns if col != 'target']  # Adjust target column name as needed
            self.actual_target_name = 'target'  # Adjust as needed

            # Update available features list
            self.available_listbox.delete(0, tk.END)
            for col in df.columns:
                self.available_listbox.insert(tk.END, col)

            # Update target combos
            self.target_combo['values'] = list(df.columns)
            self.pred_target_combo['values'] = list(df.columns)

            # Store as dataset-like object WITH PROPER FEATURE NAMES
            data = type('CSVData', (), {})()
            data.data = df[self.actual_feature_names].values
            data.target = df[self.actual_target_name].values if self.actual_target_name in df.columns else None
            data.feature_names = self.actual_feature_names  # CRITICAL: Store actual feature names
            data.columns = list(df.columns)
            data.name = os.path.basename(filepath).replace('.csv', '')

            self.current_dataset = data

            # Update data preview
            self.update_data_preview(df.values, None)

            self.log_message(f"✅ CSV loaded successfully")
            self.log_message(f"   Columns: {len(df.columns)}")
            self.log_message(f"   Samples: {len(df)}")
            self.log_message(f"   Feature names: {self.actual_feature_names}")
            self.update_status(f"Loaded CSV dataset")

            # Try to auto-load hyperparameters
            self.auto_load_parameters()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
            self.log_message(f"❌ Error loading CSV: {str(e)}")
    def set_feature_names(self):
        """Manually set feature names for the current dataset"""
        if not hasattr(self, 'current_dataset'):
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        current_names = getattr(self.current_dataset, 'feature_names', [])
        new_names = simpledialog.askstring(
            "Set Feature Names",
            f"Enter feature names (comma-separated):\nCurrent: {current_names}",
            initialvalue=", ".join(current_names) if current_names else ""
        )

        if new_names:
            feature_names = [name.strip() for name in new_names.split(',')]
            if hasattr(self.current_dataset, 'data'):
                n_features = self.current_dataset.data.shape[1]
                if len(feature_names) != n_features:
                    messagebox.showerror("Error", f"Number of feature names ({len(feature_names)}) doesn't match number of features ({n_features})")
                    return

            self.current_dataset.feature_names = feature_names
            self.log_message(f"✅ Feature names set to: {feature_names}")

    def initialize_model(self):
        """Initialize model with current parameters and feature selection - UPDATED with optional split"""
        if not hasattr(self, 'current_dataset'):
            messagebox.showwarning("Warning", "Please load a dataset first")
            return

        try:
            self.apply_parameters()

            # Apply feature selection if specified
            if self.feature_selection:
                selected_features = self.feature_selection['features']
                target_column = self.feature_selection['target']

                if hasattr(self.current_dataset, 'columns'):
                    df = pd.read_csv(self.current_csv_path)
                    features_df = df[selected_features]
                    targets = df[target_column]

                    if self.enable_split_var.get():
                        test_size = (100 - self.train_split_var.get()) / 100.0
                        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            features_df.values, targets.values,
                            test_size=test_size,
                            random_state=42,
                            stratify=targets
                        )
                        self.log_message(f"✅ Data split: {self.train_split_var.get()}% training, {100-self.train_split_var.get()}% testing")
                        self.log_message(f"   Training samples: {len(self.X_train)}")
                        self.log_message(f"   Testing samples: {len(self.X_test)}")
                    else:
                        self.X_train = features_df.values
                        self.y_train = targets.values
                        self.X_test = None
                        self.y_test = None
                        self.log_message("✅ Using 100% of data for training (no test split)")
                        self.log_message(f"   Training samples: {len(self.X_train)}")

                    self.model = ParallelCTDBNN(self.model_config)
                    self.model.compute_global_likelihoods(self.X_train, self.y_train, selected_features)
                    self.model.selected_features = selected_features
                    self.model.target_column_name = target_column

                else:
                    messagebox.showwarning("Warning", "Feature selection only available for CSV files")
                    return
            else:
                if hasattr(self.current_dataset, 'data'):
                    features = self.current_dataset.data
                    targets = getattr(self.current_dataset, 'target', None)

                    feature_names = getattr(self.current_dataset, 'feature_names', None)
                    if feature_names is None and hasattr(self.current_dataset, 'columns'):
                        feature_names = [col for col in self.current_dataset.columns if col != 'target']

                    if targets is not None:
                        if self.enable_split_var.get():
                            test_size = (100 - self.train_split_var.get()) / 100.0
                            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                features, targets,
                                test_size=test_size,
                                random_state=42,
                                stratify=targets
                            )
                            self.log_message(f"✅ Data split: {self.train_split_var.get()}% training, {100-self.train_split_var.get()}% testing")
                            self.log_message(f"   Training samples: {len(self.X_train)}")
                            self.log_message(f"   Testing samples: {len(self.X_test)}")
                        else:
                            self.X_train = features
                            self.y_train = targets
                            self.X_test = None
                            self.y_test = None
                            self.log_message("✅ Using 100% of data for training (no test split)")
                            self.log_message(f"   Training samples: {len(self.X_train)}")

                        self.model = ParallelCTDBNN(self.model_config)
                        self.model.compute_global_likelihoods(self.X_train, self.y_train, feature_names)

                    else:
                        self.log_message("⚠️  No target available for this dataset")
                        return

            self.log_message("✅ Model initialized successfully")
            self.log_message(f"   Configuration: {self.model_config}")
            if hasattr(self.model, 'feature_names'):
                self.log_message(f"   Actual feature names: {self.model.feature_names}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model: {str(e)}")
            self.log_message(f"❌ Error initializing model: {str(e)}")

    def run(self):
        """Run the GUI application"""
        if GUI_AVAILABLE:
            self.root.mainloop()
        else:
            print("GUI not available. Use command-line interface.")


if __name__ == "__main__":
    # Add ASCII art and welcome message
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

main()
