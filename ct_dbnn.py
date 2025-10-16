#!/usr/bin/env python3
"""
OPTIMIZED CT-DBNN MODULE - Fixed Label Encoding Consistency
WITH PROPER LABEL HANDLING ACROSS TRAINING AND PREDICTION
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

# Parallel processing imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading

# Optional imports for GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib
    matplotlib.use('TkAgg')
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class DataPreprocessor:
    """
    Comprehensive data preprocessing with CONSISTENT label encoding
    """

    def __init__(self, missing_value_placeholder=-99999):
        self.missing_value_placeholder = missing_value_placeholder
        self.feature_encoders = {}  # encoders for each feature column
        self.target_encoder = None  # encoder for target column
        self.target_decoder = None  # decoder for target column
        self.feature_columns = None
        self.target_column = None
        self.original_dtypes = {}
        self.is_fitted = False

    def fit_transform_features(self, features):
        """Fit and transform features with label encoding and missing value handling"""
        if isinstance(features, pd.DataFrame):
            result = self._fit_transform_dataframe(features)
        elif isinstance(features, np.ndarray):
            result = self._fit_transform_array(features)
        else:
            # Convert to numpy array and process
            features_array = np.array(features)
            result = self._fit_transform_array(features_array)

        self.is_fitted = True
        return result

    def _fit_transform_dataframe(self, df):
        """Fit and transform DataFrame features"""
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
        """Fit and transform numpy array features"""
        processed_data = array.copy().astype(object)
        n_samples, n_features = array.shape
        self.feature_columns = [f'Feature_{i+1}' for i in range(n_features)]
        self.feature_encoders = {}

        for col_idx in range(n_features):
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
                    self.feature_encoders[col_idx] = encoder
                    for i, val in enumerate(processed_data[:, col_idx]):
                        processed_data[i, col_idx] = encoder.get(val, 0)
                else:
                    # All numeric
                    processed_data[:, col_idx] = numeric_data
                    self.feature_encoders[col_idx] = 'numeric'
            except:
                # Fallback: treat as categorical
                unique_vals = np.unique(processed_data[:, col_idx])
                encoder = {val: idx for idx, val in enumerate(unique_vals)}
                self.feature_encoders[col_idx] = encoder
                for i, val in enumerate(processed_data[:, col_idx]):
                    processed_data[i, col_idx] = encoder.get(val, 0)

        return processed_data.astype(np.float64)

    def transform_features(self, features):
        """Transform features using fitted encoders"""
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
        """Transform DataFrame features using fitted encoders"""
        processed_data = df.copy()

        for col in df.columns:
            if col not in self.feature_encoders:
                raise ValueError(f"Column {col} not seen during fitting")

            # Handle missing values
            if df[col].isna().any():
                processed_data[col] = df[col].fillna(self.missing_value_placeholder)

            # Apply encoding if needed
            if self.feature_encoders[col] != 'numeric':
                encoder = self.feature_encoders[col]
                processed_data[col] = processed_data[col].map(
                    lambda x: encoder.get(x, 0) if x in encoder else 0
                )

        return processed_data.values.astype(np.float64)

    def _transform_array(self, array):
        """Transform numpy array features using fitted encoders"""
        processed_data = array.copy().astype(object)
        n_samples, n_features = array.shape

        for col_idx in range(n_features):
            if col_idx not in self.feature_encoders:
                raise ValueError(f"Column index {col_idx} not seen during fitting")

            col_data = array[:, col_idx]

            # Handle missing values
            missing_mask = pd.isna(col_data) if hasattr(pd, 'isna') else (
                (col_data == None) | ((isinstance(col_data, str)) & ((col_data == 'NaN') | (col_data == 'NA') | (col_data == '')))
            )
            if np.any(missing_mask):
                processed_data[missing_mask, col_idx] = self.missing_value_placeholder

            # Apply encoding if needed
            if self.feature_encoders[col_idx] != 'numeric':
                encoder = self.feature_encoders[col_idx]
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
        if hasattr(targets, 'target'):
            return np.array(targets.target)
        elif isinstance(targets, np.ndarray):
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


class ParallelCTDBNN:
    """
    Optimized CT-DBNN with CONSISTENT label handling across all operations
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

    def compute_global_likelihoods(self, features, targets, feature_names=None):
        """
        Compute global likelihoods ONCE on entire dataset with comprehensive preprocessing
        This is called ONLY during training
        """
        if self.likelihoods_computed:
            print("⚠️  Likelihoods already computed! Using existing global likelihoods.")
            return self.training_features_norm

        print("Computing GLOBAL likelihoods on entire dataset...")
        print("🔧 Applying comprehensive data preprocessing...")

        # Preprocess features and targets - STORE THE PREPROCESSOR STATE
        features_processed = self.preprocessor.fit_transform_features(features)
        self.training_targets_encoded = self.preprocessor.fit_transform_targets(targets)

        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(features, 'columns'):
            self.feature_names = list(features.columns)
        else:
            self.feature_names = [f'Feature_{i+1}' for i in range(features_processed.shape[1])]

        n_samples, n_features = features_processed.shape
        self.innodes = n_features
        resol = self.config['resol']

        print(f"Data shape: {features_processed.shape}, Resolution: {resol}")
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

            if abs(actual - predicted) <= self.dmyclass[0]:
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

    def evaluate(self, features, targets):
        """Evaluate model accuracy using CONSISTENT label handling"""
        # Get predictions (already in original labels)
        predictions = self.predict(features)

        # Convert targets to original labels for comparison
        targets_original = self.preprocessor.inverse_transform_targets(
            self.preprocessor.transform_targets(targets)
        )

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
            for i, v in enumerate(feature_importance):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error plotting feature importance: {e}")

    def save_model(self, filepath: str):
        """Save model to file including ALL precomputed parameters"""
        model_data = {
            'config': self.config,
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
            'training_history': self.training_history,
            'feature_names': self.feature_names,
            'preprocessor': self.preprocessor,
            'training_features_norm': self.training_features_norm,
            'training_targets_encoded': self.training_targets_encoded,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file including ALL precomputed parameters"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        for key, value in model_data.items():
            setattr(self, key, value)

        print(f"✅ Model loaded from {filepath}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'config': self.config,
            'architecture': {
                'input_nodes': self.innodes,
                'output_nodes': self.outnodes,
                'resolution': self.config['resol'],
                'classes': list(self.encoded_to_class.values()) if self.encoded_to_class else []
            },
            'training': self.training_history,
            'preprocessing': self.preprocessor.get_feature_info() if hasattr(self, 'preprocessor') else {},
            'status': {
                'is_trained': self.is_trained,
                'likelihoods_computed': self.likelihoods_computed
            }
        }


# Alias for backward compatibility
ConsistentCTDBNN = ParallelCTDBNN


class CTDBNNCommandLine:
    """Command Line Interface for CT-DBNN"""

    def __init__(self):
        self.model = None

    def run(self):
        """Main command line interface"""
        parser = argparse.ArgumentParser(description='CT-DBNN with One-Step Orthogonalization')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Train command
        train_parser = subparsers.add_parser('train', help='Train a new model')
        train_parser.add_argument('--data', required=True, help='Training data file (CSV)')
        train_parser.add_argument('--target', required=True, help='Target column name')
        train_parser.add_argument('--test-size', type=float, default=0.3, help='Test set size ratio')
        train_parser.add_argument('--resolution', type=int, default=8, help='Resolution parameter')
        train_parser.add_argument('--save-model', help='Save model to file')

        # Predict command
        predict_parser = subparsers.add_parser('predict', help='Make predictions')
        predict_parser.add_argument('--model', required=True, help='Model file')
        predict_parser.add_argument('--data', required=True, help='Data file for prediction')
        predict_parser.add_argument('--output', help='Output file for predictions')

        # Evaluate command
        eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
        eval_parser.add_argument('--model', required=True, help='Model file')
        eval_parser.add_argument('--data', required=True, help='Evaluation data file')
        eval_parser.add_argument('--target', required=True, help='Target column name')

        args = parser.parse_args()

        if args.command == 'train':
            self.train_model(args)
        elif args.command == 'predict':
            self.predict_model(args)
        elif args.command == 'evaluate':
            self.evaluate_model(args)
        else:
            parser.print_help()

    def train_model(self, args):
        """Train model from command line"""
        print("🚀 Training CT-DBNN model...")

        # Load data
        data = pd.read_csv(args.data)
        X = data.drop(columns=[args.target])
        y = data[args.target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42
        )

        # Create and train model
        self.model = ConsistentCTDBNN({
            'resol': args.resolution,
            'use_complex_tensor': True,
            'orthogonalize_weights': True
        })

        # Compute global likelihoods and train
        self.model.compute_global_likelihoods(X_train, y_train, feature_names=list(X_train.columns))
        self.model.train(X_train, y_train)

        # Evaluate
        train_acc = self.model.evaluate(X_train, y_train)
        test_acc = self.model.evaluate(X_test, y_test)

        print(f"\n📊 Final Results:")
        print(f"   Training Accuracy: {train_acc:.4f}")
        print(f"   Test Accuracy:     {test_acc:.4f}")
        print(f"   Generalization Gap: {train_acc - test_acc:.4f}")

        # Save model if requested
        if args.save_model:
            self.model.save_model(args.save_model)

    def predict_model(self, args):
        """Make predictions from command line"""
        print("🔮 Making predictions...")

        # Load model
        self.model = ConsistentCTDBNN()
        self.model.load_model(args.model)

        # Load data
        data = pd.read_csv(args.data)
        predictions = self.model.predict(data)
        probabilities = self.model.predict_proba(data)

        # Add predictions to data with confidence
        data['prediction'] = predictions
        data['prediction_confidence'] = np.max(probabilities, axis=1)

        # Add individual class probabilities
        for i, class_name in enumerate(self.model.encoded_to_class.values()):
            data[f'prob_{class_name}'] = probabilities[:, i]

        # Save or display results
        if args.output:
            data.to_csv(args.output, index=False)
            print(f"✅ Predictions saved to {args.output}")
        else:
            print("\n📋 Predictions (first 10):")
            print(data[['prediction', 'prediction_confidence']].head(10))

    def evaluate_model(self, args):
        """Evaluate model from command line"""
        print("📊 Evaluating model...")

        # Load model
        self.model = ConsistentCTDBNN()
        self.model.load_model(args.model)

        # Load data
        data = pd.read_csv(args.data)
        X = data.drop(columns=[args.target])
        y = data[args.target]

        # Evaluate
        accuracy = self.model.evaluate(X, y)

        print(f"✅ Model Accuracy: {accuracy:.4f}")

        # Detailed report
        predictions = self.model.predict(X)
        print("\n📈 Classification Report:")
        print(classification_report(y, predictions))


if GUI_AVAILABLE:
    class CTDBNNGUI:
        """Graphical User Interface for CT-DBNN"""

        def __init__(self, root):
            self.root = root
            self.root.title("CT-DBNN with One-Step Orthogonalization")
            self.root.geometry("1000x700")

            self.model = None
            self.data = None
            self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
            self.selected_features = []
            self.target_column = ""

            self.setup_ui()

        def setup_ui(self):
            """Setup the user interface"""
            # Main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Title
            title_label = ttk.Label(main_frame, text="CT-DBNN Classifier",
                                   font=('Arial', 16, 'bold'))
            title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

            # Left panel - Controls
            control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
            control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

            # Data loading
            ttk.Button(control_frame, text="Load Data (CSV)",
                      command=self.load_data).grid(row=0, column=0, pady=5, sticky=tk.W)

            # Data info display
            self.data_info_var = tk.StringVar(value="No data loaded")
            ttk.Label(control_frame, textvariable=self.data_info_var).grid(row=1, column=0, sticky=tk.W)

            # Feature selection frame
            self.feature_frame = ttk.LabelFrame(control_frame, text="Feature Selection", padding="5")
            self.feature_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))

            # Target selection
            ttk.Label(self.feature_frame, text="Target Column:").grid(row=0, column=0, sticky=tk.W)
            self.target_var = tk.StringVar()
            self.target_combo = ttk.Combobox(self.feature_frame, textvariable=self.target_var, state="readonly", width=20)
            self.target_combo.grid(row=0, column=1, padx=5, sticky=tk.W)

            # Feature list
            ttk.Label(self.feature_frame, text="Feature Columns:").grid(row=1, column=0, sticky=tk.W)

            # Frame for feature checkboxes
            self.feature_checkbox_frame = ttk.Frame(self.feature_frame)
            self.feature_checkbox_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))

            # Create canvas and scrollbar for feature list
            self.feature_canvas = tk.Canvas(self.feature_checkbox_frame, height=120, width=250)
            scrollbar = ttk.Scrollbar(self.feature_checkbox_frame, orient="vertical", command=self.feature_canvas.yview)
            self.feature_scroll_frame = ttk.Frame(self.feature_canvas)

            self.feature_scroll_frame.bind(
                "<Configure>",
                lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))
            )

            self.feature_canvas.create_window((0, 0), window=self.feature_scroll_frame, anchor="nw")
            self.feature_canvas.configure(yscrollcommand=scrollbar.set)

            self.feature_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Store feature checkboxes
            self.feature_vars = {}

            # Model configuration
            config_frame = ttk.Frame(control_frame)
            config_frame.grid(row=3, column=0, pady=10, sticky=tk.W)

            ttk.Label(config_frame, text="Resolution:").grid(row=0, column=0, sticky=tk.W)
            self.resolution_var = tk.StringVar(value="8")
            ttk.Entry(config_frame, textvariable=self.resolution_var, width=10).grid(row=0, column=1, padx=5)

            # Training buttons
            ttk.Button(control_frame, text="Compute Likelihoods",
                      command=self.compute_likelihoods).grid(row=4, column=0, pady=5, sticky=tk.W)

            ttk.Button(control_frame, text="Train Model",
                      command=self.train_model).grid(row=5, column=0, pady=5, sticky=tk.W)

            ttk.Button(control_frame, text="Evaluate Model",
                      command=self.evaluate_model).grid(row=6, column=0, pady=5, sticky=tk.W)

            # Prediction section
            ttk.Button(control_frame, text="Make Predictions",
                      command=self.make_predictions).grid(row=7, column=0, pady=5, sticky=tk.W)

            # Model saving/loading
            ttk.Button(control_frame, text="Save Model",
                      command=self.save_model).grid(row=8, column=0, pady=5, sticky=tk.W)

            ttk.Button(control_frame, text="Load Model",
                      command=self.load_model).grid(row=9, column=0, pady=5, sticky=tk.W)

            # Right panel - Output
            output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
            output_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

            self.output_text = scrolledtext.ScrolledText(output_frame, width=80, height=30)
            self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Visualization frame
            viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
            viz_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))

            ttk.Button(viz_frame, text="Show Model Info",
                      command=self.show_model_info).grid(row=0, column=0, padx=5)

            ttk.Button(viz_frame, text="Plot Feature Importance",
                      command=self.plot_feature_importance).grid(row=0, column=1, padx=5)

            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            main_frame.rowconfigure(1, weight=1)
            output_frame.columnconfigure(0, weight=1)
            output_frame.rowconfigure(0, weight=1)
            control_frame.columnconfigure(0, weight=1)

        def log_message(self, message):
            """Add message to output text"""
            self.output_text.insert(tk.END, f"{message}\n")
            self.output_text.see(tk.END)
            self.root.update()

        def load_data(self):
            """Load data from CSV file and setup feature selection"""
            filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if filepath:
                try:
                    self.data = pd.read_csv(filepath)
                    self.log_message(f"✅ Loaded data: {self.data.shape[0]} samples, {self.data.shape[1]} features")
                    self.data_info_var.set(f"Data: {self.data.shape[0]} samples, {self.data.shape[1]} features")

                    # Clear previous feature checkboxes
                    for widget in self.feature_scroll_frame.winfo_children():
                        widget.destroy()
                    self.feature_vars = {}

                    # Populate target combo
                    columns = list(self.data.columns)
                    self.target_combo['values'] = columns
                    if columns:
                        self.target_var.set(columns[-1])  # Default to last column

                    # Create feature checkboxes
                    row_idx = 0
                    for col in columns:
                        var = tk.BooleanVar(value=True)  # All features selected by default
                        self.feature_vars[col] = var
                        cb = ttk.Checkbutton(self.feature_scroll_frame, text=col, variable=var)
                        cb.grid(row=row_idx, column=0, sticky=tk.W, padx=5)
                        row_idx += 1

                    self.log_message(f"   Columns: {columns}")
                    self.log_message("   Please select target column and features to use")

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load data: {e}")

        def get_selected_features(self):
            """Get list of selected feature columns"""
            selected_features = []
            for col, var in self.feature_vars.items():
                if var.get() and col != self.target_var.get():
                    selected_features.append(col)
            return selected_features

        def compute_likelihoods(self):
            """Compute global likelihoods with selected features"""
            if self.data is None:
                messagebox.showerror("Error", "Please load data first")
                return

            target_col = self.target_var.get()
            if not target_col:
                messagebox.showerror("Error", "Please select target column")
                return

            selected_features = self.get_selected_features()
            if not selected_features:
                messagebox.showerror("Error", "Please select at least one feature")
                return

            try:
                X = self.data[selected_features]
                y = self.data[target_col]

                self.log_message(f"🔧 Using {len(selected_features)} features: {selected_features}")
                self.log_message(f"🎯 Target: {target_col}")

                self.model = ConsistentCTDBNN({
                    'resol': int(self.resolution_var.get()),
                    'use_complex_tensor': True,
                    'orthogonalize_weights': True
                })

                self.model.compute_global_likelihoods(X, y, feature_names=selected_features)
                self.log_message("✅ Global likelihoods computed successfully")

                # Store the selected features and target for later use
                self.selected_features = selected_features
                self.target_column = target_col

            except Exception as e:
                messagebox.showerror("Error", f"Failed to compute likelihoods: {e}")

        def train_model(self):
            """Train the model with selected features"""
            if self.model is None or not self.model.likelihoods_computed:
                messagebox.showerror("Error", "Please compute likelihoods first")
                return

            if self.data is None:
                messagebox.showerror("Error", "Please load data first")
                return

            try:
                X = self.data[self.selected_features]
                y = self.data[self.target_column]

                # Split data
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )

                training_time = self.model.train(self.X_train, self.y_train)
                self.log_message(f"✅ Training completed in {training_time:.3f}s")

            except Exception as e:
                messagebox.showerror("Error", f"Training failed: {e}")

        def evaluate_model(self):
            """Evaluate the model"""
            if self.model is None or not self.model.is_trained:
                messagebox.showerror("Error", "Please train model first")
                return

            if self.X_test is None or self.y_test is None:
                messagebox.showerror("Error", "No test data available")
                return

            try:
                accuracy = self.model.evaluate(self.X_test, self.y_test)
                self.log_message(f"📊 Test Accuracy: {accuracy:.4f}")

                # Show detailed results
                predictions = self.model.predict(self.X_test)
                report = classification_report(self.y_test, predictions)
                self.log_message(f"\n📈 Classification Report:\n{report}")

            except Exception as e:
                messagebox.showerror("Error", f"Evaluation failed: {e}")

        def make_predictions(self):
            """Make predictions on new data with confidence filtering"""
            if self.model is None or not self.model.is_trained:
                messagebox.showerror("Error", "Please train model first")
                return

            # Ask for prediction data file
            filepath = filedialog.askopenfilename(
                title="Select data for prediction",
                filetypes=[("CSV files", "*.csv")]
            )

            if not filepath:
                return

            try:
                # Load prediction data
                pred_data = pd.read_csv(filepath)

                # Check if required features are present
                missing_features = set(self.selected_features) - set(pred_data.columns)
                if missing_features:
                    messagebox.showerror("Error", f"Missing features in prediction data: {missing_features}")
                    return

                # Extract features for prediction
                X_pred = pred_data[self.selected_features]

                # Get predictions and probabilities
                predictions = self.model.predict(X_pred)
                probabilities = self.model.predict_proba(X_pred)

                # Calculate confidence (max probability for each sample)
                confidence_scores = np.max(probabilities, axis=1)

                # Create results dataframe with all original data
                results = pred_data.copy()
                results['predicted_class'] = predictions
                results['prediction_confidence'] = confidence_scores

                # Add individual class probabilities
                for i, class_name in enumerate(self.model.encoded_to_class.values()):
                    results[f'prob_{class_name}'] = probabilities[:, i]

                # Sort by confidence (descending)
                results_sorted = results.sort_values('prediction_confidence', ascending=False)

                # Apply confidence filter: keep only predictions with confidence >= 1/3 of max confidence
                max_confidence = np.max(confidence_scores)
                confidence_threshold = max_confidence / 3.0

                filtered_results = results_sorted[results_sorted['prediction_confidence'] >= confidence_threshold]
                discarded_count = len(results_sorted) - len(filtered_results)

                self.log_message(f"📊 Prediction Results:")
                self.log_message(f"   Total samples: {len(results_sorted)}")
                self.log_message(f"   High-confidence predictions: {len(filtered_results)}")
                self.log_message(f"   Low-confidence discarded: {discarded_count}")
                self.log_message(f"   Confidence threshold: {confidence_threshold:.4f}")

                # Save results
                output_file = filedialog.asksaveasfilename(
                    title="Save predictions",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")]
                )

                if output_file:
                    # Save both full results and filtered results
                    results_sorted.to_csv(output_file, index=False)

                    # Save filtered results with "_high_confidence" suffix
                    base_name = output_file.replace('.csv', '')
                    filtered_file = f"{base_name}_high_confidence.csv"
                    filtered_results.to_csv(filtered_file, index=False)

                    self.log_message(f"✅ Full predictions saved to: {output_file}")
                    self.log_message(f"✅ High-confidence predictions saved to: {filtered_file}")

                    # Show preview
                    self.log_message(f"\n📋 Preview of high-confidence predictions (top 5):")
                    preview = filtered_results.head(5)[['predicted_class', 'prediction_confidence'] + self.selected_features[:3]]
                    self.log_message(str(preview))

            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")

        def save_model(self):
            """Save model to file"""
            if self.model is None:
                messagebox.showerror("Error", "No model to save")
                return

            filepath = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                   filetypes=[("Pickle files", "*.pkl")])
            if filepath:
                try:
                    self.model.save_model(filepath)
                    self.log_message(f"✅ Model saved to {filepath}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save model: {e}")

        def load_model(self):
            """Load model from file"""
            filepath = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
            if filepath:
                try:
                    self.model = ConsistentCTDBNN()
                    self.model.load_model(filepath)
                    self.log_message(f"✅ Model loaded from {filepath}")

                    # Update feature selection if available
                    if hasattr(self.model, 'feature_names') and self.model.feature_names:
                        self.selected_features = self.model.feature_names
                        self.log_message(f"   Features: {self.selected_features}")

                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load model: {e}")

        def show_model_info(self):
            """Display model information"""
            if self.model is None:
                messagebox.showerror("Error", "No model available")
                return

            info = self.model.get_model_info()
            self.log_message("\n📋 Model Information:")
            for key, value in info.items():
                if key == 'config':
                    self.log_message(f"   {key}:")
                    for k, v in value.items():
                        self.log_message(f"     {k}: {v}")
                elif key == 'training_history':
                    self.log_message(f"   {key}:")
                    for k, v in value.items():
                        self.log_message(f"     {k}: {v}")
                else:
                    self.log_message(f"   {key}: {value}")

        def plot_feature_importance(self):
            """Plot feature importance"""
            if self.model is None or not self.model.is_trained:
                messagebox.showerror("Error", "Please train model first")
                return

            try:
                self.model.plot_feature_importance()
                self.log_message("✅ Feature importance plot displayed")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to plot feature importance: {e}")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Command line mode
        cli = CTDBNNCommandLine()
        cli.run()
    else:
        # GUI mode
        if not GUI_AVAILABLE:
            print("GUI dependencies not available. Please install tkinter and matplotlib.")
            print("Using command line interface instead.")
            cli = CTDBNNCommandLine()
            cli.run()
        else:
            root = tk.Tk()
            app = CTDBNNGUI(root)
            root.mainloop()


def run_demo():
    """Run demo with COMPLETE label consistency"""
    print("🚀 Running CT-DBNN Demo with COMPLETE Label Consistency...")

    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"Original target classes: {np.unique(y)}")

    # Add some categorical data and missing values for testing
    np.random.seed(42)

    # Add categorical feature
    categorical_feature = np.random.choice(['A', 'B', 'C'], size=len(X))
    X_with_categorical = np.column_stack([X, categorical_feature])
    feature_names = list(feature_names) + ['Categorical_Feature']

    # Add some missing values
    mask = np.random.random(X_with_categorical.shape) < 0.05
    X_with_categorical[mask] = np.nan

    # Convert to DataFrame for better demonstration
    df = pd.DataFrame(X_with_categorical, columns=feature_names)
    print("Dataset shape:", df.shape)
    print("Feature names:", feature_names)
    print("Missing values:", df.isna().sum().sum())

    # Split data - IMPORTANT: Use proper stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Training targets: {np.unique(y_train)}, Test targets: {np.unique(y_test)}")

    # Initialize and train model
    model = ParallelCTDBNN({
        'resol': 8,
        'use_complex_tensor': True,
        'orthogonalize_weights': True,
        'parallel_processing': True,
        'missing_value_placeholder': -99999
    })

    # Compute global likelihoods (with preprocessing) - ONLY ONCE
    print("\n🔧 Computing global likelihoods with preprocessing...")
    features_norm = model.compute_global_likelihoods(X_train, y_train, feature_names)

    # Train model - uses precomputed likelihoods
    print("\n🎯 Training model...")
    training_time = model.train(X_train, y_train)

    # Evaluate - uses CONSISTENT label handling
    print("\n📊 Evaluating model...")
    test_accuracy = model.evaluate(X_test, y_test)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f}")

    # Show model info
    print("\n📋 Model Information:")
    model_info = model.get_model_info()
    print(f"Input Features: {model_info['architecture']['input_nodes']}")
    print(f"Output Classes: {model_info['architecture']['output_nodes']}")
    print(f"Classes: {model_info['architecture']['classes']}")
    print(f"Training Time: {model_info['training']['training_time']:.3f}s")
    print(f"Train Accuracy: {model_info['training']['train_accuracy']:.4f}")
    print(f"Test Accuracy: {model_info['training']['test_accuracy']:.4f}")

    # Show encoding details
    print(f"\n🔧 Encoding Details:")
    print(f"   Class to Encoded: {model.class_to_encoded}")
    print(f"   Encoded to Class: {model.encoded_to_class}")

    # Make predictions with confidence
    print("\n🔮 Making predictions with confidence...")
    probabilities = model.predict_proba(X_test[:5])
    predictions = model.predict(X_test[:5])

    # Get actual test targets in original format for comparison
    actual_test_targets = y_test[:5]

    for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, actual_test_targets)):
        confidence = np.max(prob)
        print(f"Sample {i+1}: Predicted={pred}, Actual={actual}, Match={pred==actual}, Confidence={confidence:.4f}")

    return model


if __name__ == "__main__":
    print("🚀 CT-DBNN with One-Step Orthogonalization")
    print(f"🔧 CPU cores: {mp.cpu_count()}")

    # Quick test
    iris = load_iris()
    X, y = iris.data[:100], iris.target[:100]

    print("\n🧪 Quick performance test...")

    start_time = time.time()
    model = ConsistentCTDBNN()
    model.compute_global_likelihoods(X, y)
    model.train(X, y)
    test_time = time.time() - start_time

    print(f"✅ Test completed in {test_time:.3f}s")

    # Start appropriate interface
    main()
