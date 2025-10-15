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
import sys
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

# Try to import GUI components, fallback gracefully
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

# Replace dbnn import with ct_dbnn
try:
    import ct_dbnn
except ImportError:
    print("❌ CT-DBNN module not found. Please ensure ct_dbnn.py is in the same directory.")
    exit(1)

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
            'reference': 'UCI Machine Learning Repository'
        },
        'iris': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            'description': 'Iris plants database',
            'target_column': 'class',
            'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'best_accuracy': 0.973,
            'best_method': 'Multiple methods',
            'reference': 'UCI Machine Learning Repository'
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
            'reference': 'UCI Machine Learning Repository'
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
            'reference': 'UCI Machine Learning Repository'
        },
        'wine-quality': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
            'description': 'Wine Quality',
            'target_column': 'quality',
            'delimiter': ';',
            'header': 0,
            'best_accuracy': 0.683,
            'best_method': 'SVM with RBF kernel',
            'reference': 'UCI Machine Learning Repository'
        },
        'car': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
            'description': 'Car Evaluation',
            'target_column': 'class',
            'feature_names': ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
            'best_accuracy': 0.968,
            'best_method': 'Decision Trees',
            'reference': 'UCI Machine Learning Repository'
        },
        'banknote': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt',
            'description': 'Banknote Authentication',
            'target_column': 'class',
            'feature_names': ['variance', 'skewness', 'curtosis', 'entropy'],
            'best_accuracy': 0.998,
            'best_method': 'Multiple methods',
            'reference': 'UCI Machine Learning Repository'
        },
        'seeds': {
            'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt',
            'description': 'Seeds Dataset',
            'target_column': 'type',
            'delimiter': '\t',
            'feature_names': ['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry', 'groove_length'],
            'best_accuracy': 0.914,
            'best_method': 'LDA',
            'reference': 'UCI Machine Learning Repository'
        }
    }

    @staticmethod
    def get_available_uci_datasets() -> List[str]:
        """Get list of available UCI datasets"""
        return list(UCIDatasetHandler.UCI_DATASETS.keys())

    @staticmethod
    def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
        """Get detailed information about a dataset"""
        dataset_name = dataset_name.lower()
        if dataset_name in UCIDatasetHandler.UCI_DATASETS:
            return UCIDatasetHandler.UCI_DATASETS[dataset_name]
        return {}

    @staticmethod
    def download_uci_dataset(dataset_name: str) -> Tuple[bool, str]:
        """Download dataset from UCI repository"""
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
        """Create configuration file for UCI dataset"""
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
            'adaptive_learning': {
                'enable_adaptive': True,
                'initial_samples_per_class': 5,
                'max_adaptive_rounds': 20,
                'patience': 10,
                'min_improvement': 0.001,
                'enable_acid_test': True,
            },
            'ctdbnn_config': {
                'resol': 100,
                'use_complex_tensor': True,
                'orthogonalize_weights': True,
                'parallel_processing': True,
                'smoothing_factor': 1e-8,
            }
        }

        config_file = os.path.join(output_dir, f"{dataset_name}.conf")

        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"✅ Created configuration file: {config_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to create configuration file: {e}")
            return False

    @staticmethod
    def setup_uci_dataset(dataset_name: str) -> bool:
        """Complete setup for UCI dataset: download, load, and create config"""
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
            feature_names.append(col)

        # Stack all features
        if processed_features:
            X_processed = np.column_stack(processed_features)
        else:
            X_processed = np.empty((len(X), 0))

        return X_processed, feature_names

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

    def preprocess_dataset(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess entire dataset"""
        print("🔧 Preprocessing dataset...")

        # Separate features and target
        if self.target_column not in data.columns:
            # Try to find the target column automatically
            possible_targets = ['class', 'target', 'label', 'outcome', 'diagnosis', 'type', 'quality']
            for possible_target in possible_targets:
                if possible_target in data.columns:
                    self.target_column = possible_target
                    print(f"🔍 Auto-detected target column: {self.target_column}")
                    break

            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in dataset. Available columns: {list(data.columns)}")

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

        return X_processed, y_processed, feature_names

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

class CTDBNNWrapper:
    """
    Wrapper for ct_dbnn.py module that implements the exact adaptive learning requirements
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or {}

        # Initialize the core CT-DBNN
        ct_dbnn_config = {
            'resol': self.config.get('resol', 100),
            'use_complex_tensor': self.config.get('use_complex_tensor', True),
            'orthogonalize_weights': self.config.get('orthogonalize_weights', True),
            'parallel_processing': self.config.get('parallel_processing', True),
            'smoothing_factor': self.config.get('smoothing_factor', 1e-8),
            'n_jobs': self.config.get('n_jobs', -1),
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

        # Feature information
        self.feature_names = []
        self.initialized_with_full_data = False

    def load_data(self, file_path: str = None):
        """Load data from file with robust preprocessing - with UCI auto-download"""
        if file_path is None:
            # Try to find dataset file - prioritize original data files
            possible_files = [
                f"{self.dataset_name}.csv",
                f"{self.dataset_name}.data",
                f"{self.dataset_name}.txt",
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
            if self.dataset_name.lower() in available_uci:
                print(f"🎯 Detected UCI dataset: {self.dataset_name}")
                # Setup UCI dataset (download + create config)
                success = UCIDatasetHandler.setup_uci_dataset(self.dataset_name)
                if success:
                    file_path = f"{self.dataset_name}.csv"
                else:
                    raise ValueError(f"Failed to setup UCI dataset: {self.dataset_name}")
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
                    print(f"❌ No data file found for dataset: {self.dataset_name}")
                    print(f"📋 Available UCI datasets: {', '.join(available_uci)}")
                    print("💡 You can use one of these UCI datasets or provide your own data file")
                    raise ValueError("No data file found. Please provide a CSV or DAT file.")

        # Load the data file
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

        return self.data

    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        return self.preprocessor.preprocess_dataset(self.data)

    def initialize_with_full_data(self, X: np.ndarray, y: np.ndarray):
        """Step 1: Initialize CT-DBNN architecture with full dataset"""
        print("🏗️ Initializing CT-DBNN architecture with full dataset...")

        try:
            # For CT-DBNN, architecture initialization means computing global likelihoods
            feature_cols = [f'feature_{i}' for i in range(X.shape[1])]

            # Compute global likelihoods (CT-DBNN's architecture setup)
            self.core.compute_global_likelihoods(X, y, feature_cols)

            # Initialize orthogonal weights
            self.core.initialize_orthogonal_weights()

            print("✅ CT-DBNN architecture initialized with full dataset")
            self.initialized_with_full_data = True

            # Freeze the architecture
            self.freeze_architecture()

        except Exception as e:
            print(f"❌ CT-DBNN initialization error: {e}")
            import traceback
            traceback.print_exc()

    def train_with_data(self, X_train: np.ndarray, y_train: np.ndarray, reset_weights: bool = True):
        """Step 2: Train with given data using CT-DBNN"""
        if not self.initialized_with_full_data:
            # Try to initialize if not already done
            print("⚠️  CT-DBNN not initialized, attempting initialization...")
            self.initialize_with_full_data(X_train, y_train)
            if not self.initialized_with_full_data:
                raise ValueError("CT-DBNN must be initialized with full data first")

        print(f"🎯 Training CT-DBNN with {len(X_train)} samples...")

        try:
            # For CT-DBNN, training is the orthogonal weight initialization
            # after global likelihoods are computed

            # Recompute likelihoods with current training data if needed
            if reset_weights or not self.core.likelihoods_computed:
                feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
                self.core.compute_global_likelihoods(X_train, y_train, feature_cols)

            # CT-DBNN uses one-step orthogonal weight initialization
            self.core.initialize_orthogonal_weights()

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

    def freeze_architecture(self):
        """Freeze architectural components"""
        self.architecture_frozen = True
        self.frozen_components = {
            'config': self.core.config.copy(),
            'feature_names': self.feature_names.copy() if hasattr(self, 'feature_names') else [],
            'target_column': self.target_column,
            'innodes': getattr(self.core, 'innodes', 0),
            'outnodes': getattr(self.core, 'outnodes', 0),
        }
        print("✅ CT-DBNN architecture frozen")

    def _reset_weights(self):
        """Reset weights while preserving architecture - for CT-DBNN this means reinitializing orthogonal weights"""
        if hasattr(self.core, 'initialize_orthogonal_weights'):
            self.core.initialize_orthogonal_weights()
            print("✅ CT-DBNN weights reset with orthogonal initialization")
        else:
            print("⚠️  Cannot reset weights - CT-DBNN not properly initialized")

    def reset_weights(self):
        """Reset weights - for CT-DBNN, we reinitialize orthogonal weights"""
        print("🔄 Resetting CT-DBNN weights with orthogonal initialization...")
        self._reset_weights()

class AdaptiveCTDBNN:
    """Wrapper for CT-DBNN that implements sophisticated adaptive learning with comprehensive analysis"""

    def __init__(self, dataset_name: str = None, config: Dict = None):
        # Handle dataset selection if not provided
        if dataset_name is None:
            dataset_name = self._select_dataset()

        self.dataset_name = dataset_name
        self.config = config or self._load_config(dataset_name)

        # Ensure config has required fields by using DatasetConfig
        if 'target_column' not in self.config:
            dataset_config = DatasetConfig.load_config(dataset_name)
            if dataset_config:
                self.config.update(dataset_config)

        # Add CT-DBNN specific configuration
        self.config.update({
            'use_complex_tensor': self.config.get('use_complex_tensor', True),
            'orthogonalize_weights': self.config.get('orthogonalize_weights', True),
            'parallel_processing': self.config.get('parallel_processing', True),
            'smoothing_factor': self.config.get('smoothing_factor', 1e-8),
            'n_jobs': self.config.get('n_jobs', -1),
        })

        # Enhanced adaptive learning configuration with proper defaults
        self.adaptive_config = self.config.get('adaptive_learning', {})
        # Set defaults for any missing parameters
        default_config = {
            "enable_adaptive": True,
            "initial_samples_per_class": 5,
            "max_margin_samples_per_class": 3,
            "margin_tolerance": 0.15,
            "kl_threshold": 0.1,
            "max_adaptive_rounds": 20,
            "patience": 10,
            "min_improvement": 0.001,
            "training_convergence_epochs": 50,
            "min_training_accuracy": 0.95,
            "min_samples_to_add_per_class": 5,
            "adaptive_margin_relaxation": 0.1,
            "max_divergence_samples_per_class": 5,
            "exhaust_all_failed": True,
            "min_failed_threshold": 10,
            "enable_kl_divergence": True,
            "max_samples_per_class_fallback": 2,
            "enable_3d_visualization": True,
            "3d_snapshot_interval": 10,
            "learning_rate": 1.0,
            "enable_acid_test": True,
            "min_training_percentage_for_stopping": 10.0,
            "max_training_percentage": 90.0,
            "margin_tolerance": 0.15,
            "kl_divergence_threshold": 0.1,
            "max_kl_samples_per_class": 5,
            "disable_sample_limit": False,
        }
        for key, default_value in default_config.items():
            if key not in self.adaptive_config:
                self.adaptive_config[key] = default_value

        self.stats_config = self.config.get('statistics', {
            'enable_confusion_matrix': True,
            'enable_progress_plots': True,
            'color_progress': 'green',
            'color_regression': 'red',
            'save_plots': True,
            'create_interactive_plots': True,
            'create_sample_analysis': True
        })

        # Visualization configuration
        self.viz_config = self.config.get('visualization_config', {
            'enabled': True,
            'output_dir': 'adaptive_visualizations',
            'create_animations': False,
            'create_reports': True,
            'create_3d_visualizations': True
        })

        # Initialize the base CT-DBNN model using our wrapper
        self.model = CTDBNNWrapper(dataset_name, config=self.config)

        # Adaptive learning state
        self.training_indices = []
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.best_round = 0
        self.adaptive_round = 0
        self.patience_counter = 0

        # Statistics tracking
        self.round_stats = []
        self.previous_confusion = None
        self.start_time = datetime.now()
        self.adaptive_start_time = None
        self.device_type = self._get_device_type()

        # Store the full dataset for adaptive learning
        self.X_full = None
        self.y_full = None
        self.y_full_original = None
        self.original_data_shape = None

        # Track all selected samples for analysis
        self.all_selected_samples = defaultdict(list)
        self.sample_selection_history = []

        # Initialize label encoder for adaptive learning
        self.label_encoder = LabelEncoder()

        # Initialize visualizers
        self.adaptive_visualizer = None
        self._initialize_visualizers()

        # Update config file with default settings if they don't exist
        self._update_config_file()

        # Show current settings
        self.show_adaptive_settings()

        # Add 3D visualization initialization
        self._initialize_3d_visualization()

    def _load_config(self, dataset_name: str) -> Dict:
        """Load configuration from file"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _select_dataset(self) -> str:
        """Select dataset from available configuration files, data files, or UCI datasets"""
        available_configs = DatasetConfig.get_available_config_files()
        available_uci = UCIDatasetHandler.get_available_uci_datasets()

        # Also look for data files
        csv_files = glob.glob("*.csv")
        dat_files = glob.glob("*.dat")
        txt_files = glob.glob("*.txt")
        data_files = csv_files + dat_files + txt_files

        if available_configs or data_files or available_uci:
            print("📁 Available datasets and configuration files:")

            # Show configuration-based datasets
            if available_configs:
                print("\n🎯 Configuration files:")
                for i, config in enumerate(available_configs, 1):
                    base_name = config['file'].replace('.json', '').replace('.conf', '')
                    print(f"  {i}. {base_name} ({config['type']} configuration)")

            # Show data files
            if data_files:
                print("\n📊 Data files:")
                start_idx = len(available_configs) + 1
                for i, data_file in enumerate(data_files, start_idx):
                    print(f"  {i}. {data_file}")

            # Show UCI datasets
            if available_uci:
                print("\n🌐 UCI Repository datasets:")
                start_idx = len(available_configs) + len(data_files) + 1
                for i, uci_dataset in enumerate(available_uci, start_idx):
                    print(f"  {i}. {uci_dataset} (UCI - auto-download)")

            print(f"\n  {len(available_configs) + len(data_files) + len(available_uci) + 1}. Enter custom dataset name")

            try:
                total_options = len(available_configs) + len(data_files) + len(available_uci) + 1
                choice = input(f"\nSelect a dataset (1-{total_options}): ").strip()
                choice_idx = int(choice) - 1

                if 0 <= choice_idx < len(available_configs):
                    selected_config = available_configs[choice_idx]
                    selected_dataset = selected_config['file'].replace('.json', '').replace('.conf', '')
                    print(f"🎯 Selected configuration: {selected_dataset} ({selected_config['type']})")
                    return selected_dataset
                elif len(available_configs) <= choice_idx < len(available_configs) + len(data_files):
                    data_file_idx = choice_idx - len(available_configs)
                    selected_file = data_files[data_file_idx]
                    dataset_name = selected_file.replace('.csv', '').replace('.dat', '').replace('.txt', '')
                    print(f"📁 Selected data file: {selected_file}")
                    return dataset_name
                elif len(available_configs) + len(data_files) <= choice_idx < len(available_configs) + len(data_files) + len(available_uci):
                    uci_idx = choice_idx - len(available_configs) - len(data_files)
                    selected_uci = available_uci[uci_idx]
                    print(f"🌐 Selected UCI dataset: {selected_uci} (will auto-download)")
                    # Setup UCI dataset
                    UCIDatasetHandler.setup_uci_dataset(selected_uci)
                    return selected_uci
                elif choice_idx == len(available_configs) + len(data_files) + len(available_uci):
                    # Custom dataset name
                    custom_name = input("Enter custom dataset name: ").strip()
                    if not custom_name:
                        return self._select_dataset()  # Recursive call if empty

                    # Check if custom name exists as file
                    possible_files = [f"{custom_name}.csv", f"{custom_name}.data", f"{custom_name}.txt"]
                    for file in possible_files:
                        if os.path.exists(file):
                            print(f"📁 Found existing file: {file}")
                            return custom_name

                    # Check if custom name is a known UCI dataset
                    if custom_name.lower() in available_uci:
                        print(f"🌐 Custom name matches UCI dataset: {custom_name}")
                        UCIDatasetHandler.setup_uci_dataset(custom_name)
                        return custom_name

                    print(f"📝 Using custom dataset name: {custom_name}")
                    return custom_name
                else:
                    print("❌ Invalid selection")
                    return input("Enter dataset name: ").strip()
            except ValueError:
                print("❌ Invalid input")
                return input("Enter dataset name: ").strip()
        else:
            print("❌ No configuration files, data files, or known UCI datasets found.")
            print("   Looking for: *.json, *.conf, *.csv, *.dat, *.txt")
            print(f"   Available UCI datasets: {', '.join(available_uci)}")
            dataset_name = input("Enter dataset name: ").strip()
            if not dataset_name:
                dataset_name = "default_dataset"
            return dataset_name

    def _get_device_type(self) -> str:
        """Get the device type (CPU/GPU)"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
                return f"GPU: {gpu_name}"
            else:
                return "CPU"
        except:
            return "Unknown Device"

    def _initialize_visualizers(self):
        """Initialize visualization systems"""
        # Initialize adaptive visualizer
        if self.viz_config.get('enabled', True):
            try:
                self.adaptive_visualizer = CTDBNNVisualizer(
                    self.model,
                    output_dir=self.viz_config.get('output_dir', 'adaptive_visualizations'),
                    enabled=True
                )
                print("✓ Adaptive visualizer initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize adaptive visualizer: {e}")
                self.adaptive_visualizer = None

        # Create output directory
        os.makedirs(self.viz_config.get('output_dir', 'adaptive_visualizations'), exist_ok=True)

    def _update_config_file(self):
        """Update the dataset configuration file with adaptive learning settings"""
        config_path = f"{self.dataset_name}.conf"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            if 'adaptive_learning' not in config:
                config['adaptive_learning'] = {}

            config['adaptive_learning'].update(self.adaptive_config)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"✅ Updated configuration file: {config_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not update config file: {str(e)}")

    def show_adaptive_settings(self):
        """Display the current adaptive learning settings"""
        print("\n🔧 Advanced Adaptive Learning Settings:")
        print("=" * 60)
        for key, value in self.adaptive_config.items():
            if key in ['margin_tolerance', 'kl_divergence_threshold', 'max_kl_samples_per_class']:
                print(f"  {key:40}: {value} (KL Divergence)")
            elif key == 'disable_sample_limit':
                status = "DISABLED 🚫" if value else "ENABLED ✅"
                print(f"  {key:40}: {value} ({status})")
            else:
                print(f"  {key:40}: {value}")

        # Show CT-DBNN specific settings
        print(f"\n🎯 CT-DBNN Specific Settings:")
        print(f"  use_complex_tensor: {self.config.get('use_complex_tensor', True)}")
        print(f"  orthogonalize_weights: {self.config.get('orthogonalize_weights', True)}")
        print(f"  parallel_processing: {self.config.get('parallel_processing', True)}")

        # Show best known results if available
        dataset_info = UCIDatasetHandler.get_dataset_info(self.dataset_name)
        if dataset_info and 'best_accuracy' in dataset_info:
            print(f"\n🏆 Best Known Results for {self.dataset_name}:")
            print(f"  Best Accuracy: {dataset_info['best_accuracy']}")
            print(f"  Best Method: {dataset_info.get('best_method', 'Unknown')}")
            print(f"  Reference: {dataset_info.get('reference', 'Unknown')}")

        print(f"\n💻 Device: {self.device_type}")
        mode = "KL Divergence" if self.adaptive_config.get('enable_kl_divergence', False) else "Margin-Based"
        limit_status = "UNLIMITED" if self.adaptive_config.get('disable_sample_limit', False) else "LIMITED"
        print(f"🎯 Selection Mode: {mode} ({limit_status})")
        print()

    def _initialize_3d_visualization(self):
        """Initialize 3D visualization system"""
        self.visualization_output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        os.makedirs(f'{self.visualization_output_dir}/3d_animations', exist_ok=True)
        self.feature_grid_history = []
        self.epoch_timestamps = []

        print("🎨 3D Visualization system initialized")

    def prepare_full_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare the full dataset for adaptive learning"""
        print("📊 Preparing full dataset...")

        # Load data using the model's method
        self.model.load_data()

        # Preprocess data using the enhanced preprocessor
        X, y, feature_names = self.model.preprocess_data()

        # Store original y for reference (before encoding)
        y_original = y.copy()

        # Store the full dataset
        self.X_full = X
        self.y_full = y
        self.y_full_original = y_original
        self.original_data_shape = X.shape

        print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Classes: {len(np.unique(y))} ({np.unique(y_original)})")
        print(f"🔧 Features: {feature_names}")

        return X, y, y_original

    def adaptive_learn(self, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning method with acid test-based stopping criteria"""
        print("\n🚀 STARTING ADAPTIVE LEARNING WITH CT-DBNN")
        print("=" * 60)

        # Use provided data or prepare full data
        if X is None or y is None:
            print("📊 Preparing dataset...")
            X, y, y_original = self.prepare_full_data()
        else:
            y_original = y.copy()
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)

        # Store the full dataset
        self.X_full = X.copy()
        self.y_full = y.copy()
        self.y_full_original = y_original

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes: {np.unique(y_original)}")

        # STEP 1: Initialize CT-DBNN architecture with full dataset
        self.model.initialize_with_full_data(X, y)

        # STEP 2: Select initial diverse training samples
        X_train, y_train, initial_indices = self._select_initial_training_samples(X, y)
        remaining_indices = [i for i in range(len(X)) if i not in initial_indices]

        print(f"📊 Initial training set: {len(X_train)} samples")
        print(f"📊 Remaining test set: {len(remaining_indices)} samples")

        # Initialize tracking variables for acid test-based stopping
        self.best_accuracy = 0.0
        self.best_training_indices = initial_indices.copy()
        self.best_round = 0
        acid_test_history = []
        patience_counter = 0

        max_rounds = self.adaptive_config['max_adaptive_rounds']
        patience = self.adaptive_config['patience']
        min_improvement = self.adaptive_config['min_improvement']

        print(f"\n🔄 Starting adaptive learning for up to {max_rounds} rounds...")
        print(f"📊 Stopping criteria: 100% acid test OR no improvement for {patience} rounds")
        self.adaptive_start_time = datetime.now()

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num

            print(f"\n🎯 Round {round_num}/{max_rounds}")
            print("-" * 40)

            # STEP 2 (continued): Train with current training data (no split)
            print("🎯 Training with current training data...")
            success = self.model.train_with_data(X_train, y_train, reset_weights=True)

            if not success:
                print("❌ Training failed, stopping...")
                break

            # STEP 3: Run acid test on entire dataset - THIS IS OUR MAIN STOPPING CRITERION
            print("🧪 Running acid test on entire dataset...")
            try:
                all_predictions = self.model.predict(X)
                # Ensure predictions and y have same data type
                all_predictions = all_predictions.astype(y.dtype)
                acid_test_accuracy = accuracy_score(y, all_predictions)
                acid_test_history.append(acid_test_accuracy)
                print(f"📊 Acid test accuracy: {acid_test_accuracy:.4f}")

                # PRIMARY STOPPING CRITERION 1: 100% accuracy on entire dataset
                if acid_test_accuracy >= 0.9999:  # 99.99% accuracy (accounting for floating point)
                    print("🎉 REACHED 100% ACCURACY ON ENTIRE DATASET! Stopping adaptive learning.")
                    self.best_accuracy = acid_test_accuracy
                    self.best_training_indices = initial_indices.copy()
                    self.best_round = round_num
                    break

            except Exception as e:
                print(f"❌ Acid test failed: {e}")
                acid_test_accuracy = 0.0
                acid_test_history.append(0.0)
                continue

            # STEP 3 (continued): Check if we have any remaining samples to process
            if not remaining_indices:
                print("💤 No more samples to add to training set")
                # Check if we should stop based on acid test performance
                if len(acid_test_history) >= 3:
                    recent_improvement = acid_test_history[-1] - acid_test_history[-3]
                    if recent_improvement < min_improvement:
                        print("📊 Acid test performance flattened - stopping adaptive learning.")
                        break
                continue

            # STEP 4: Identify failed candidates in remaining data
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
                # PRIMARY STOPPING CRITERION 2: No errors in remaining data
                print("🎉 PERFECT CLASSIFICATION ON REMAINING DATA! Stopping adaptive learning.")
                self.best_accuracy = acid_test_accuracy
                self.best_training_indices = initial_indices.copy()
                self.best_round = round_num
                break

            print(f"📊 Found {len(misclassified_indices)} misclassified samples in remaining data")

            # STEP 5: Select most divergent failed candidates
            samples_to_add_indices = self._select_divergent_samples(
                X_remaining, y_remaining, remaining_predictions, remaining_posteriors,
                misclassified_indices, remaining_indices
            )

            if not samples_to_add_indices:
                print("💤 No divergent samples to add")
                # Check if we should stop based on acid test performance
                if len(acid_test_history) >= 3:
                    recent_improvement = acid_test_history[-1] - acid_test_history[-3]
                    if recent_improvement < min_improvement:
                        print("📊 Acid test performance flattened - stopping adaptive learning.")
                        break
                continue

            # Update training set
            initial_indices.extend(samples_to_add_indices)
            remaining_indices = [i for i in remaining_indices if i not in samples_to_add_indices]

            X_train = X[initial_indices]
            y_train = y[initial_indices]

            print(f"📈 Training set size: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}% of total)")
            print(f"📊 Remaining set size: {len(remaining_indices)} samples")

            # STEP 6: Update best model and check for improvement
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

            # SECONDARY STOPPING CRITERION: Check for performance plateau
            if len(acid_test_history) >= 5:
                recent_trend = acid_test_history[-5:]
                max_recent = max(recent_trend)
                min_recent = min(recent_trend)
                fluctuation = max_recent - min_recent

                if fluctuation < min_improvement * 2:  # Very small fluctuations
                    print(f"📊 Acid test performance plateaued (fluctuation: {fluctuation:.4f}) - stopping adaptive learning.")
                    break

        # Finalize with best configuration
        print(f"\n🎉 Adaptive learning completed after {self.adaptive_round} rounds!")

        # Ensure we have valid best values
        if not hasattr(self, 'best_accuracy') or self.best_accuracy == 0.0:
            # Use final values if best wasn't set
            self.best_accuracy = acid_test_history[-1] if acid_test_history else 0.0
            self.best_training_indices = initial_indices.copy()
            self.best_round = self.adaptive_round

        print(f"🏆 Best acid test accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
        print(f"📊 Final training set: {len(self.best_training_indices)} samples ({len(self.best_training_indices)/len(X)*100:.1f}% of total)")

        # Use best configuration for final model
        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        X_test_best = X[[i for i in range(len(X)) if i not in self.best_training_indices]]
        y_test_best = y[[i for i in range(len(X)) if i not in self.best_training_indices]]

        # Train final model with best configuration
        print("🔧 Training final model with best configuration...")
        self.model.train_with_data(X_train_best, y_train_best, reset_weights=True)

        # Final acid test verification
        final_predictions = self.model.predict(X)
        final_accuracy = accuracy_score(y, final_predictions)

        print(f"📊 Final acid test accuracy: {final_accuracy:.4f}")
        print(f"📈 Final training set size: {len(X_train_best)}")
        print(f"📊 Final remaining set size: {len(X_test_best)}")

        # Generate reports
        self._generate_adaptive_learning_report()

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
        max_samples = self.adaptive_config.get('max_margin_samples_per_class', 2)

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

                # Track selection
                self.all_selected_samples[self._get_original_class_label(class_id)].append({
                    'index': sample['index'],
                    'margin': sample['margin'],
                    'selection_type': 'divergent',
                    'round': self.adaptive_round
                })

            if selected_for_class:
                print(f"   ✅ Class {class_id}: Selected {len(selected_for_class)} divergent samples")

        print(f"📥 Total divergent samples to add: {len(samples_to_add)}")
        return samples_to_add

    def _get_original_class_label(self, encoded_class: int) -> str:
        """Convert encoded class back to original label"""
        if hasattr(self.label_encoder, 'classes_'):
            try:
                return str(self.label_encoder.inverse_transform([encoded_class])[0])
            except:
                return str(encoded_class)
        return str(encoded_class)

    def _generate_adaptive_learning_report(self):
        """Generate comprehensive adaptive learning report"""
        print("\n📊 Generating Adaptive Learning Report...")

        # Ensure we have valid statistics
        if not hasattr(self, 'best_accuracy'):
            self.best_accuracy = 0.0
        if not hasattr(self, 'best_training_indices'):
            self.best_training_indices = []
        if not hasattr(self, 'best_round'):
            self.best_round = 0

        total_time = str(datetime.now() - self.adaptive_start_time) if hasattr(self, 'adaptive_start_time') and self.adaptive_start_time else "N/A"

        # Get best known results for comparison
        dataset_info = UCIDatasetHandler.get_dataset_info(self.dataset_name)
        best_known_accuracy = dataset_info.get('best_accuracy', 'Unknown') if dataset_info else 'Unknown'

        report = {
            'dataset': self.dataset_name,
            'total_samples': len(self.X_full) if hasattr(self, 'X_full') else 0,
            'final_training_size': len(self.best_training_indices),
            'final_remaining_size': (len(self.X_full) - len(self.best_training_indices)) if hasattr(self, 'X_full') else 0,
            'best_accuracy': float(self.best_accuracy),
            'best_known_accuracy': best_known_accuracy,
            'best_round': self.best_round,
            'total_rounds': getattr(self, 'adaptive_round', 0),
            'total_time': total_time,
            'adaptive_config': self.adaptive_config,
            'ctdbnn_config': {
                'use_complex_tensor': self.config.get('use_complex_tensor', True),
                'orthogonalize_weights': self.config.get('orthogonalize_weights', True),
                'parallel_processing': self.config.get('parallel_processing', True),
            },
            'round_statistics': getattr(self, 'round_stats', []),
            'selected_samples_by_class': {k: len(v) for k, v in self.all_selected_samples.items()}
        }

        # Save report
        report_path = f"{self.viz_config.get('output_dir', 'adaptive_visualizations')}/adaptive_learning_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)

        print(f"✅ Report saved to: {report_path}")

        # Print summary with proper formatting
        print("\n📈 Adaptive Learning Summary:")
        print("=" * 50)
        print(f"   Dataset: {report['dataset']}")
        print(f"   Total samples: {report['total_samples']}")

        if report['total_samples'] > 0:
            training_percentage = (report['final_training_size'] / report['total_samples']) * 100
            print(f"   Final training set: {report['final_training_size']} ({training_percentage:.1f}%)")
        else:
            print(f"   Final training set: {report['final_training_size']}")

        print(f"   Best acid test accuracy: {report['best_accuracy']:.4f}")
        if best_known_accuracy != 'Unknown':
            print(f"   Best known accuracy: {best_known_accuracy}")
            if isinstance(best_known_accuracy, (int, float)) and report['best_accuracy'] >= best_known_accuracy:
                print("   🎉 EXCELLENT! Matched or exceeded best known accuracy!")
            elif isinstance(best_known_accuracy, (int, float)) and report['best_accuracy'] >= best_known_accuracy * 0.95:
                print("   👍 GOOD! Within 5% of best known accuracy!")
        print(f"   Achieved in round: {report['best_round']}")
        print(f"   Total rounds: {report['total_rounds']}")
        print(f"   Total time: {report['total_time']}")
        print("=" * 50)

class AdaptiveCTDBNNGUI:
    """GUI interface for Adaptive CT-DBNN"""

    def __init__(self, root):
        self.root = root
        self.root.title("Adaptive CT-DBNN with UCI Dataset Support")
        self.root.geometry("1200x800")

        self.adaptive_model = None
        self.model_trained = False

        self.setup_gui()

    def setup_gui(self):
        """Setup the main GUI"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Adaptive CT-DBNN",
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)

        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="10")
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

        # Dataset info frame
        info_frame = ttk.LabelFrame(main_frame, text="Dataset Information", padding="10")
        info_frame.pack(fill=tk.X, pady=5)

        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, width=100)
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Run Adaptive Learning",
                  command=self.run_adaptive_learning, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Make Predictions",
                  command=self.make_predictions, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Show Results",
                  command=self.show_results, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Exit",
                  command=self.root.quit, width=10).pack(side=tk.RIGHT, padx=5)

        # Output frame
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=100)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def log_output(self, message):
        """Add message to output text"""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()

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
        info_text += f"🔗 URL: {dataset_info.get('url', 'N/A')}\n\n"
        info_text += f"📋 Features: {', '.join(dataset_info.get('feature_names', []))}"

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.log_output(f"✅ Loaded information for dataset: {dataset_name}")

    def run_adaptive_learning(self):
        """Run adaptive learning"""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("Warning", "Please select a dataset first.")
            return

        try:
            self.log_output(f"🚀 Starting adaptive learning for {dataset_name}...")

            # Create and run adaptive model
            self.adaptive_model = AdaptiveCTDBNN(dataset_name)
            X_train, y_train, X_test, y_test = self.adaptive_model.adaptive_learn()

            self.model_trained = True
            self.log_output("✅ Adaptive learning completed successfully!")
            self.log_output(f"📊 Final training set: {len(X_train)} samples")
            self.log_output(f"📈 Final test set: {len(X_test)} samples")
            self.log_output(f"🏆 Best accuracy: {self.adaptive_model.best_accuracy:.4f}")

        except Exception as e:
            self.log_output(f"❌ Error during adaptive learning: {e}")

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
            else:
                self.log_output("❌ No data available for predictions.")

        except Exception as e:
            self.log_output(f"❌ Error during predictions: {e}")

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

        except Exception as e:
            self.log_output(f"❌ Error showing results: {e}")

def main():
    """Main function to run adaptive CT-DBNN"""
    import sys

    # Check for GUI flag
    if "--gui" in sys.argv or "-g" in sys.argv:
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
    for i, arg in enumerate(sys.argv):
        if arg in ["--config", "-c"] and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
            break

    # Load configuration if provided
    config = {}
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration from: {config_file}")

            # Print configuration summary
            if 'feature_columns' in config:
                print(f"📊 Using {len(config['feature_columns'])} features: {config['feature_columns']}")
            if 'target_column' in config:
                print(f"🎯 Target column: {config['target_column']}")
            if 'dataset_name' in config:
                print(f"📁 Dataset: {config['dataset_name']}")

        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            config = {}
    else:
        # Show available UCI datasets
        available_uci = UCIDatasetHandler.get_available_uci_datasets()
        print(f"📋 Available UCI datasets: {', '.join(available_uci)}")
        print("\n💡 Use: python adaptive_ctdbnn.py --config <filename> to use a specific configuration")
        print("💡 Or use: python adaptive_ctdbnn.py --gui for graphical interface")
        print("💡 Or use UCI dataset names directly")

    # Create adaptive CT-DBNN
    adaptive_model = AdaptiveCTDBNN(config.get('dataset_name'), config)

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
