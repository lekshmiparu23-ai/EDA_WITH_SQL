"""
preprocessing.py — Data transformation functions
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def handle_missing_values(df, columns, method, custom_value=None):
    """
    Handle missing values in specified columns.
    
    Parameters:
        df: DataFrame
        columns: list of column names
        method: one of 'drop', 'mean', 'median', 'mode', 'custom'
        custom_value: used when method='custom'
    
    Returns:
        (DataFrame, str) — modified df and summary message
    """
    mod_df = df.copy()
    initial_nulls = mod_df[columns].isnull().sum().sum()
    
    try:
        if method == 'drop':
            mod_df = mod_df.dropna(subset=columns)
        elif method == 'mean':
            for col in columns:
                if pd.api.types.is_numeric_dtype(mod_df[col]):
                    mod_df[col] = mod_df[col].fillna(mod_df[col].mean())
        elif method == 'median':
            for col in columns:
                if pd.api.types.is_numeric_dtype(mod_df[col]):
                    mod_df[col] = mod_df[col].fillna(mod_df[col].median())
        elif method == 'mode':
            for col in columns:
                mode_val = mod_df[col].mode()
                if not mode_val.empty:
                    mod_df[col] = mod_df[col].fillna(mode_val[0])
        elif method == 'custom':
            for col in columns:
                mod_df[col] = mod_df[col].fillna(custom_value)
                
        final_nulls = mod_df[columns].isnull().sum().sum()
        replaced = initial_nulls - final_nulls
        summary = f"Imputed {replaced} missing values using '{method}' for columns: {', '.join(columns)}."
        return mod_df, summary
        
    except Exception as e:
        return df, f"Error handling missing values: {e}"

def remove_duplicates(df):
    """Remove duplicate rows. Returns (df, summary)."""
    try:
        initial_len = len(df)
        mod_df = df.drop_duplicates()
        removed = initial_len - len(mod_df)
        summary = f"Removed {removed} duplicate rows."
        return mod_df, summary
    except Exception as e:
        return df, f"Error removing duplicates: {e}"

def encode_categorical(df, columns, method):
    """
    Encode categorical columns.
    
    Parameters:
        method: 'label' or 'onehot'
    
    Returns:
        (DataFrame, str)
    """
    mod_df = df.copy()
    try:
        if method == 'label':
            le = LabelEncoder()
            for col in columns:
                mod_df[col] = le.fit_transform(mod_df[col].astype(str))
            summary = f"Applied Label Encoding to columns: {', '.join(columns)}."
        elif method == 'onehot':
            mod_df = pd.get_dummies(mod_df, columns=columns)
            summary = f"Applied One-Hot Encoding resulting in {len(mod_df.columns)} total columns."
        else:
            summary = "Unknown encoding method."
            
        return mod_df, summary
    except Exception as e:
        return df, f"Error encoding categoricals: {e}"

def scale_numerical(df, columns, method):
    """
    Scale numerical columns.
    
    Parameters:
        method: 'minmax' or 'standard'
    
    Returns:
        (DataFrame, str)
    """
    mod_df = df.copy()
    try:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            return df, "Unknown matching method."

        mod_df[columns] = scaler.fit_transform(mod_df[columns])
        summary = f"Applied {method} scaling to columns: {', '.join(columns)}."
        return mod_df, summary
    except Exception as e:
        return df, f"Error scaling data: {e}"

def detect_column_types(df):
    """
    Auto-detect column types.
    
    Returns dict:
    {
      'numerical': [...],
      'categorical': [...],
      'datetime': [...],
      'text': [...]
    }
    """
    types = {'numerical': [], 'categorical': [], 'datetime': [], 'text': []}
    
    for col in df.columns:
        # Try datetime first
        if df[col].dtype == 'object':
            try:
                # To prevent arbitrary strings being parsed as dates, we use simple heuristic check 
                # before applying to_datetime
                if df[col].dropna().astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
                     pd.to_datetime(df[col], errors='raise')
                     types['datetime'].append(col)
                     continue
            except:
                pass
                
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types['datetime'].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            types['numerical'].append(col)
        else:
            nunique = df[col].nunique()
            if nunique < 20:
                types['categorical'].append(col)
            else:
                types['text'].append(col)
                
    return types

def get_dataset_health_score(df):
    """
    Calculate dataset health score 0-100.
    """
    try:
        missing_pct = df.isnull().sum().sum() / df.size * 100 if df.size > 0 else 0
        dup_pct = df.duplicated().sum() / len(df) * 100 if len(df) > 0 else 0
        
        # Outlier detection using IQR on all numerical columns
        outlier_count = 0
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            # Avoid divide by zero or extreme counts if IQR is 0
            if IQR > 0:
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                outlier_count += len(outliers)
                
        outlier_pct = outlier_count / max(1, len(df)) * 100
        
        score = 100 - (missing_pct * 0.4) - (dup_pct * 0.3) - (max(0, min(100, outlier_pct)) * 0.3)
        return max(0, min(100, round(score, 1)))
    except Exception:
        return 0
