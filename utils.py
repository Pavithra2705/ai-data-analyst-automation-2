import pandas as pd
import numpy as np
import streamlit as st
import io
import base64
from datetime import datetime
import tempfile
import os

class FileHandler:
    """
    Utility class for handling file operations
    """
    
    @staticmethod
    def read_uploaded_file(uploaded_file):
        """
        Read uploaded CSV or Excel file
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for CSV
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
            
            return df, None
            
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def save_dataframe_to_temp(df, filename_prefix="data", format="csv"):
        """
        Save DataFrame to temporary file and return the path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "csv":
            filename = f"{filename_prefix}_{timestamp}.csv"
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            df.to_csv(temp_path, index=False)
        elif format.lower() in ["xlsx", "excel"]:
            filename = f"{filename_prefix}_{timestamp}.xlsx"
            temp_path = os.path.join(tempfile.gettempdir(), filename)
            df.to_excel(temp_path, index=False)
        else:
            raise ValueError("Unsupported format")
        
        return temp_path
    
    @staticmethod
    def get_file_size_mb(file):
        """
        Get file size in MB
        """
        return len(file.getvalue()) / (1024 * 1024)

def download_button_with_data(data, filename, label, mime_type="text/csv"):
    """
    Create a download button for data
    """
    return st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime_type
    )

def create_download_link(df, filename, file_format="csv"):
    """
    Create a download link for DataFrame
    """
    if file_format.lower() == "csv":
        data = df.to_csv(index=False)
        mime_type = "text/csv"
    elif file_format.lower() == "xlsx":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data', index=False)
        data = buffer.getvalue()
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        raise ValueError("Unsupported format")
    
    b64 = base64.b64encode(data.encode() if isinstance(data, str) else data).decode()
    
    if file_format.lower() == "csv":
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {file_format.upper()}</a>'
    else:
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download {file_format.upper()}</a>'
    
    return href

def format_number(number):
    """
    Format number for display
    """
    if pd.isna(number):
        return "N/A"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.2f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.2f}K"
    elif abs(number) < 1 and number != 0:
        return f"{number:.4f}"
    else:
        return f"{number:.2f}"

def safe_divide(numerator, denominator):
    """
    Safe division to avoid division by zero
    """
    if denominator == 0:
        return np.inf
    return numerator / denominator

def detect_date_columns(df):
    """
    Detect potential date columns in DataFrame
    """
    date_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            date_columns.append(col)
        elif df[col].dtype == 'object':
            # Try to parse as date
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, infer_datetime_format=True)
                    # If successful for sample, likely a date column
                    date_columns.append(col)
                except:
                    pass
    
    return date_columns

def get_memory_usage(df):
    """
    Get detailed memory usage of DataFrame
    """
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        'total_mb': total_memory / 1024**2,
        'total_kb': total_memory / 1024,
        'by_column': memory_usage.to_dict()
    }

def validate_dataframe(df):
    """
    Validate DataFrame and return issues
    """
    issues = []
    
    # Check if DataFrame is empty
    if df.empty:
        issues.append("DataFrame is empty")
    
    # Check for columns with all NaN values
    all_nan_cols = df.columns[df.isnull().all()].tolist()
    if all_nan_cols:
        issues.append(f"Columns with all NaN values: {all_nan_cols}")
    
    # Check for duplicate column names
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        issues.append(f"Duplicate column names: {duplicate_cols}")
    
    # Check for extremely high cardinality in categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    high_cardinality_cols = []
    
    for col in cat_cols:
        if df[col].nunique() / len(df) > 0.9:
            high_cardinality_cols.append(col)
    
    if high_cardinality_cols:
        issues.append(f"High cardinality categorical columns (>90% unique): {high_cardinality_cols}")
    
    return issues

def suggest_data_types(df):
    """
    Suggest optimal data types for DataFrame columns
    """
    suggestions = {}
    
    for col in df.columns:
        current_type = str(df[col].dtype)
        suggested_type = current_type
        reason = ""
        
        if current_type == 'object':
            # Try to convert to numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_null_ratio = numeric_series.notna().sum() / len(df)
                
                if non_null_ratio > 0.8:
                    # Check if it could be integer
                    if (numeric_series.dropna() == numeric_series.dropna().astype(int)).all():
                        suggested_type = 'int64'
                        reason = "Can be converted to integer"
                    else:
                        suggested_type = 'float64'
                        reason = "Can be converted to float"
            except:
                pass
            
            # Try to convert to datetime
            if suggested_type == 'object':
                try:
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    non_null_ratio = datetime_series.notna().sum() / len(df)
                    
                    if non_null_ratio > 0.8:
                        suggested_type = 'datetime64[ns]'
                        reason = "Can be converted to datetime"
                except:
                    pass
            
            # Check if it should be categorical
            if suggested_type == 'object':
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.5:
                    suggested_type = 'category'
                    reason = "Low cardinality, suitable for categorical"
        
        elif current_type in ['int64', 'float64']:
            # Check if float can be int
            if current_type == 'float64':
                if df[col].dropna().equals(df[col].dropna().astype(int)):
                    suggested_type = 'int64'
                    reason = "No decimal values, can be integer"
        
        if suggested_type != current_type:
            suggestions[col] = {
                'current': current_type,
                'suggested': suggested_type,
                'reason': reason
            }
    
    return suggestions

class DataProfiler:
    """
    Class for profiling datasets and generating comprehensive reports
    """
    
    def __init__(self, df):
        self.df = df
    
    def generate_profile_report(self):
        """
        Generate a comprehensive profile report
        """
        report = {
            'overview': self._get_overview(),
            'variables': self._profile_variables(),
            'interactions': self._analyze_interactions(),
            'correlations': self._correlation_analysis(),
            'missing_values': self._missing_value_analysis(),
            'duplicate_rows': self._duplicate_analysis()
        }
        
        return report
    
    def _get_overview(self):
        """
        Get dataset overview
        """
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'n_cells': self.df.size,
            'n_missing_cells': self.df.isnull().sum().sum(),
            'missing_cells_percentage': (self.df.isnull().sum().sum() / self.df.size) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'duplicate_rows_percentage': (self.df.duplicated().sum() / len(self.df)) * 100,
            'memory_size_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
    
    def _profile_variables(self):
        """
        Profile each variable in the dataset
        """
        variables = {}
        
        for col in self.df.columns:
            col_profile = {
                'name': col,
                'type': str(self.df[col].dtype),
                'missing_count': self.df[col].isnull().sum(),
                'missing_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique_count': self.df[col].nunique(),
                'unique_percentage': (self.df[col].nunique() / len(self.df)) * 100,
            }
            
            if self.df[col].dtype in ['int64', 'float64']:
                col_profile.update({
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'median': self.df[col].median(),
                    'q25': self.df[col].quantile(0.25),
                    'q75': self.df[col].quantile(0.75),
                })
            else:
                top_values = self.df[col].value_counts().head(5)
                col_profile['top_values'] = top_values.to_dict()
            
            variables[col] = col_profile
        
        return variables
    
    def _analyze_interactions(self):
        """
        Analyze interactions between variables
        """
        # This is a placeholder for more complex interaction analysis
        # Could include chi-square tests for categorical variables, etc.
        return {'message': 'Interaction analysis not implemented yet'}
    
    def _correlation_analysis(self):
        """
        Perform correlation analysis
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': 'Not enough numeric columns for correlation analysis'}
        
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find highest correlations
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for reporting
                    high_corr_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr_pairs
        }
    
    def _missing_value_analysis(self):
        """
        Analyze missing value patterns
        """
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        return {
            'missing_by_column': missing_data.to_dict(),
            'missing_percentage_by_column': missing_percentage.to_dict(),
            'columns_with_missing': missing_data[missing_data > 0].index.tolist(),
            'complete_rows': len(self.df) - self.df.isnull().any(axis=1).sum(),
            'complete_rows_percentage': ((len(self.df) - self.df.isnull().any(axis=1).sum()) / len(self.df)) * 100
        }
    
    def _duplicate_analysis(self):
        """
        Analyze duplicate rows
        """
        duplicates = self.df.duplicated()
        
        return {
            'duplicate_count': duplicates.sum(),
            'duplicate_percentage': (duplicates.sum() / len(self.df)) * 100,
            'unique_rows': len(self.df) - duplicates.sum(),
            'first_few_duplicates': self.df[duplicates].head().to_dict() if duplicates.any() else {}
        }
