import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataCleaner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def clean_data(self, df, remove_duplicates=True, missing_strategy="Auto", 
                   detect_outliers=True, normalize=False):
        """
        Comprehensive data cleaning pipeline
        """
        original_shape = df.shape
        cleaning_report = {}
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Step 1: Data type inference and conversion
        cleaned_df, type_report = self._infer_and_convert_types(cleaned_df)
        cleaning_report['data_type_conversion'] = type_report
        
        # Step 2: Remove duplicates
        if remove_duplicates:
            duplicates_before = cleaned_df.duplicated().sum()
            cleaned_df = cleaned_df.drop_duplicates()
            duplicates_removed = duplicates_before - cleaned_df.duplicated().sum()
            cleaning_report['duplicate_removal'] = {
                'duplicates_found': duplicates_before,
                'duplicates_removed': duplicates_removed,
                'rows_after_removal': len(cleaned_df)
            }
        
        # Step 3: Handle missing values
        cleaned_df, missing_report = self._handle_missing_values(cleaned_df, missing_strategy)
        cleaning_report['missing_value_handling'] = missing_report
        
        # Step 4: Outlier detection and handling
        if detect_outliers:
            cleaned_df, outlier_report = self._handle_outliers(cleaned_df)
            cleaning_report['outlier_handling'] = outlier_report
        
        # Step 5: Data normalization
        if normalize:
            cleaned_df, norm_report = self._normalize_data(cleaned_df)
            cleaning_report['normalization'] = norm_report
        
        # Final summary
        cleaning_report['summary'] = {
            'original_shape': original_shape,
            'final_shape': cleaned_df.shape,
            'rows_removed': original_shape[0] - cleaned_df.shape[0],
            'columns_removed': original_shape[1] - cleaned_df.shape[1],
            'cleaning_efficiency': f"{((original_shape[0] - cleaned_df.shape[0]) / original_shape[0] * 100):.2f}% rows removed"
        }
        
        return cleaned_df, cleaning_report
    
    def _infer_and_convert_types(self, df):
        """
        Automatically infer and convert data types with better handling for complex objects
        """
        report = {}
        converted_columns = []
        
        for column in df.columns:
            original_dtype = str(df[column].dtype)
            
            # Handle complex objects from JSON (dicts, lists) by converting to strings FIRST
            if df[column].dtype == 'object':
                try:
                    # Check if column contains complex objects (dict, list)
                    sample_val = df[column].dropna().iloc[0] if len(df[column].dropna()) > 0 else None
                    if sample_val is not None and isinstance(sample_val, (dict, list)):
                        df[column] = df[column].apply(lambda x: str(x) if isinstance(x, (dict, list)) else x)
                        converted_columns.append({
                            'column': column,
                            'from': original_dtype,
                            'to': 'string (from complex object)'
                        })
                        continue
                except Exception as e:
                    # If there's any issue, convert to string as fallback
                    df[column] = df[column].astype(str)
                    continue
                
                # Check if it looks like a date
                sample_values = df[column].dropna().head(100)
                if len(sample_values) > 0:
                    try:
                        pd.to_datetime(sample_values, infer_datetime_format=True, errors='raise')
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                        if str(df[column].dtype) != original_dtype:
                            converted_columns.append({
                                'column': column,
                                'from': original_dtype,
                                'to': str(df[column].dtype)
                            })
                    except:
                        # Try to convert to numeric
                        try:
                            numeric_series = pd.to_numeric(df[column], errors='coerce')
                            # Only convert if most values are successfully converted
                            if numeric_series.notna().sum() / len(df) > 0.8:
                                df[column] = numeric_series
                                converted_columns.append({
                                    'column': column,
                                    'from': original_dtype,
                                    'to': str(df[column].dtype)
                                })
                        except:
                            # Keep as string/object if all conversions fail
                            pass
        
        report['converted_columns'] = converted_columns
        report['final_dtypes'] = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        
        return df, report
    
    def _handle_missing_values(self, df, strategy):
        """
        Handle missing values based on strategy
        """
        missing_before = df.isnull().sum().sum()
        missing_by_column = df.isnull().sum().to_dict()
        
        report = {
            'missing_values_before': missing_before,
            'missing_by_column_before': missing_by_column,
            'strategy_used': strategy
        }
        
        if strategy == "Drop rows":
            df = df.dropna()
        elif strategy == "Drop columns":
            # Drop columns with more than 50% missing values
            threshold = len(df) * 0.5
            df = df.dropna(axis=1, thresh=threshold)
        elif strategy == "Fill with mean/mode":
            # Fill numerical columns with mean
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].mean(), inplace=True)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
        
        elif strategy == "Forward fill":
            df = df.fillna(method='ffill')
        elif strategy == "Backward fill":
            df = df.fillna(method='bfill')
        elif strategy == "Auto":
            # Intelligent strategy selection
            for column in df.columns:
                missing_ratio = df[column].isnull().sum() / len(df)
                
                if missing_ratio > 0.5:
                    # Drop columns with too many missing values
                    df = df.drop(columns=[column])
                elif missing_ratio > 0:
                    if df[column].dtype in ['int64', 'float64']:
                        # Use mean for numerical
                        df[column].fillna(df[column].mean(), inplace=True)
                    else:
                        # Use mode for categorical
                        mode_value = df[column].mode().iloc[0] if not df[column].mode().empty else 'Unknown'
                        df[column].fillna(mode_value, inplace=True)
        
        missing_after = df.isnull().sum().sum()
        missing_by_column_after = df.isnull().sum().to_dict()
        
        report.update({
            'missing_values_after': missing_after,
            'missing_by_column_after': missing_by_column_after,
            'missing_values_resolved': missing_before - missing_after
        })
        
        return df, report
    
    def _handle_outliers(self, df):
        """
        Detect and handle outliers using IQR method
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        total_outliers = 0
        
        for column in numerical_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers_count = outliers_mask.sum()
            total_outliers += outliers_count
            
            if outliers_count > 0:
                # Cap outliers instead of removing them
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound
                
                outlier_report[column] = {
                    'outliers_found': outliers_count,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'action': 'capped'
                }
        
        outlier_report['total_outliers'] = total_outliers
        outlier_report['method'] = 'IQR with capping'
        
        return df, outlier_report
    
    def _normalize_data(self, df):
        """
        Normalize numerical columns
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        normalized_columns = []
        
        if len(numerical_cols) > 0:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            normalized_columns = list(numerical_cols)
        
        report = {
            'normalized_columns': normalized_columns,
            'method': 'StandardScaler (z-score normalization)',
            'mean': 0,
            'std': 1
        }
        
        return df, report
