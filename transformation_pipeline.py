import pandas as pd
import numpy as np
from datetime import datetime
import json

class TransformationPipeline:
    def __init__(self):
        self.transformations = []
        self.history = []
    
    def add_transformation(self, transformation_type, params):
        """
        Add a transformation to the pipeline
        """
        self.transformations.append({
            'type': transformation_type,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
    
    def apply_pipeline(self, df):
        """
        Apply all transformations in the pipeline
        """
        result_df = df.copy()
        execution_log = []
        
        for idx, transform in enumerate(self.transformations):
            try:
                before_shape = result_df.shape
                result_df = self._apply_single_transformation(result_df, transform)
                after_shape = result_df.shape
                
                execution_log.append({
                    'step': idx + 1,
                    'type': transform['type'],
                    'params': transform['params'],
                    'before_shape': before_shape,
                    'after_shape': after_shape,
                    'status': 'success'
                })
            except Exception as e:
                execution_log.append({
                    'step': idx + 1,
                    'type': transform['type'],
                    'params': transform['params'],
                    'status': 'error',
                    'error': str(e)
                })
        
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'original_shape': df.shape,
            'final_shape': result_df.shape,
            'execution_log': execution_log
        })
        
        return result_df, execution_log
    
    def _apply_single_transformation(self, df, transform):
        """
        Apply a single transformation
        """
        transform_type = transform['type']
        params = transform['params']
        
        if transform_type == 'filter_rows':
            return self._filter_rows(df, params)
        elif transform_type == 'select_columns':
            return self._select_columns(df, params)
        elif transform_type == 'drop_columns':
            return self._drop_columns(df, params)
        elif transform_type == 'rename_column':
            return self._rename_column(df, params)
        elif transform_type == 'create_column':
            return self._create_column(df, params)
        elif transform_type == 'fill_missing':
            return self._fill_missing(df, params)
        elif transform_type == 'replace_values':
            return self._replace_values(df, params)
        elif transform_type == 'convert_type':
            return self._convert_type(df, params)
        elif transform_type == 'sort_data':
            return self._sort_data(df, params)
        elif transform_type == 'group_aggregate':
            return self._group_aggregate(df, params)
        elif transform_type == 'bin_column':
            return self._bin_column(df, params)
        elif transform_type == 'normalize_column':
            return self._normalize_column(df, params)
        elif transform_type == 'extract_datetime':
            return self._extract_datetime(df, params)
        else:
            raise ValueError(f"Unknown transformation type: {transform_type}")
    
    def _filter_rows(self, df, params):
        """Filter rows based on condition"""
        column = params['column']
        operator = params['operator']
        value = params['value']
        
        if operator == 'equals':
            return df[df[column] == value]
        elif operator == 'not_equals':
            return df[df[column] != value]
        elif operator == 'greater_than':
            return df[df[column] > float(value)]
        elif operator == 'less_than':
            return df[df[column] < float(value)]
        elif operator == 'contains':
            return df[df[column].astype(str).str.contains(str(value), na=False)]
        elif operator == 'not_contains':
            return df[~df[column].astype(str).str.contains(str(value), na=False)]
        elif operator == 'is_null':
            return df[df[column].isnull()]
        elif operator == 'not_null':
            return df[df[column].notnull()]
        else:
            return df
    
    def _select_columns(self, df, params):
        """Select specific columns"""
        columns = params['columns']
        return df[columns]
    
    def _drop_columns(self, df, params):
        """Drop specific columns"""
        columns = params['columns']
        return df.drop(columns=columns, errors='ignore')
    
    def _rename_column(self, df, params):
        """Rename a column"""
        old_name = params['old_name']
        new_name = params['new_name']
        return df.rename(columns={old_name: new_name})
    
    def _create_column(self, df, params):
        """Create a new column based on expression"""
        new_column = params['name']
        expression = params['expression']
        
        # Safe eval with limited scope
        allowed_functions = {
            'abs': abs, 'round': round, 'len': len,
            'str': str, 'int': int, 'float': float,
            'min': min, 'max': max, 'sum': sum
        }
        
        # Simple expression evaluation
        if '+' in expression or '-' in expression or '*' in expression or '/' in expression:
            # Mathematical expression between columns
            df[new_column] = df.eval(expression)
        else:
            # Copy or transformation of single column
            if expression in df.columns:
                df[new_column] = df[expression]
            else:
                df[new_column] = expression
        
        return df
    
    def _fill_missing(self, df, params):
        """Fill missing values"""
        column = params['column']
        method = params['method']
        value = params.get('value', None)
        
        if method == 'value':
            df[column].fillna(value, inplace=True)
        elif method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == 'median':
            df[column].fillna(df[column].median(), inplace=True)
        elif method == 'mode':
            mode_val = df[column].mode().iloc[0] if not df[column].mode().empty else 0
            df[column].fillna(mode_val, inplace=True)
        elif method == 'forward_fill':
            df[column].fillna(method='ffill', inplace=True)
        elif method == 'backward_fill':
            df[column].fillna(method='bfill', inplace=True)
        
        return df
    
    def _replace_values(self, df, params):
        """Replace specific values"""
        column = params['column']
        old_value = params['old_value']
        new_value = params['new_value']
        
        df[column] = df[column].replace(old_value, new_value)
        return df
    
    def _convert_type(self, df, params):
        """Convert column data type"""
        column = params['column']
        new_type = params['type']
        
        if new_type == 'int':
            df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
        elif new_type == 'float':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif new_type == 'string':
            df[column] = df[column].astype(str)
        elif new_type == 'datetime':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        elif new_type == 'category':
            df[column] = df[column].astype('category')
        
        return df
    
    def _sort_data(self, df, params):
        """Sort data by columns"""
        columns = params['columns'] if isinstance(params['columns'], list) else [params['columns']]
        ascending = params.get('ascending', True)
        
        return df.sort_values(by=columns, ascending=ascending)
    
    def _group_aggregate(self, df, params):
        """Group and aggregate data"""
        group_by = params['group_by']
        agg_column = params['agg_column']
        agg_function = params['agg_function']
        
        if agg_function == 'mean':
            result = df.groupby(group_by)[agg_column].mean().reset_index()
        elif agg_function == 'sum':
            result = df.groupby(group_by)[agg_column].sum().reset_index()
        elif agg_function == 'count':
            result = df.groupby(group_by)[agg_column].count().reset_index()
        elif agg_function == 'min':
            result = df.groupby(group_by)[agg_column].min().reset_index()
        elif agg_function == 'max':
            result = df.groupby(group_by)[agg_column].max().reset_index()
        else:
            result = df
        
        return result
    
    def _bin_column(self, df, params):
        """Bin numerical column into categories"""
        column = params['column']
        bins = params['bins']
        labels = params.get('labels', None)
        
        new_column = f"{column}_binned"
        df[new_column] = pd.cut(df[column], bins=bins, labels=labels)
        
        return df
    
    def _normalize_column(self, df, params):
        """Normalize numerical column"""
        column = params['column']
        method = params.get('method', 'minmax')
        
        if method == 'minmax':
            # Min-max normalization (0-1 scale)
            min_val = df[column].min()
            max_val = df[column].max()
            df[f"{column}_normalized"] = (df[column] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            # Z-score normalization
            mean_val = df[column].mean()
            std_val = df[column].std()
            df[f"{column}_normalized"] = (df[column] - mean_val) / std_val
        
        return df
    
    def _extract_datetime(self, df, params):
        """Extract components from datetime column"""
        column = params['column']
        component = params['component']
        
        # Ensure column is datetime
        if df[column].dtype != 'datetime64[ns]':
            df[column] = pd.to_datetime(df[column], errors='coerce')
        
        if component == 'year':
            df[f"{column}_year"] = df[column].dt.year
        elif component == 'month':
            df[f"{column}_month"] = df[column].dt.month
        elif component == 'day':
            df[f"{column}_day"] = df[column].dt.day
        elif component == 'dayofweek':
            df[f"{column}_dayofweek"] = df[column].dt.dayofweek
        elif component == 'hour':
            df[f"{column}_hour"] = df[column].dt.hour
        elif component == 'quarter':
            df[f"{column}_quarter"] = df[column].dt.quarter
        
        return df
    
    def clear_pipeline(self):
        """Clear all transformations"""
        self.transformations = []
    
    def export_pipeline(self):
        """Export pipeline as JSON"""
        return json.dumps(self.transformations, indent=2)
    
    def import_pipeline(self, pipeline_json):
        """Import pipeline from JSON"""
        self.transformations = json.loads(pipeline_json)
    
    def get_pipeline_summary(self):
        """Get summary of current pipeline"""
        return {
            'total_steps': len(self.transformations),
            'transformations': [
                {
                    'step': idx + 1,
                    'type': t['type'],
                    'params': t['params']
                }
                for idx, t in enumerate(self.transformations)
            ]
        }
