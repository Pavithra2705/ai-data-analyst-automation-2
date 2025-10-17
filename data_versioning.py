import pandas as pd
import json
from datetime import datetime
import hashlib
import pickle

class DataVersioning:
    def __init__(self):
        self.versions = []
        self.current_version = 0
    
    def create_version(self, df, action_description, metadata=None):
        """
        Create a new version of the dataset
        """
        version_id = len(self.versions) + 1
        timestamp = datetime.now().isoformat()
        
        # Create data hash for integrity - handle unhashable types
        try:
            # Try to convert dict/list columns to strings first
            df_copy = df.copy()
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].apply(
                        lambda x: str(x) if isinstance(x, (dict, list)) else x,
                        errors='ignore'
                    )
            data_hash = hashlib.md5(pd.util.hash_pandas_object(df_copy, index=True).values).hexdigest()
        except Exception as e:
            # Fallback: hash based on shape, columns, and content
            hash_input = f"{df.shape}_{list(df.columns)}_{datetime.now().isoformat()}"
            data_hash = hashlib.md5(hash_input.encode()).hexdigest()
        
        version_info = {
            'version_id': version_id,
            'timestamp': timestamp,
            'action': action_description,
            'shape': df.shape,
            'columns': list(df.columns),
            'data_hash': data_hash,
            'metadata': metadata or {}
        }
        
        # Store serialized dataframe
        version_info['data'] = pickle.dumps(df)
        
        self.versions.append(version_info)
        self.current_version = version_id
        
        return version_id
    
    def get_version(self, version_id):
        """
        Retrieve a specific version
        """
        for version in self.versions:
            if version['version_id'] == version_id:
                df = pickle.loads(version['data'])
                return df, version
        return None, None
    
    def rollback(self, version_id):
        """
        Rollback to a specific version
        """
        df, version_info = self.get_version(version_id)
        if df is not None:
            self.current_version = version_id
            return df, version_info
        return None, None
    
    def get_version_history(self):
        """
        Get history of all versions
        """
        history = []
        for version in self.versions:
            history.append({
                'version_id': version['version_id'],
                'timestamp': version['timestamp'],
                'action': version['action'],
                'shape': version['shape'],
                'columns_count': len(version['columns']),
                'data_hash': version['data_hash'][:8]  # Short hash for display
            })
        return history
    
    def compare_versions(self, version_id1, version_id2):
        """
        Compare two versions
        """
        df1, info1 = self.get_version(version_id1)
        df2, info2 = self.get_version(version_id2)
        
        if df1 is None or df2 is None:
            return None
        
        comparison = {
            'version_1': {
                'id': version_id1,
                'timestamp': info1['timestamp'],
                'shape': info1['shape'],
                'action': info1['action']
            },
            'version_2': {
                'id': version_id2,
                'timestamp': info2['timestamp'],
                'shape': info2['shape'],
                'action': info2['action']
            },
            'differences': {
                'rows_diff': df2.shape[0] - df1.shape[0],
                'cols_diff': df2.shape[1] - df1.shape[1],
                'columns_added': list(set(df2.columns) - set(df1.columns)),
                'columns_removed': list(set(df1.columns) - set(df2.columns)),
            }
        }
        
        # Check for value changes in common columns
        common_cols = set(df1.columns) & set(df2.columns)
        if common_cols and df1.shape[0] == df2.shape[0]:
            value_changes = {}
            for col in common_cols:
                if df1[col].dtype == df2[col].dtype:
                    try:
                        changed_rows = (df1[col] != df2[col]).sum()
                        if changed_rows > 0:
                            value_changes[col] = changed_rows
                    except:
                        pass
            
            comparison['differences']['value_changes'] = value_changes
        
        return comparison
    
    def delete_version(self, version_id):
        """
        Delete a specific version
        """
        if version_id == self.current_version:
            return False, "Cannot delete current version"
        
        self.versions = [v for v in self.versions if v['version_id'] != version_id]
        return True, "Version deleted successfully"
    
    def get_current_version(self):
        """
        Get the current version
        """
        if self.current_version > 0:
            return self.get_version(self.current_version)
        return None, None
    
    def export_version_history(self):
        """
        Export version history as JSON (without data)
        """
        history = []
        for version in self.versions:
            history.append({
                'version_id': version['version_id'],
                'timestamp': version['timestamp'],
                'action': version['action'],
                'shape': version['shape'],
                'columns': version['columns'],
                'data_hash': version['data_hash'],
                'metadata': version['metadata']
            })
        
        return json.dumps(history, indent=2)
    
    def get_version_count(self):
        """
        Get total number of versions
        """
        return len(self.versions)
    
    def clear_all_versions(self):
        """
        Clear all versions
        """
        self.versions = []
        self.current_version = 0
