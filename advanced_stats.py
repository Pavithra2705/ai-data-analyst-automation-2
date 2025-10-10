import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class AdvancedStatistics:
    def __init__(self):
        self.results = {}
    
    def perform_regression_analysis(self, df, target_col, feature_cols=None):
        """
        Perform linear regression analysis
        """
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        # Select features
        if feature_cols is None:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numerical_cols if col != target_col]
        
        if len(feature_cols) == 0:
            return {'error': 'No feature columns available for regression'}
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_col].fillna(df[target_col].mean())
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R-squared
        r_squared = model.score(X, y)
        
        # Calculate predictions and residuals
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Feature importance
        coefficients = pd.DataFrame({
            'Feature': feature_cols,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        return {
            'model_type': 'Linear Regression',
            'target': target_col,
            'features': feature_cols,
            'r_squared': r_squared,
            'intercept': model.intercept_,
            'coefficients': coefficients.to_dict('records'),
            'mean_absolute_error': np.mean(np.abs(residuals)),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'residual_stats': {
                'mean': residuals.mean(),
                'std': residuals.std(),
                'min': residuals.min(),
                'max': residuals.max()
            }
        }
    
    def perform_hypothesis_tests(self, df, col1, col2=None, test_type='auto'):
        """
        Perform various hypothesis tests with safety checks for non-numeric data.
        """
        results = {}

        if col1 not in df.columns:
            return {'error': f'Column {col1} not found'}

        # Convert to numeric if possible, else drop
        data1 = pd.to_numeric(df[col1], errors='coerce').dropna()

        # If completely non-numeric
        if data1.empty:
            return {'error': f'Column {col1} has no numeric data to analyze'}

        # ---------- SINGLE SAMPLE TESTS ----------
        if col2 is None:
            if len(data1) >= 8:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data1)
                    results['normality_test'] = {
                        'test': 'Shapiro-Wilk',
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05,
                        'interpretation': 'Data is normally distributed' if shapiro_p > 0.05 else 'Data is not normally distributed'
                    }
                except Exception as e:
                    results['normality_test_error'] = str(e)

            try:
                t_stat, t_p = stats.ttest_1samp(data1, 0)
                results['one_sample_ttest'] = {
                    'test': 'One-Sample t-test (vs 0)',
                    'statistic': t_stat,
                    'p_value': t_p,
                    'significant': t_p < 0.05,
                    'interpretation': f'Mean is {"significantly" if t_p < 0.05 else "not significantly"} different from 0'
                }
            except Exception as e:
                results['one_sample_ttest_error'] = str(e)

        # ---------- TWO SAMPLE TESTS ----------
        else:
            if col2 not in df.columns:
                return {'error': f'Column {col2} not found'}

            data2 = pd.to_numeric(df[col2], errors='coerce').dropna()

            if data2.empty:
                return {'error': f'Column {col2} has no numeric data to analyze'}

            # Both numerical - correlation and t-test
            if np.issubdtype(df[col1].dtype, np.number) and np.issubdtype(df[col2].dtype, np.number):
                try:
                    corr, corr_p = stats.pearsonr(data1, data2)
                    results['pearson_correlation'] = {
                        'test': 'Pearson Correlation',
                        'correlation': corr,
                        'p_value': corr_p,
                        'significant': corr_p < 0.05,
                        'interpretation': f'{"Significant" if corr_p < 0.05 else "No significant"} correlation (r={corr:.3f})'
                    }
                except Exception as e:
                    results['pearson_correlation_error'] = str(e)

                try:
                    t_stat, t_p = stats.ttest_ind(data1, data2)
                    results['independent_ttest'] = {
                        'test': 'Independent t-test',
                        'statistic': t_stat,
                        'p_value': t_p,
                        'significant': t_p < 0.05,
                        'interpretation': f'Means are {"significantly" if t_p < 0.05 else "not significantly"} different'
                    }
                except Exception as e:
                    results['independent_ttest_error'] = str(e)

            else:
                # Handle categorical vs numeric (ANOVA or Chi-Square)
                if np.issubdtype(df[col1].dtype, np.number):
                    cat_col, num_col = col2, col1
                else:
                    cat_col, num_col = col1, col2

                # If categorical & numeric combo → ANOVA
                if np.issubdtype(df[num_col].dtype, np.number) and not np.issubdtype(df[cat_col].dtype, np.number):
                    groups = [df[df[cat_col] == cat][num_col].dropna() for cat in df[cat_col].unique()]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) >= 2:
                        f_stat, anova_p = stats.f_oneway(*groups)
                        results['anova'] = {
                            'test': 'One-way ANOVA',
                            'statistic': f_stat,
                            'p_value': anova_p,
                            'significant': anova_p < 0.05,
                            'interpretation': f'Group means are {"significantly" if anova_p < 0.05 else "not significantly"} different'
                        }
                    else:
                        results['anova'] = {'error': 'Not enough groups for ANOVA'}

                # Both categorical → Chi-square
                elif not np.issubdtype(df[col1].dtype, np.number) and not np.issubdtype(df[col2].dtype, np.number):
                    contingency_table = pd.crosstab(df[col1], df[col2])
                    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)
                    results['chi_square'] = {
                        'test': 'Chi-Square Test of Independence',
                        'statistic': chi2,
                        'p_value': chi_p,
                        'degrees_of_freedom': dof,
                        'significant': chi_p < 0.05,
                        'interpretation': f'Variables are {"significantly" if chi_p < 0.05 else "not significantly"} associated'
                    }

        return results

    def detect_anomalies(self, df, columns=None, method='isolation_forest', contamination=0.1):
        """
        Detect anomalies using various methods
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(columns) == 0:
            return {'error': 'No numerical columns available for anomaly detection'}
        
        # Prepare data
        X = df[columns].fillna(df[columns].mean())
        
        if method == 'isolation_forest':
            # Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(X)
            
            # -1 for anomalies, 1 for normal
            anomaly_mask = predictions == -1
            anomaly_scores = iso_forest.score_samples(X)
            
            results = {
                'method': 'Isolation Forest',
                'total_anomalies': anomaly_mask.sum(),
                'anomaly_percentage': (anomaly_mask.sum() / len(df)) * 100,
                'anomaly_indices': np.where(anomaly_mask)[0].tolist()[:50],  # First 50
                'contamination_rate': contamination
            }
        
        elif method == 'statistical':
            # Statistical method (Z-score)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            z_scores = np.abs(X_scaled)
            
            # Consider anomaly if z-score > 3 in any dimension
            anomaly_mask = (z_scores > 3).any(axis=1)
            
            results = {
                'method': 'Statistical (Z-score > 3)',
                'total_anomalies': anomaly_mask.sum(),
                'anomaly_percentage': (anomaly_mask.sum() / len(df)) * 100,
                'anomaly_indices': np.where(anomaly_mask)[0].tolist()[:50]  # First 50
            }
        
        elif method == 'dbscan':
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            labels = dbscan.fit_predict(StandardScaler().fit_transform(X))
            
            # -1 label indicates anomalies/noise points
            anomaly_mask = labels == -1
            
            results = {
                'method': 'DBSCAN Clustering',
                'total_anomalies': anomaly_mask.sum(),
                'anomaly_percentage': (anomaly_mask.sum() / len(df)) * 100,
                'anomaly_indices': np.where(anomaly_mask)[0].tolist()[:50],  # First 50
                'num_clusters': len(set(labels)) - (1 if -1 in labels else 0)
            }
        
        # Add anomaly details
        if anomaly_mask.sum() > 0:
            anomaly_data = df[anomaly_mask][columns].head(10)
            results['sample_anomalies'] = anomaly_data.to_dict('records')
        
        return results
    
    def perform_distribution_analysis(self, df, column):
        """
        Detailed distribution analysis
        """
        if column not in df.columns:
            return {'error': f'Column {column} not found'}
        
        data = df[column].dropna()
        
        if len(data) == 0:
            return {'error': f'No data in column {column}'}
        
        # Basic statistics
        results = {
            'column': column,
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'mode': data.mode().iloc[0] if not data.mode().empty else None,
            'std': data.std(),
            'variance': data.var(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'q1': data.quantile(0.25),
            'q3': data.quantile(0.75),
            'iqr': data.quantile(0.75) - data.quantile(0.25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.inf
        }
        
        # Distribution shape
        if results['skewness'] > 1:
            results['shape'] = 'Right-skewed (positively skewed)'
        elif results['skewness'] < -1:
            results['shape'] = 'Left-skewed (negatively skewed)'
        else:
            results['shape'] = 'Approximately symmetric'
        
        # Kurtosis interpretation
        if results['kurtosis'] > 3:
            results['tail_behavior'] = 'Heavy tails (leptokurtic)'
        elif results['kurtosis'] < -3:
            results['tail_behavior'] = 'Light tails (platykurtic)'
        else:
            results['tail_behavior'] = 'Normal tails (mesokurtic)'
        
        # Normality test
        if len(data) >= 8:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            results['normality_test'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Outlier detection
        lower_bound = results['q1'] - 1.5 * results['iqr']
        upper_bound = results['q3'] + 1.5 * results['iqr']
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        results['outliers'] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return results
