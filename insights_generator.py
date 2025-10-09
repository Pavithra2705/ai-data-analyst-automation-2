import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class InsightsGenerator:
    def __init__(self):
        self.insights = {}
    
    def generate_insights(self, df):
        """
        Generate comprehensive insights from the dataset
        """
        insights = {}
        
        # Basic dataset information
        insights['dataset_overview'] = self._get_dataset_overview(df)
        
        # Data quality insights
        insights['data_quality'] = self._analyze_data_quality(df)
        
        # Statistical insights
        insights['statistical_summary'] = self._get_statistical_insights(df)
        
        # Correlation insights
        insights['correlations'] = self._analyze_correlations(df)
        
        # Distribution insights
        insights['distributions'] = self._analyze_distributions(df)
        
        # Categorical insights
        insights['categorical_analysis'] = self._analyze_categorical_data(df)
        
        # Outlier insights
        insights['outliers'] = self._detect_outliers(df)
        
        # Feature insights
        insights['feature_insights'] = self._analyze_features(df)
        
        # Recommendations
        insights['recommendations'] = self._generate_recommendations(df, insights)
        
        return insights
    
    def _get_dataset_overview(self, df):
        """
        Get basic dataset overview
        """
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'duplicate_rows': df.duplicated().sum(),
            'total_missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100
        }
    
    def _analyze_data_quality(self, df):
        """
        Analyze data quality issues
        """
        quality_issues = {}
        
        # Missing values analysis
        missing_by_column = df.isnull().sum()
        missing_percentage = (missing_by_column / len(df)) * 100
        
        quality_issues['columns_with_missing'] = missing_by_column[missing_by_column > 0].to_dict()
        quality_issues['high_missing_columns'] = missing_percentage[missing_percentage > 50].to_dict()
        
        # Data type consistency
        object_columns = df.select_dtypes(include=['object']).columns
        potential_numeric = []
        
        for col in object_columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                potential_numeric.append(col)
            except:
                pass
        
        quality_issues['potential_numeric_columns'] = potential_numeric
        
        # Duplicate analysis
        quality_issues['duplicate_rows'] = df.duplicated().sum()
        
        return quality_issues
    
    def _get_statistical_insights(self, df):
        """
        Generate statistical insights for numerical columns
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            return {'message': 'No numerical columns found for statistical analysis'}
        
        insights = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                insights[col] = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'skewness': stats.skew(col_data),
                    'kurtosis': stats.kurtosis(col_data),
                    'coefficient_of_variation': col_data.std() / col_data.mean() if col_data.mean() != 0 else np.inf
                }
        
        return insights
    
    def _analyze_correlations(self, df):
        """
        Analyze correlations between numerical variables
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {'message': 'Need at least 2 numerical columns for correlation analysis'}
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Find strong correlations
        strong_correlations = []
        weak_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                
                if not np.isnan(corr_val):
                    if abs(corr_val) > 0.7:
                        strong_correlations.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': corr_val,
                            'strength': 'strong positive' if corr_val > 0 else 'strong negative'
                        })
                    elif abs(corr_val) < 0.1:
                        weak_correlations.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': corr_val
                        })
        
        return {
            'strong_correlations': strong_correlations,
            'weak_correlations': weak_correlations[:10],  # Limit to top 10
            'correlation_matrix_available': True
        }
    
    def _analyze_distributions(self, df):
        """
        Analyze distributions of numerical variables
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        distribution_insights = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                # Test for normality
                _, normality_p = stats.normaltest(col_data)
                is_normal = normality_p > 0.05
                
                # Detect distribution shape
                skewness = stats.skew(col_data)
                if skewness > 1:
                    shape = "right-skewed"
                elif skewness < -1:
                    shape = "left-skewed"
                else:
                    shape = "approximately symmetric"
                
                distribution_insights[col] = {
                    'is_normal': is_normal,
                    'normality_p_value': normality_p,
                    'shape': shape,
                    'skewness': skewness,
                    'unique_values': col_data.nunique(),
                    'unique_percentage': (col_data.nunique() / len(col_data)) * 100
                }
        
        return distribution_insights
    
    def _analyze_categorical_data(self, df):
        """
        Analyze categorical variables
        """
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return {'message': 'No categorical columns found'}
        
        categorical_insights = {}
        
        for col in categorical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                
                categorical_insights[col] = {
                    'unique_values': col_data.nunique(),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                    'cardinality': 'high' if col_data.nunique() > len(col_data) * 0.8 else 'low',
                    'top_5_values': value_counts.head().to_dict()
                }
        
        return categorical_insights
    
    def _detect_outliers(self, df):
        """
        Detect outliers in numerical columns using IQR method
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                
                outlier_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': (len(outliers) / len(col_data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'has_outliers': len(outliers) > 0
                }
        
        return outlier_info
    
    def _analyze_features(self, df):
        """
        Analyze feature characteristics and relationships
        """
        insights = {}
        
        # Column types distribution
        type_counts = df.dtypes.value_counts().to_dict()
        insights['data_type_distribution'] = {str(k): v for k, v in type_counts.items()}
        
        # Feature diversity (unique values ratio)
        feature_diversity = {}
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.95:
                diversity = "very high"
            elif unique_ratio > 0.8:
                diversity = "high"
            elif unique_ratio > 0.5:
                diversity = "medium"
            else:
                diversity = "low"
            
            feature_diversity[col] = {
                'unique_ratio': unique_ratio,
                'diversity': diversity,
                'unique_count': df[col].nunique()
            }
        
        insights['feature_diversity'] = feature_diversity
        
        # Constant and quasi-constant features
        constant_features = []
        quasi_constant_features = []
        
        for col in df.columns:
            unique_count = df[col].nunique()
            if unique_count == 1:
                constant_features.append(col)
            elif unique_count < 5 and df[col].value_counts().iloc[0] / len(df) > 0.95:
                quasi_constant_features.append(col)
        
        insights['constant_features'] = constant_features
        insights['quasi_constant_features'] = quasi_constant_features
        
        return insights
    
    def _generate_recommendations(self, df, insights):
        """
        Generate actionable recommendations based on insights
        """
        recommendations = []
        
        # Data quality recommendations
        data_quality = insights.get('data_quality', {})
        
        if data_quality.get('high_missing_columns'):
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'message': f"Consider removing or imputing columns with >50% missing values: {list(data_quality['high_missing_columns'].keys())}"
            })
        
        if data_quality.get('duplicate_rows', 0) > 0:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'medium',
                'message': f"Remove {data_quality['duplicate_rows']} duplicate rows to improve data quality"
            })
        
        # Feature recommendations
        feature_insights = insights.get('feature_insights', {})
        
        if feature_insights.get('constant_features'):
            recommendations.append({
                'type': 'feature_engineering',
                'priority': 'medium',
                'message': f"Remove constant features as they provide no information: {feature_insights['constant_features']}"
            })
        
        # Correlation recommendations
        correlations = insights.get('correlations', {})
        if correlations.get('strong_correlations'):
            recommendations.append({
                'type': 'analysis',
                'priority': 'low',
                'message': f"Investigate strong correlations found between variables - potential multicollinearity"
            })
        
        # Distribution recommendations
        distributions = insights.get('distributions', {})
        skewed_features = []
        
        for col, dist_info in distributions.items():
            if isinstance(dist_info, dict) and abs(dist_info.get('skewness', 0)) > 2:
                skewed_features.append(col)
        
        if skewed_features:
            recommendations.append({
                'type': 'preprocessing',
                'priority': 'low',
                'message': f"Consider log transformation for highly skewed features: {skewed_features[:5]}"
            })
        
        # Outlier recommendations
        outliers = insights.get('outliers', {})
        high_outlier_cols = []
        
        for col, outlier_info in outliers.items():
            if isinstance(outlier_info, dict) and outlier_info.get('outlier_percentage', 0) > 5:
                high_outlier_cols.append(col)
        
        if high_outlier_cols:
            recommendations.append({
                'type': 'preprocessing',
                'priority': 'medium',
                'message': f"Investigate outliers in columns: {high_outlier_cols} (>5% outliers detected)"
            })
        
        return recommendations
