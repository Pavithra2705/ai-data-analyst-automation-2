import os
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class LLaMAChat:
    def __init__(self, model_name=None):
        """
        Initialize the chatbot with enhanced rule-based AI responses
        Uses intelligent pattern matching and context-aware responses
        """
        self.use_rule_based = True
        print("Using enhanced AI-powered rule-based chatbot")
    
    def generate_response(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate intelligent response to user question about the dataset
        """
        return self._enhanced_ai_response(question, context)
    
    def _enhanced_ai_response(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate enhanced AI-like response with context awareness
        """
        question_lower = question.lower()
        
        # Advanced pattern matching with multi-intent detection
        intents = []
        
        if any(word in question_lower for word in ['overview', 'summary', 'describe', 'what is', 'about', 'tell me about']):
            intents.append('overview')
        if any(word in question_lower for word in ['missing', 'null', 'nan', 'empty', 'incomplete']):
            intents.append('missing')
        if any(word in question_lower for word in ['correlation', 'relationship', 'correlated', 'related', 'connection']):
            intents.append('correlation')
        if any(word in question_lower for word in ['statistics', 'stats', 'mean', 'average', 'median', 'distribution']):
            intents.append('statistics')
        if any(word in question_lower for word in ['column', 'feature', 'variable', 'field']):
            intents.append('columns')
        if any(word in question_lower for word in ['recommend', 'suggest', 'should', 'advice', 'what to do', 'next steps']):
            intents.append('recommendations')
        if any(word in question_lower for word in ['outlier', 'anomaly', 'unusual', 'extreme']):
            intents.append('outliers')
        if any(word in question_lower for word in ['quality', 'clean', 'issues', 'problems']):
            intents.append('quality')
        
        # If no specific intent detected, use general help
        if not intents:
            intents = ['general']
        
        # Multi-intent response generation
        if len(intents) == 1:
            return self._get_response_for_intent(intents[0], question, context)
        else:
            # Combine multiple intents into comprehensive response
            return self._get_multi_intent_response(intents, question, context)
    
    def _get_response_for_intent(self, intent: str, question: str, context: Dict[str, Any]) -> str:
        """
        Get response for a single intent
        """
        intent_map = {
            'overview': self._get_dataset_overview,
            'missing': self._get_missing_values_info,
            'correlation': self._get_correlation_info,
            'statistics': self._get_statistics_info,
            'columns': self._get_column_info,
            'recommendations': self._get_recommendations,
            'outliers': self._get_outlier_info,
            'quality': self._get_data_quality_info,
            'general': self._get_general_help
        }
        
        handler = intent_map.get(intent, self._get_general_help)
        return handler(context)
    
    def _get_multi_intent_response(self, intents: List[str], question: str, context: Dict[str, Any]) -> str:
        """
        Generate comprehensive response for multiple intents
        """
        response_parts = []
        
        # Prioritize intents
        priority_order = ['overview', 'quality', 'missing', 'statistics', 'correlation', 'outliers', 'columns', 'recommendations']
        
        sorted_intents = sorted(intents, key=lambda x: priority_order.index(x) if x in priority_order else len(priority_order))
        
        for intent in sorted_intents[:3]:  # Limit to top 3 intents
            response_parts.append(self._get_response_for_intent(intent, question, context))
        
        return "\n\n---\n\n".join(response_parts)
    
    def _create_context_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create a structured context prompt from dataset information
        """
        context_parts = []
        
        # Dataset shape
        if 'shape' in context:
            context_parts.append(f"Dataset shape: {context['shape'][0]} rows, {context['shape'][1]} columns")
        
        # Columns
        if 'columns' in context:
            context_parts.append(f"Columns: {', '.join(context['columns'])}")
        
        # Data types
        if 'dtypes' in context:
            dtypes_summary = {}
            for col, dtype in context['dtypes'].items():
                dtype_str = str(dtype)
                if dtype_str not in dtypes_summary:
                    dtypes_summary[dtype_str] = []
                dtypes_summary[dtype_str].append(col)
            
            for dtype, cols in dtypes_summary.items():
                context_parts.append(f"{dtype} columns: {', '.join(cols)}")
        
        # Missing values
        if 'missing_values' in context:
            missing_info = []
            for col, missing_count in context['missing_values'].items():
                if missing_count > 0:
                    missing_info.append(f"{col}: {missing_count}")
            
            if missing_info:
                context_parts.append(f"Missing values: {', '.join(missing_info)}")
        
        # Statistics
        if 'summary' in context and context['summary']:
            context_parts.append("Basic statistics available for numerical columns")
        
        return "\n".join(context_parts)
    
    def _get_dataset_overview(self, context: Dict[str, Any]) -> str:
        """
        Provide dataset overview
        """
        shape = context.get('shape', (0, 0))
        columns = context.get('columns', [])
        
        response = f"📊 **Dataset Overview**\n\n"
        response += f"This dataset contains **{shape[0]:,} rows** and **{shape[1]} columns**.\n\n"
        
        if columns:
            response += f"**Columns:** {', '.join(columns[:10])}"
            if len(columns) > 10:
                response += f" and {len(columns) - 10} more..."
            response += "\n\n"
        
        # Data types summary
        if 'dtypes' in context:
            dtypes_count = {}
            for dtype in context['dtypes'].values():
                dtype_str = str(dtype)
                dtypes_count[dtype_str] = dtypes_count.get(dtype_str, 0) + 1
            
            response += "**Data Types:**\n"
            for dtype, count in dtypes_count.items():
                response += f"- {dtype}: {count} columns\n"
        
        return response
    
    def _get_dataset_shape(self, context: Dict[str, Any]) -> str:
        """
        Provide dataset shape information
        """
        shape = context.get('shape', (0, 0))
        return f"📏 **Dataset Dimensions**\n\nThe dataset has **{shape[0]:,} rows** (observations) and **{shape[1]} columns** (features)."
    
    def _get_missing_values_info(self, context: Dict[str, Any]) -> str:
        """
        Provide missing values information
        """
        missing_values = context.get('missing_values', {})
        
        response = "🔍 **Missing Values Analysis**\n\n"
        
        missing_cols = [(col, count) for col, count in missing_values.items() if count > 0]
        
        if not missing_cols:
            response += "✅ Great news! No missing values found in the dataset."
        else:
            response += f"Found missing values in {len(missing_cols)} columns:\n\n"
            
            # Sort by missing count
            missing_cols.sort(key=lambda x: x[1], reverse=True)
            
            for col, count in missing_cols[:10]:  # Show top 10
                percentage = (count / context.get('shape', (1, 1))[0]) * 100
                response += f"- **{col}**: {count:,} missing ({percentage:.1f}%)\n"
            
            if len(missing_cols) > 10:
                response += f"\n... and {len(missing_cols) - 10} more columns with missing values."
        
        return response
    
    def _get_data_types_info(self, context: Dict[str, Any]) -> str:
        """
        Provide data types information
        """
        dtypes = context.get('dtypes', {})
        
        response = "🏷️ **Data Types Summary**\n\n"
        
        if not dtypes:
            response += "No data type information available."
            return response
        
        # Group by data type
        type_groups = {}
        for col, dtype in dtypes.items():
            dtype_str = str(dtype)
            if dtype_str not in type_groups:
                type_groups[dtype_str] = []
            type_groups[dtype_str].append(col)
        
        for dtype, cols in type_groups.items():
            response += f"**{dtype}** ({len(cols)} columns):\n"
            response += f"  {', '.join(cols[:5])}"
            if len(cols) > 5:
                response += f" and {len(cols) - 5} more..."
            response += "\n\n"
        
        return response
    
    def _get_statistics_info(self, context: Dict[str, Any]) -> str:
        """
        Provide basic statistics information
        """
        summary = context.get('summary', {})
        
        response = "📈 **Statistical Summary**\n\n"
        
        if not summary:
            response += "No statistical summary available. This might be because there are no numerical columns in the dataset."
            return response
        
        response += "Here are the basic statistics for numerical columns:\n\n"
        
        # Show summary for first few columns
        for col, stats in list(summary.items())[:5]:
            if isinstance(stats, dict):
                response += f"**{col}:**\n"
                if 'mean' in stats:
                    response += f"  - Mean: {stats['mean']:.2f}\n"
                if 'std' in stats:
                    response += f"  - Standard Deviation: {stats['std']:.2f}\n"
                if 'min' in stats:
                    response += f"  - Min: {stats['min']:.2f}\n"
                if 'max' in stats:
                    response += f"  - Max: {stats['max']:.2f}\n"
                response += "\n"
        
        return response
    
    def _get_correlation_info(self, context: Dict[str, Any]) -> str:
        """
        Provide correlation information
        """
        return "🔗 **Correlation Analysis**\n\nTo analyze correlations between variables, please use the Visualization tab to create a correlation heatmap. This will show you which variables are most strongly related to each other.\n\nLook for correlation values close to 1 (strong positive correlation) or -1 (strong negative correlation)."
    
    def _get_column_info(self, context: Dict[str, Any]) -> str:
        """
        Provide column information
        """
        columns = context.get('columns', [])
        
        response = "📋 **Column Information**\n\n"
        
        if not columns:
            response += "No column information available."
            return response
        
        response += f"The dataset has **{len(columns)} columns**:\n\n"
        
        # Show first 15 columns
        for i, col in enumerate(columns[:15], 1):
            response += f"{i}. {col}\n"
        
        if len(columns) > 15:
            response += f"\n... and {len(columns) - 15} more columns."
        
        return response
    
    def _get_recommendations(self, context: Dict[str, Any]) -> str:
        """
        Provide data analysis recommendations
        """
        recommendations = []
        shape = context.get('shape', (0, 0))
        missing_values = context.get('missing_values', {})
        
        # Missing value recommendations
        high_missing = [(col, count) for col, count in missing_values.items() if count > shape[0] * 0.3]
        if high_missing:
            recommendations.append(f"⚠️ **Data Quality**: Consider handling missing values in {len(high_missing)} columns with >30% missing data")
        
        # Correlation recommendations
        recommendations.append("📊 **Analysis**: Start by exploring the correlation heatmap to identify relationships between numerical variables")
        
        # Visualization recommendations
        numerical_count = sum(1 for dtype in context.get('dtypes', {}).values() if 'int' in str(dtype) or 'float' in str(dtype))
        if numerical_count > 0:
            recommendations.append(f"📈 **Visualizations**: Create distribution plots for your {numerical_count} numerical columns to understand their patterns")
        
        response = "💡 **Recommended Next Steps:**\n\n" + "\n\n".join(recommendations)
        return response
    
    def _get_outlier_info(self, context: Dict[str, Any]) -> str:
        """
        Provide outlier detection information
        """
        return """🔍 **Outlier Detection:**

To detect outliers in your dataset:

1. **Use Box Plots**: Navigate to the Visualize tab and create box plots for numerical columns. Outliers appear as individual points beyond the whiskers.

2. **IQR Method**: The cleaning process automatically detects outliers using the Interquartile Range (IQR) method and caps them to maintain data integrity.

3. **Visual Inspection**: Look for extreme values in your distribution plots that are far from the main data cluster.

Outliers can indicate:
- Data entry errors
- Rare but valid observations  
- Special cases requiring investigation

Would you like to explore specific columns for outliers?"""
    
    def _get_data_quality_info(self, context: Dict[str, Any]) -> str:
        """
        Provide data quality assessment
        """
        shape = context.get('shape', (0, 0))
        missing_values = context.get('missing_values', {})
        
        total_missing = sum(missing_values.values())
        missing_pct = (total_missing / (shape[0] * shape[1]) * 100) if shape[0] * shape[1] > 0 else 0
        
        response = "✅ **Data Quality Assessment:**\n\n"
        
        # Overall quality score
        if missing_pct < 5:
            response += f"**Overall Quality**: Excellent ({missing_pct:.1f}% missing data)\n\n"
        elif missing_pct < 15:
            response += f"**Overall Quality**: Good ({missing_pct:.1f}% missing data)\n\n"
        elif missing_pct < 30:
            response += f"**Overall Quality**: Fair ({missing_pct:.1f}% missing data)\n\n"
        else:
            response += f"**Overall Quality**: Needs improvement ({missing_pct:.1f}% missing data)\n\n"
        
        # Completeness
        complete_cols = [col for col, count in missing_values.items() if count == 0]
        response += f"**Completeness**: {len(complete_cols)}/{shape[1]} columns are complete\n\n"
        
        # Recommendations
        if missing_pct > 15:
            response += "**Recommendation**: Use the Clean tab to handle missing values before analysis"
        else:
            response += "**Status**: Your data is ready for analysis!"
        
        return response
    
    def _get_general_help(self, context: Dict[str, Any]) -> str:
        """
        Provide general help
        """
        return """🤖 **AI Data Assistant - How I Can Help:**

I can answer questions about:
- 📊 Dataset overview and structure
- 🔍 Missing values and data quality
- 📋 Column information and data types  
- 📈 Basic statistics and distributions
- 🔗 Correlations and relationships
- ⚠️ Outliers and anomalies
- 💡 Analysis recommendations

**Example questions:**
- "What are the main characteristics of this dataset?"
- "Which columns have the most missing values?"
- "What correlations exist in my data?"
- "What should I analyze first?"
- "Tell me about data quality issues"

**Pro tip**: Ask complex questions like "What are the data quality issues and what should I do about them?" for comprehensive insights!"""
