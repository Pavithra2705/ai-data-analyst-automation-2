import os
import json
from typing import Dict, Any, List
import pandas as pd
import numpy as np

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LLaMAChat:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a local model
        Using DialoGPT as a fallback if LLaMA is not available
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if not TRANSFORMERS_AVAILABLE:
            self.use_rule_based = True
            print("Transformers not available, using rule-based responses")
        else:
            self.use_rule_based = False
            self._load_model()
    
    def _load_model(self):
        """
        Load the language model
        """
        try:
            # Try to load a smaller, more efficient model first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create pipeline for text generation
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print(f"Successfully loaded model: {self.model_name}")
            
        except Exception as e:
            print(f"Failed to load model {self.model_name}: {str(e)}")
            print("Falling back to rule-based responses")
            self.use_rule_based = True
    
    def generate_response(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate response to user question about the dataset
        """
        if self.use_rule_based:
            return self._rule_based_response(question, context)
        else:
            return self._ai_response(question, context)
    
    def _ai_response(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate AI response using the loaded model
        """
        try:
            # Create context prompt
            context_str = self._create_context_prompt(context)
            
            # Create full prompt
            prompt = f"""You are a data analyst assistant. Here is information about the dataset:

{context_str}

User Question: {question}

Please provide a helpful and accurate response based on the dataset information above. Be specific and cite relevant statistics when possible.

Answer:"""

            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part
            answer_start = generated_text.find("Answer:") + len("Answer:")
            answer = generated_text[answer_start:].strip()
            
            # If answer is empty or too short, fall back to rule-based
            if len(answer) < 10:
                return self._rule_based_response(question, context)
            
            return answer
            
        except Exception as e:
            print(f"Error in AI response generation: {str(e)}")
            return self._rule_based_response(question, context)
    
    def _rule_based_response(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generate rule-based response when AI model is not available
        """
        question_lower = question.lower()
        
        # Dataset overview questions
        if any(word in question_lower for word in ['overview', 'summary', 'describe', 'what is', 'about']):
            return self._get_dataset_overview(context)
        
        # Shape/size questions
        elif any(word in question_lower for word in ['shape', 'size', 'rows', 'columns', 'dimensions']):
            return self._get_dataset_shape(context)
        
        # Missing values questions
        elif any(word in question_lower for word in ['missing', 'null', 'nan', 'empty']):
            return self._get_missing_values_info(context)
        
        # Data types questions
        elif any(word in question_lower for word in ['types', 'dtype', 'data type']):
            return self._get_data_types_info(context)
        
        # Statistics questions
        elif any(word in question_lower for word in ['statistics', 'stats', 'mean', 'average', 'median']):
            return self._get_statistics_info(context)
        
        # Correlation questions
        elif any(word in question_lower for word in ['correlation', 'relationship', 'correlated']):
            return self._get_correlation_info(context)
        
        # Column-specific questions
        elif any(word in question_lower for word in ['column', 'feature', 'variable']):
            return self._get_column_info(context)
        
        # Default response
        else:
            return self._get_general_help(context)
    
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
    
    def _get_general_help(self, context: Dict[str, Any]) -> str:
        """
        Provide general help
        """
        return """🤖 **How I can help you analyze your data:**

I can answer questions about:
- Dataset overview and structure
- Missing values and data quality
- Column information and data types  
- Basic statistics and summaries
- Recommendations for analysis

**Example questions you can ask:**
- "What are the main characteristics of this dataset?"
- "Which columns have missing values?"
- "What data types are in my dataset?"
- "Can you summarize the key features?"

Feel free to ask me anything about your data!"""
