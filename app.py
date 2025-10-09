import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import tempfile
import os
import json

from data_cleaner import DataCleaner
from visualizer import DataVisualizer
from chatbot import LLaMAChat
from insights_generator import InsightsGenerator
from utils import FileHandler, download_button_with_data

# Page config
st.set_page_config(
    page_title="AutoDataAnalyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'cleaning_report' not in st.session_state:
    st.session_state.cleaning_report = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

# Initialize components
@st.cache_resource
def load_chatbot():
    return LLaMAChat()

def main():
    st.title("🤖 AutoDataAnalyst")
    st.markdown("### AI-Powered Data Cleaning, Insights, and Visualization Platform")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("---")
        
        if st.session_state.uploaded_data is not None:
            st.success(f"✅ Data loaded: {st.session_state.uploaded_data.shape[0]} rows, {st.session_state.uploaded_data.shape[1]} columns")
        
        if st.session_state.cleaned_data is not None:
            st.success("✅ Data cleaned successfully")
        
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("- Upload CSV or XLSX files up to 100MB")
        st.markdown("- Review cleaning report before proceeding")
        st.markdown("- Use the chat feature to ask questions about your data")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Upload", "🧹 Clean", "📈 Insights", "📊 Visualize", "💬 Chat"])
    
    with tab1:
        upload_tab()
    
    with tab2:
        cleaning_tab()
    
    with tab3:
        insights_tab()
    
    with tab4:
        visualization_tab()
    
    with tab5:
        chat_tab()

def upload_tab():
    st.header("📤 Upload Your Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV or XLSX file",
        type=['csv', 'xlsx', 'xls'],
        help="Maximum file size: 100MB",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size = len(uploaded_file.getvalue())
            if file_size > 100 * 1024 * 1024:  # 100MB
                st.error("File size exceeds 100MB limit. Please upload a smaller file.")
                return
            
            st.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            # Load data
            with st.spinner("Loading data..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.uploaded_data = df
                st.success(f"✅ Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns!")
            
            # Display data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Column information
            st.subheader("📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def cleaning_tab():
    st.header("🧹 Data Cleaning")
    
    if st.session_state.uploaded_data is None:
        st.warning("Please upload a dataset first in the Upload tab.")
        return
    
    df = st.session_state.uploaded_data.copy()
    
    # Cleaning options
    st.subheader("🔧 Cleaning Options")
    
    col1, col2 = st.columns(2)
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate rows", value=True)
        handle_missing = st.selectbox(
            "Handle missing values",
            ["Auto", "Drop rows", "Drop columns", "Fill with mean/mode", "Forward fill", "Backward fill"]
        )
    
    with col2:
        detect_outliers = st.checkbox("Detect and handle outliers", value=True)
        normalize_data = st.checkbox("Normalize numerical columns", value=False)
    
    if st.button("🚀 Start Cleaning", type="primary"):
        with st.spinner("Cleaning data..."):
            cleaner = DataCleaner()
            
            # Perform cleaning
            cleaned_df, report = cleaner.clean_data(
                df,
                remove_duplicates=remove_duplicates,
                missing_strategy=handle_missing,
                detect_outliers=detect_outliers,
                normalize=normalize_data
            )
            
            st.session_state.cleaned_data = cleaned_df
            st.session_state.cleaning_report = report
            
            st.success("✅ Data cleaning completed!")
    
    # Show cleaning results
    if st.session_state.cleaned_data is not None and st.session_state.cleaning_report is not None:
        st.subheader("📋 Cleaning Results")
        
        # Before/After comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Cleaning:**")
            st.metric("Rows", df.shape[0])
            st.metric("Columns", df.shape[1])
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Duplicates", df.duplicated().sum())
        
        with col2:
            st.markdown("**After Cleaning:**")
            cleaned_df = st.session_state.cleaned_data
            st.metric("Rows", cleaned_df.shape[0])
            st.metric("Columns", cleaned_df.shape[1])
            st.metric("Missing Values", cleaned_df.isnull().sum().sum())
            st.metric("Duplicates", cleaned_df.duplicated().sum())
        
        # Cleaning report
        st.subheader("📊 Detailed Cleaning Report")
        report = st.session_state.cleaning_report
        
        for step, details in report.items():
            with st.expander(f"🔍 {step.replace('_', ' ').title()}"):
                if isinstance(details, dict):
                    st.json(details)
                else:
                    st.write(details)
        
        # Download cleaned data
        st.subheader("💾 Download Cleaned Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = cleaned_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv_data,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                cleaned_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
            excel_data = buffer.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Preview cleaned data
        st.subheader("👀 Cleaned Data Preview")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

def insights_tab():
    st.header("📈 Statistical Insights")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the Clean tab.")
        return
    
    df = st.session_state.cleaned_data
    
    if st.button("🔍 Generate Insights", type="primary"):
        with st.spinner("Generating insights..."):
            insights_gen = InsightsGenerator()
            insights = insights_gen.generate_insights(df)
            st.session_state.insights = insights
            st.success("✅ Insights generated!")
    
    if st.session_state.insights is not None:
        insights = st.session_state.insights
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        # Numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.markdown("**Numerical Columns:**")
            st.dataframe(df[numerical_cols].describe(), use_container_width=True)
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.markdown("**Categorical Columns:**")
            cat_stats = pd.DataFrame({
                'Column': categorical_cols,
                'Unique Values': [df[col].nunique() for col in categorical_cols],
                'Most Frequent': [df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A' for col in categorical_cols],
                'Frequency': [df[col].value_counts().iloc[0] if not df[col].empty else 0 for col in categorical_cols]
            })
            st.dataframe(cat_stats, use_container_width=True)
        
        # Correlation analysis
        if len(numerical_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.markdown("**Strong Correlations (|r| > 0.7):**")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': round(corr_val, 3)
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
            else:
                st.info("No strong correlations found (|r| > 0.7)")
        
        # Key insights
        st.subheader("💡 Key Insights")
        for insight_type, insight_data in insights.items():
            with st.expander(f"🔍 {insight_type.replace('_', ' ').title()}"):
                if isinstance(insight_data, dict):
                    for key, value in insight_data.items():
                        st.write(f"**{key}:** {value}")
                else:
                    st.write(insight_data)

def visualization_tab():
    st.header("📊 Data Visualizations")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the Clean tab.")
        return
    
    df = st.session_state.cleaned_data
    visualizer = DataVisualizer(df)
    
    # Visualization options
    st.subheader("🎨 Create Visualizations")
    
    viz_type = st.selectbox(
        "Select visualization type",
        ["Distribution Plot", "Scatter Plot", "Box Plot", "Bar Chart", "Time Series", "Correlation Heatmap"]
    )
    
    if viz_type == "Distribution Plot":
        col = st.selectbox("Select column", df.select_dtypes(include=[np.number]).columns)
        if col:
            fig = visualizer.create_histogram(col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Scatter Plot":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", numerical_cols)
            with col2:
                y_col = st.selectbox("Y-axis", numerical_cols)
            
            if x_col and y_col:
                fig = visualizer.create_scatter_plot(x_col, y_col)
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        col = st.selectbox("Select column", df.select_dtypes(include=[np.number]).columns)
        if col:
            fig = visualizer.create_box_plot(col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart":
        col = st.selectbox("Select column", df.select_dtypes(include=['object', 'category']).columns)
        if col:
            fig = visualizer.create_bar_chart(col)
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time Series":
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Try to find columns that might be dates
            potential_date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if potential_date_cols:
                st.info("No datetime columns found. Try converting these columns to datetime first:")
                st.write(potential_date_cols)
            else:
                st.warning("No datetime columns found in the dataset.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Date column", date_cols)
            with col2:
                value_col = st.selectbox("Value column", df.select_dtypes(include=[np.number]).columns)
            
            if date_col and value_col:
                fig = visualizer.create_time_series(date_col, value_col)
                st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Correlation Heatmap":
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            fig = visualizer.create_correlation_heatmap()
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numerical columns for correlation heatmap.")

def chat_tab():
    st.header("💬 Chat with Your Data")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the Clean tab.")
        return
    
    # Initialize chatbot
    if st.session_state.chatbot is None:
        with st.spinner("Loading AI chatbot... This may take a moment."):
            try:
                st.session_state.chatbot = load_chatbot()
                st.success("✅ Chatbot loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load chatbot: {str(e)}")
                st.info("The chatbot feature requires additional model downloads. You can still use other features of the application.")
                return
    
    df = st.session_state.cleaned_data
    chatbot = st.session_state.chatbot
    
    # Data context for chatbot
    context = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'missing_values': df.isnull().sum().to_dict(),
        'sample_data': df.head(3).to_dict()
    }
    
    # Chat interface
    st.subheader("🤖 Ask questions about your dataset")
    
    # Example questions
    st.markdown("**💡 Example questions you can ask:**")
    example_questions = [
        "What are the main characteristics of this dataset?",
        "Which columns have the most missing values?",
        "What are the strongest correlations in the data?",
        "Can you summarize the key insights from this data?",
        "What columns should I focus on for analysis?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(f"💭 {question}", key=f"example_{i}"):
            st.session_state.current_question = question
    
    # User input
    user_question = st.text_input("Your question:", key="user_question")
    
    if st.button("🚀 Ask", type="primary") and user_question:
        with st.spinner("Thinking..."):
            try:
                response = chatbot.generate_response(user_question, context)
                st.markdown("### 🤖 AI Response:")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Handle example question clicks
    if hasattr(st.session_state, 'current_question'):
        with st.spinner("Thinking..."):
            try:
                response = chatbot.generate_response(st.session_state.current_question, context)
                st.markdown("### 🤖 AI Response:")
                st.markdown(response)
                del st.session_state.current_question
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
