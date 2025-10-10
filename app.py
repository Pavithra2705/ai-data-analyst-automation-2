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
from advanced_stats import AdvancedStatistics
from transformation_pipeline import TransformationPipeline
from pdf_report import PDFReportGenerator
from data_versioning import DataVersioning
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
if 'transformation_pipeline' not in st.session_state:
    st.session_state.transformation_pipeline = TransformationPipeline()
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None
if 'data_versioning' not in st.session_state:
    st.session_state.data_versioning = DataVersioning()

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📤 Upload", "🧹 Clean", "📈 Insights", "📊 Visualize", "🔬 Advanced Stats", "🔄 Transform", "💬 Chat"])
    
    with tab1:
        upload_tab()
    
    with tab2:
        cleaning_tab()
    
    with tab3:
        insights_tab()
    
    with tab4:
        visualization_tab()
    
    with tab5:
        advanced_stats_tab()
    
    with tab6:
        transformation_pipeline_tab()
    
    with tab7:
        chat_tab()

def upload_tab():
    st.header("📤 Upload Your Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a data file",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
        help="Supported formats: CSV, Excel, JSON, Parquet (Maximum file size: 100MB)",
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
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        df = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        df = pd.read_parquet(uploaded_file)
                    else:
                        st.error("Unsupported file format")
                        return
                except ImportError as ie:
                    if 'parquet' in str(ie).lower() or 'pyarrow' in str(ie).lower():
                        st.error("⚠️ Parquet format requires pyarrow package. Please ensure it's installed in your environment.")
                    else:
                        st.error(f"Missing dependency: {str(ie)}")
                    return
                except Exception as load_error:
                    st.error(f"Error loading file: {str(load_error)}")
                    return
                
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
            
            # Create version before cleaning
            versioning = st.session_state.data_versioning
            if versioning.get_version_count() == 0:
                versioning.create_version(df, "Original uploaded data", {"source": "upload"})
            
            # Perform cleaning
            cleaned_df, report = cleaner.clean_data(
                df,
                remove_duplicates=remove_duplicates,
                missing_strategy=handle_missing,
                detect_outliers=detect_outliers,
                normalize=normalize_data
            )
            
            # Create version after cleaning
            cleaning_desc = f"Cleaned data (duplicates:{remove_duplicates}, missing:{handle_missing}, outliers:{detect_outliers})"
            versioning.create_version(cleaned_df, cleaning_desc, {
                "remove_duplicates": remove_duplicates,
                "missing_strategy": handle_missing,
                "detect_outliers": detect_outliers,
                "normalize": normalize_data
            })
            
            st.session_state.cleaned_data = cleaned_df
            st.session_state.cleaning_report = report
            
            st.success("✅ Data cleaning completed and version saved!")
    
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = cleaned_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
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
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Parquet download
            parquet_buffer = io.BytesIO()
            cleaned_df.to_parquet(parquet_buffer, index=False)
            st.download_button(
                label="📥 Download Parquet",
                data=parquet_buffer.getvalue(),
                file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet",
                mime="application/octet-stream"
            )
        
        # Preview cleaned data
        st.subheader("👀 Cleaned Data Preview")
        st.dataframe(cleaned_df.head(10), use_container_width=True)
    
    # Version History
    versioning = st.session_state.data_versioning
    if versioning.get_version_count() > 0:
        st.markdown("---")
        st.subheader("🕐 Version History")
        
        history = versioning.get_version_history()
        
        # Display versions in a table
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Rollback
            version_ids = [v['version_id'] for v in history]
            selected_version = st.selectbox("Rollback to version:", version_ids, key="rollback_version")
            
            if st.button("⏮️ Rollback", type="secondary"):
                with st.spinner("Rolling back..."):
                    rolled_back_df, version_info = versioning.rollback(selected_version)
                    if rolled_back_df is not None:
                        st.session_state.cleaned_data = rolled_back_df
                        st.success(f"✅ Rolled back to version {selected_version}: {version_info['action']}")
                        st.rerun()
                    else:
                        st.error("Failed to rollback")
        
        with col2:
            # Compare versions
            if len(version_ids) >= 2:
                v1 = st.selectbox("Compare version 1:", version_ids, key="compare_v1")
                v2 = st.selectbox("Compare version 2:", version_ids, index=min(1, len(version_ids)-1), key="compare_v2")
                
                if st.button("🔄 Compare", type="secondary"):
                    comparison = versioning.compare_versions(v1, v2)
                    if comparison:
                        st.json(comparison)
        
        with col3:
            # Export history
            history_json = versioning.export_version_history()
            st.download_button(
                label="📥 Export History",
                data=history_json,
                file_name=f"version_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

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
        
        # Generate PDF Report
        st.markdown("---")
        st.subheader("📄 Generate Report")
        
        if st.button("📥 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_gen = PDFReportGenerator()
                    pdf_data = pdf_gen.generate_report(
                        df,
                        cleaning_report=st.session_state.cleaning_report,
                        insights=insights
                    )
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")

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
    
    fig = None  # Store the current figure for export
    
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
    
    # Export functionality
    if fig is not None:
        st.markdown("---")
        st.subheader("💾 Export Visualization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Export as PNG
            try:
                img_bytes = fig.to_image(format="png", engine="kaleido", width=1200, height=800)
                st.download_button(
                    label="📥 Download PNG",
                    data=img_bytes,
                    file_name=f"{viz_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"PNG export error: {str(e)}")
        
        with col2:
            # Export as SVG
            try:
                svg_bytes = fig.to_image(format="svg", engine="kaleido", width=1200, height=800)
                st.download_button(
                    label="📥 Download SVG",
                    data=svg_bytes,
                    file_name=f"{viz_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                    mime="image/svg+xml"
                )
            except Exception as e:
                st.error(f"SVG export error: {str(e)}")
        
        with col3:
            # Export as HTML (interactive)
            html_str = fig.to_html(include_plotlyjs='cdn')
            st.download_button(
                label="📥 Download HTML",
                data=html_str,
                file_name=f"{viz_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

def advanced_stats_tab():
    st.header("🔬 Advanced Statistical Analysis")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the Clean tab.")
        return
    
    df = st.session_state.cleaned_data
    adv_stats = AdvancedStatistics()
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Regression Analysis", "Hypothesis Testing", "Anomaly Detection", "Distribution Analysis"]
    )
    
    if analysis_type == "Regression Analysis":
        st.subheader("📊 Linear Regression Analysis")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            st.warning("Need at least 2 numerical columns for regression analysis.")
            return
        
        target_col = st.selectbox("Select Target Variable (Y)", numerical_cols)
        
        feature_options = [col for col in numerical_cols if col != target_col]
        feature_cols = st.multiselect("Select Feature Variables (X)", feature_options, default=feature_options[:3] if len(feature_options) >= 3 else feature_options)
        
        if st.button("Run Regression Analysis", type="primary"):
            with st.spinner("Analyzing..."):
                results = adv_stats.perform_regression_analysis(df, target_col, feature_cols)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("✅ Regression analysis complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R-squared", f"{results['r_squared']:.4f}")
                    with col2:
                        st.metric("RMSE", f"{results['rmse']:.4f}")
                    with col3:
                        st.metric("MAE", f"{results['mean_absolute_error']:.4f}")
                    
                    st.subheader("Feature Coefficients")
                    coef_df = pd.DataFrame(results['coefficients'])
                    st.dataframe(coef_df[['Feature', 'Coefficient']], use_container_width=True)
                    
                    # Interpretation
                    st.subheader("📝 Interpretation")
                    if results['r_squared'] > 0.7:
                        st.success(f"Strong model fit (R² = {results['r_squared']:.3f}). The model explains {results['r_squared']*100:.1f}% of the variance.")
                    elif results['r_squared'] > 0.4:
                        st.info(f"Moderate model fit (R² = {results['r_squared']:.3f}). The model explains {results['r_squared']*100:.1f}% of the variance.")
                    else:
                        st.warning(f"Weak model fit (R² = {results['r_squared']:.3f}). Consider adding more relevant features.")
    
    elif analysis_type == "Hypothesis Testing":
        st.subheader("🧪 Hypothesis Testing")
        
        all_cols = df.columns.tolist()
        
        col1 = st.selectbox("Select First Variable", all_cols)
        
        test_type = st.radio("Test Type", ["Single Variable Analysis", "Two Variable Comparison"])
        
        col2 = None
        if test_type == "Two Variable Comparison":
            remaining_cols = [c for c in all_cols if c != col1]
            col2 = st.selectbox("Select Second Variable", remaining_cols)
        
        if st.button("Run Hypothesis Test", type="primary"):
            with st.spinner("Testing..."):
                results = adv_stats.perform_hypothesis_tests(df, col1, col2)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("✅ Hypothesis tests complete!")
                    
                    for test_name, test_results in results.items():
                        with st.expander(f"📊 {test_results['test']}", expanded=True):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Test Statistic", f"{test_results['statistic']:.4f}")
                            with col_b:
                                st.metric("P-value", f"{test_results['p_value']:.4f}")
                            
                            if 'significant' in test_results:
                                if test_results['significant']:
                                    st.success(f"✅ {test_results['interpretation']} (p < 0.05)")
                                else:
                                    st.info(f"ℹ️ {test_results['interpretation']} (p ≥ 0.05)")
                            else:
                                st.info(test_results['interpretation'])
    
    elif analysis_type == "Anomaly Detection":
        st.subheader("🔍 Anomaly Detection")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) == 0:
            st.warning("No numerical columns available for anomaly detection.")
            return
        
        selected_cols = st.multiselect("Select Columns for Analysis", numerical_cols, default=numerical_cols[:3] if len(numerical_cols) >= 3 else numerical_cols)
        
        method = st.selectbox("Detection Method", ["Isolation Forest", "Statistical (Z-score)", "DBSCAN Clustering"])
        
        method_map = {
            "Isolation Forest": "isolation_forest",
            "Statistical (Z-score)": "statistical",
            "DBSCAN Clustering": "dbscan"
        }
        
        contamination = st.slider("Expected Contamination Rate", 0.01, 0.3, 0.1, 0.01) if method == "Isolation Forest" else 0.1
        
        if st.button("Detect Anomalies", type="primary"):
            with st.spinner("Detecting anomalies..."):
                results = adv_stats.detect_anomalies(df, selected_cols, method_map[method], contamination)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("✅ Anomaly detection complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Anomalies", results['total_anomalies'])
                    with col2:
                        st.metric("Anomaly Rate", f"{results['anomaly_percentage']:.2f}%")
                    
                    st.subheader("📋 Sample Anomalies")
                    if 'sample_anomalies' in results:
                        st.dataframe(pd.DataFrame(results['sample_anomalies']), use_container_width=True)
                    else:
                        st.info("No anomalies detected.")
                    
                    st.info(f"**Method**: {results['method']}")
    
    elif analysis_type == "Distribution Analysis":
        st.subheader("📈 Distribution Analysis")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) == 0:
            st.warning("No numerical columns available for distribution analysis.")
            return
        
        selected_col = st.selectbox("Select Column", numerical_cols)
        
        if st.button("Analyze Distribution", type="primary"):
            with st.spinner("Analyzing distribution..."):
                results = adv_stats.perform_distribution_analysis(df, selected_col)
                
                if 'error' in results:
                    st.error(results['error'])
                else:
                    st.success("✅ Distribution analysis complete!")
                    
                    # Key metrics
                    st.subheader("📊 Key Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{results['mean']:.2f}")
                        st.metric("Std Dev", f"{results['std']:.2f}")
                    with col2:
                        st.metric("Median", f"{results['median']:.2f}")
                        st.metric("IQR", f"{results['iqr']:.2f}")
                    with col3:
                        st.metric("Skewness", f"{results['skewness']:.2f}")
                        st.metric("Kurtosis", f"{results['kurtosis']:.2f}")
                    with col4:
                        st.metric("Min", f"{results['min']:.2f}")
                        st.metric("Max", f"{results['max']:.2f}")
                    
                    # Shape and behavior
                    st.subheader("📐 Distribution Characteristics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Shape**: {results['shape']}")
                    with col2:
                        st.info(f"**Tail Behavior**: {results['tail_behavior']}")
                    
                    # Normality
                    if 'normality_test' in results:
                        if results['normality_test']['is_normal']:
                            st.success(f"✅ Data is normally distributed (p = {results['normality_test']['p_value']:.4f})")
                        else:
                            st.warning(f"⚠️ Data is not normally distributed (p = {results['normality_test']['p_value']:.4f})")
                    
                    # Outliers
                    st.subheader("🔍 Outlier Information")
                    st.metric("Outlier Count", f"{results['outliers']['count']} ({results['outliers']['percentage']:.1f}%)")
                    st.info(f"Outlier bounds: [{results['outliers']['lower_bound']:.2f}, {results['outliers']['upper_bound']:.2f}]")

def transformation_pipeline_tab():
    st.header("🔄 Custom Data Transformation Pipeline")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the Clean tab.")
        return
    
    df = st.session_state.cleaned_data
    pipeline = st.session_state.transformation_pipeline
    
    # Pipeline builder
    st.subheader("🛠️ Build Your Pipeline")
    
    transform_type = st.selectbox(
        "Select Transformation Type",
        [
            "Filter Rows", "Select Columns", "Drop Columns", "Rename Column",
            "Create Column", "Fill Missing Values", "Replace Values", "Convert Data Type",
            "Sort Data", "Group & Aggregate", "Bin Column", "Normalize Column", "Extract DateTime"
        ]
    )
    
    # Parameters based on transformation type
    params = {}
    
    if transform_type == "Filter Rows":
        col1, col2 = st.columns(2)
        with col1:
            params['column'] = st.selectbox("Column", df.columns.tolist())
        with col2:
            params['operator'] = st.selectbox("Operator", ["equals", "not_equals", "greater_than", "less_than", "contains", "not_contains", "is_null", "not_null"])
        if params['operator'] not in ['is_null', 'not_null']:
            params['value'] = st.text_input("Value")
    
    elif transform_type == "Select Columns":
        params['columns'] = st.multiselect("Select Columns to Keep", df.columns.tolist())
    
    elif transform_type == "Drop Columns":
        params['columns'] = st.multiselect("Select Columns to Drop", df.columns.tolist())
    
    elif transform_type == "Rename Column":
        col1, col2 = st.columns(2)
        with col1:
            params['old_name'] = st.selectbox("Original Column", df.columns.tolist())
        with col2:
            params['new_name'] = st.text_input("New Name")
    
    elif transform_type == "Create Column":
        params['name'] = st.text_input("New Column Name")
        params['expression'] = st.text_input("Expression (e.g., 'col1 + col2' or column name)")
        st.caption("Example: 'price * 1.1' or 'revenue - cost'")
    
    elif transform_type == "Fill Missing Values":
        col1, col2 = st.columns(2)
        with col1:
            params['column'] = st.selectbox("Column", df.columns.tolist())
        with col2:
            params['method'] = st.selectbox("Method", ["value", "mean", "median", "mode", "forward_fill", "backward_fill"])
        if params['method'] == 'value':
            params['value'] = st.text_input("Fill Value")
    
    elif transform_type == "Replace Values":
        params['column'] = st.selectbox("Column", df.columns.tolist())
        col1, col2 = st.columns(2)
        with col1:
            params['old_value'] = st.text_input("Old Value")
        with col2:
            params['new_value'] = st.text_input("New Value")
    
    elif transform_type == "Convert Data Type":
        col1, col2 = st.columns(2)
        with col1:
            params['column'] = st.selectbox("Column", df.columns.tolist())
        with col2:
            params['type'] = st.selectbox("New Type", ["int", "float", "string", "datetime", "category"])
    
    elif transform_type == "Sort Data":
        params['columns'] = st.multiselect("Sort by Columns", df.columns.tolist())
        params['ascending'] = st.checkbox("Ascending", value=True)
    
    elif transform_type == "Group & Aggregate":
        col1, col2, col3 = st.columns(3)
        with col1:
            params['group_by'] = st.selectbox("Group By", df.columns.tolist())
        with col2:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            params['agg_column'] = st.selectbox("Aggregate Column", numerical_cols)
        with col3:
            params['agg_function'] = st.selectbox("Function", ["mean", "sum", "count", "min", "max"])
    
    elif transform_type == "Bin Column":
        params['column'] = st.selectbox("Column", df.select_dtypes(include=[np.number]).columns.tolist())
        params['bins'] = st.number_input("Number of Bins", min_value=2, max_value=20, value=5)
    
    elif transform_type == "Normalize Column":
        col1, col2 = st.columns(2)
        with col1:
            params['column'] = st.selectbox("Column", df.select_dtypes(include=[np.number]).columns.tolist())
        with col2:
            params['method'] = st.selectbox("Method", ["minmax", "zscore"])
    
    elif transform_type == "Extract DateTime":
        col1, col2 = st.columns(2)
        with col1:
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not date_cols:
                date_cols = df.columns.tolist()
            params['column'] = st.selectbox("DateTime Column", date_cols)
        with col2:
            params['component'] = st.selectbox("Extract", ["year", "month", "day", "dayofweek", "hour", "quarter"])
    
    # Add transformation to pipeline
    if st.button("➕ Add to Pipeline", type="primary"):
        transform_type_map = {
            "Filter Rows": "filter_rows",
            "Select Columns": "select_columns",
            "Drop Columns": "drop_columns",
            "Rename Column": "rename_column",
            "Create Column": "create_column",
            "Fill Missing Values": "fill_missing",
            "Replace Values": "replace_values",
            "Convert Data Type": "convert_type",
            "Sort Data": "sort_data",
            "Group & Aggregate": "group_aggregate",
            "Bin Column": "bin_column",
            "Normalize Column": "normalize_column",
            "Extract DateTime": "extract_datetime"
        }
        
        pipeline.add_transformation(transform_type_map[transform_type], params)
        st.success(f"✅ Added '{transform_type}' to pipeline")
        st.rerun()
    
    # Display current pipeline
    st.markdown("---")
    st.subheader("📋 Current Pipeline")
    
    summary = pipeline.get_pipeline_summary()
    
    if summary['total_steps'] == 0:
        st.info("No transformations added yet. Add your first transformation above!")
    else:
        st.write(f"**Total Steps:** {summary['total_steps']}")
        
        for step in summary['transformations']:
            with st.expander(f"Step {step['step']}: {step['type']}", expanded=False):
                st.json(step['params'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Pipeline"):
                pipeline.clear_pipeline()
                st.success("Pipeline cleared!")
                st.rerun()
        
        with col2:
            pipeline_json = pipeline.export_pipeline()
            st.download_button(
                label="📥 Download Pipeline",
                data=pipeline_json,
                file_name=f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Execute pipeline
    st.markdown("---")
    st.subheader("▶️ Execute Pipeline")
    
    if summary['total_steps'] > 0:
        if st.button("🚀 Run Pipeline", type="primary"):
            with st.spinner("Executing transformations..."):
                try:
                    transformed_df, execution_log = pipeline.apply_pipeline(df)
                    st.session_state.transformed_data = transformed_df
                    
                    st.success(f"✅ Pipeline executed successfully!")
                    
                    # Show execution log
                    st.subheader("📊 Execution Log")
                    for log_entry in execution_log:
                        if log_entry['status'] == 'success':
                            st.success(f"Step {log_entry['step']} ({log_entry['type']}): {log_entry['before_shape']} → {log_entry['after_shape']}")
                        else:
                            st.error(f"Step {log_entry['step']} ({log_entry['type']}): {log_entry.get('error', 'Unknown error')}")
                    
                    # Show transformed data preview
                    st.subheader("👀 Transformed Data Preview")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rows", transformed_df.shape[0])
                    with col2:
                        st.metric("Columns", transformed_df.shape[1])
                    
                    st.dataframe(transformed_df.head(10), use_container_width=True)
                    
                    # Download transformed data
                    st.subheader("💾 Download Transformed Data")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv_data = transformed_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                            transformed_df.to_excel(writer, index=False)
                        st.download_button(
                            label="📥 Download Excel",
                            data=buffer.getvalue(),
                            file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                except Exception as e:
                    st.error(f"Pipeline execution failed: {str(e)}")

def chat_tab():
    st.header("💬 Chat with Your Data")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please clean your data first in the Clean tab.")
        return
    
    # Initialize chatbot
    if st.session_state.chatbot is None:
        with st.spinner("Loading AI assistant..."):
            try:
                st.session_state.chatbot = load_chatbot()
                st.success("✅ AI assistant ready!")
            except Exception as e:
                st.error(f"Failed to load chatbot: {str(e)}")
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
