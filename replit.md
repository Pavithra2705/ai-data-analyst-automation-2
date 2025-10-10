# AutoDataAnalyst

## Overview

AutoDataAnalyst is a comprehensive Python-based web application for automated data analysis, cleaning, visualization, and insights generation. Built with Streamlit as the frontend framework, it provides an interactive interface for users to upload datasets (CSV/Excel), automatically clean and preprocess data, generate statistical insights, create visualizations, and interact with an AI-powered chatbot for dataset exploration. The application focuses on making data analysis accessible through automation while providing advanced statistical capabilities and PDF reporting.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web interface with wide layout configuration
- **State Management**: Streamlit session state for persisting data across user interactions (uploaded_data, cleaned_data, insights, chatbot instances)
- **UI Components**: Organized into sidebar for file upload/controls and main content area for results display
- **Visualization Engine**: Plotly for interactive charts (histograms, scatter plots, correlation heatmaps) with custom color palettes

### Backend Architecture
- **Modular Design**: Separated into distinct Python modules for specific functionality:
  - `data_cleaner.py`: Data preprocessing pipeline with configurable strategies
  - `insights_generator.py`: Statistical analysis and insight extraction
  - `advanced_stats.py`: Regression analysis and advanced statistical methods
  - `visualizer.py`: Chart generation and visual analytics
  - `chatbot.py`: Rule-based AI assistant for dataset Q&A
  - `transformation_pipeline.py`: Sequential data transformation execution
  - `pdf_report.py`: PDF generation for comprehensive reports
  - `data_versioning.py`: Dataset version control and rollback capabilities
  - `utils.py`: File handling utilities and helper functions

- **Data Processing Flow**:
  1. File upload → validation and parsing (CSV/Excel with multiple encoding support)
  2. Automatic data cleaning (type inference, duplicate removal, missing value handling, outlier detection)
  3. Insight generation (statistical summaries, correlations, distributions, recommendations)
  4. Visualization creation (interactive Plotly charts)
  5. AI chatbot context preparation for user queries

- **AI/ML Components**:
  - Scikit-learn for preprocessing (StandardScaler, LabelEncoder, SimpleImputer)
  - Statistical analysis via SciPy and NumPy
  - Outlier detection using IsolationForest and DBSCAN
  - Regression analysis (Linear and Logistic)
  - Feature importance and mutual information analysis
  - Rule-based chatbot with enhanced pattern matching (no external LLM dependency)

- **Data Quality Pipeline**:
  - Automatic data type inference and conversion
  - Configurable missing value strategies (Auto, Mean/Median, Forward/Backward fill, Drop)
  - IQR-based outlier detection with capping/removal options
  - Optional data normalization using StandardScaler

- **Reporting System**:
  - ReportLab-based PDF generation with custom styling
  - Comprehensive reports including dataset overview, cleaning logs, statistical summaries, and insights
  - Multi-format export (CSV, Excel, PDF)

### Data Storage Solutions
- **In-Memory Storage**: Primary data storage using Pandas DataFrames in Streamlit session state
- **Versioning System**: Pickle-based serialization for dataset versioning with MD5 hash integrity checks
- **Temporary File Storage**: OS temporary directory for intermediate file operations and downloads
- **No Persistent Database**: Application operates entirely in-memory; data persistence only through downloads and versioning within session

### Authentication and Authorization
- **No Authentication**: Application is designed as a standalone tool without user authentication
- **Session Isolation**: Each user session maintains isolated state through Streamlit's session management
- **File Size Limits**: Implicit constraints through file upload limitations (intended for datasets up to 20MB based on requirements)

### Design Patterns
- **Pipeline Pattern**: TransformationPipeline implements sequential data transformations with execution logging
- **Strategy Pattern**: Configurable cleaning strategies for missing values and outlier handling
- **Factory Pattern**: Multiple file format handlers (CSV with encoding fallback, Excel)
- **Observer Pattern**: Streamlit's reactive model for UI updates based on state changes
- **Command Pattern**: Versioning system with action descriptions and rollback capabilities

## External Dependencies

### Python Libraries
- **Web Framework**: Streamlit (UI and session management)
- **Data Processing**: Pandas (primary data structure), NumPy (numerical operations)
- **Visualization**: Plotly Express/Graph Objects (interactive charts), Seaborn/Matplotlib (additional plotting)
- **Statistical Analysis**: SciPy (statistical functions), Scikit-learn (ML preprocessing and algorithms)
- **PDF Generation**: ReportLab (report creation with custom styling)
- **File Operations**: Built-in Python libraries (io, tempfile, pickle, hashlib, json, base64)

### Third-Party Services
- **None**: Application operates entirely offline without external API dependencies
- **AI/Chatbot**: Rule-based system using pattern matching (no external LLM service)

### Data Format Support
- **Input**: CSV (UTF-8, Latin-1, CP1252 encoding support), Excel (.xlsx, .xls)
- **Output**: CSV, Excel, PDF reports, JSON (cleaning reports and insights)

### Machine Learning Models
- **Preprocessing**: StandardScaler, LabelEncoder, SimpleImputer (Scikit-learn)
- **Anomaly Detection**: IsolationForest, DBSCAN (Scikit-learn)
- **Regression**: LinearRegression, LogisticRegression (Scikit-learn)
- **Feature Analysis**: Mutual information (regression and classification variants)

### No External Database
The application does not use PostgreSQL, MySQL, MongoDB, or any external database system. All data operations occur in-memory with optional versioning through pickle serialization.