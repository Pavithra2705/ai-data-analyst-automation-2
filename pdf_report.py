import io
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import pandas as pd
import numpy as np

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.buffer = io.BytesIO()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#ff7f0e'),
            spaceAfter=10,
            spaceBefore=10
        )
    
    def generate_report(self, df, cleaning_report=None, insights=None):
        """
        Generate comprehensive PDF report
        """
        doc = SimpleDocTemplate(
            self.buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title
        elements.append(Paragraph("AutoDataAnalyst Report", self.title_style))
        elements.append(Spacer(1, 12))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y %I:%M %p")
        elements.append(Paragraph(f"<b>Generated:</b> {report_date}", self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Dataset Overview
        elements.append(Paragraph("Dataset Overview", self.heading_style))
        overview_data = [
            ['Metric', 'Value'],
            ['Total Rows', f"{df.shape[0]:,}"],
            ['Total Columns', f"{df.shape[1]:,}"],
            ['Missing Values', f"{df.isnull().sum().sum():,}"],
            ['Duplicates', f"{df.duplicated().sum():,}"],
            ['Memory Usage', f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"]
        ]
        
        overview_table = Table(overview_data, colWidths=[3*inch, 3*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(overview_table)
        elements.append(Spacer(1, 20))
        
        # Column Information
        elements.append(Paragraph("Column Information", self.heading_style))
        col_data = [['Column', 'Type', 'Non-Null', 'Null %']]
        
        for col in df.columns[:15]:  # Limit to first 15 columns
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_pct = (df[col].isnull().sum() / len(df) * 100)
            col_data.append([col[:30], dtype, f"{non_null:,}", f"{null_pct:.1f}%"])
        
        if len(df.columns) > 15:
            col_data.append(['...', '...', '...', '...'])
        
        col_table = Table(col_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 1*inch])
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(col_table)
        elements.append(Spacer(1, 20))
        
        # Cleaning Report (if provided)
        if cleaning_report:
            elements.append(PageBreak())
            elements.append(Paragraph("Data Cleaning Report", self.heading_style))
            
            if 'summary' in cleaning_report:
                summary = cleaning_report['summary']
                elements.append(Paragraph(f"<b>Original Shape:</b> {summary['original_shape']}", self.styles['Normal']))
                elements.append(Paragraph(f"<b>Final Shape:</b> {summary['final_shape']}", self.styles['Normal']))
                elements.append(Paragraph(f"<b>Rows Removed:</b> {summary['rows_removed']}", self.styles['Normal']))
                elements.append(Spacer(1, 12))
            
            # Cleaning steps
            for step, details in cleaning_report.items():
                if step != 'summary':
                    elements.append(Paragraph(step.replace('_', ' ').title(), self.subheading_style))
                    if isinstance(details, dict):
                        for key, value in list(details.items())[:5]:  # Limit details
                            elements.append(Paragraph(f"• {key}: {value}", self.styles['Normal']))
                    elements.append(Spacer(1, 8))
        
        # Statistical Summary
        elements.append(PageBreak())
        elements.append(Paragraph("Statistical Summary", self.heading_style))
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10
        if len(numerical_cols) > 0:
            stats_data = [['Column', 'Mean', 'Std', 'Min', 'Max']]
            
            for col in numerical_cols:
                stats_data.append([
                    col[:25],
                    f"{df[col].mean():.2f}",
                    f"{df[col].std():.2f}",
                    f"{df[col].min():.2f}",
                    f"{df[col].max():.2f}"
                ])
            
            stats_table = Table(stats_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(stats_table)
        else:
            elements.append(Paragraph("No numerical columns available for statistical summary.", self.styles['Normal']))
        
        elements.append(Spacer(1, 20))
        
        # Insights (if provided)
        if insights:
            elements.append(PageBreak())
            elements.append(Paragraph("Data Insights", self.heading_style))
            
            # Data Quality
            if 'data_quality' in insights:
                elements.append(Paragraph("Data Quality Issues", self.subheading_style))
                dq = insights['data_quality']
                
                if 'columns_with_missing' in dq and dq['columns_with_missing']:
                    elements.append(Paragraph(f"<b>Columns with Missing Values:</b> {len(dq['columns_with_missing'])}", self.styles['Normal']))
                
                if 'duplicate_rows' in dq:
                    elements.append(Paragraph(f"<b>Duplicate Rows:</b> {dq['duplicate_rows']}", self.styles['Normal']))
                
                elements.append(Spacer(1, 12))
            
            # Correlations
            if 'correlations' in insights and 'strong_correlations' in insights['correlations']:
                strong_corr = insights['correlations']['strong_correlations']
                if strong_corr:
                    elements.append(Paragraph("Strong Correlations", self.subheading_style))
                    for corr in strong_corr[:5]:  # Limit to 5
                        elements.append(Paragraph(
                            f"• {corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']:.3f}",
                            self.styles['Normal']
                        ))
                    elements.append(Spacer(1, 12))
            
            # Recommendations
            if 'recommendations' in insights:
                elements.append(Paragraph("Recommendations", self.subheading_style))
                for rec in insights['recommendations'][:5]:  # Limit to 5
                    priority_color = {
                        'high': colors.red,
                        'medium': colors.orange,
                        'low': colors.green
                    }.get(rec.get('priority', 'low'), colors.black)
                    
                    elements.append(Paragraph(
                        f"• [{rec.get('priority', 'low').upper()}] {rec['message'][:150]}",
                        self.styles['Normal']
                    ))
                elements.append(Spacer(1, 12))
        
        # Data Sample
        elements.append(PageBreak())
        elements.append(Paragraph("Data Sample", self.heading_style))
        
        # Create sample table (first 10 rows, first 8 columns)
        sample_cols = df.columns[:8].tolist()
        sample_data = [sample_cols]
        
        for idx, row in df.head(10).iterrows():
            row_data = []
            for col in sample_cols:
                val = str(row[col])[:20]  # Truncate long values
                row_data.append(val)
            sample_data.append(row_data)
        
        col_width = 6.5 * inch / len(sample_cols)
        sample_table = Table(sample_data, colWidths=[col_width] * len(sample_cols))
        sample_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(sample_table)
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF data
        pdf_data = self.buffer.getvalue()
        self.buffer.close()
        
        return pdf_data
