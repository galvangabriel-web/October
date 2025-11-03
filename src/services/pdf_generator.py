"""
PDF Report Generator - Sprint 3 Task 1
=======================================

Generate professional PDF reports from Model Predictions analysis.
Includes patterns, corner analysis, track intelligence, and charts.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RacingReportGenerator:
    """Generate professional PDF reports from racing telemetry analysis"""

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Setup custom paragraph styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5,
            backColor=colors.HexColor('#ecf0f1')
        ))

        # Subsection style
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#2980b9'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))

        # Pattern name style
        self.styles.add(ParagraphStyle(
            name='PatternName',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#2c3e50'),
            fontName='Helvetica-Bold',
            spaceAfter=6
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='ReportBody',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=6,
            leading=14
        ))

        # Footer style
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_CENTER
        ))

    def generate_report(
        self,
        vehicle_number: int,
        track_name: str,
        patterns_data: Optional[List[Dict]] = None,
        corner_analyses: Optional[List[Dict]] = None,
        features_data: Optional[Dict] = None,
        prediction_result: Optional[Dict] = None
    ) -> bytes:
        """
        Generate PDF report from analysis data

        Args:
            vehicle_number: Vehicle number
            track_name: Track name (e.g., 'circuit-of-the-americas')
            patterns_data: List of detected patterns
            corner_analyses: List of corner analysis results
            features_data: Feature extraction results
            prediction_result: Prediction results (optional)

        Returns:
            bytes: PDF content ready for download
        """
        logger.info(f"Generating PDF report for Vehicle #{vehicle_number} at {track_name}")

        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build report content
        story = []

        # Title page
        story.extend(self._create_title_page(vehicle_number, track_name, features_data))

        # Summary section
        if features_data:
            story.extend(self._create_summary_section(features_data))

        # Patterns section
        if patterns_data and len(patterns_data) > 0:
            story.extend(self._create_patterns_section(patterns_data))

        # Corner analysis section
        if corner_analyses and len(corner_analyses) > 0:
            story.extend(self._create_corners_section(corner_analyses))

        # Footer
        story.extend(self._create_footer())

        # Build PDF
        doc.build(story)

        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()

        logger.info(f"PDF report generated successfully ({len(pdf_bytes)} bytes)")
        return pdf_bytes

    def _create_title_page(
        self,
        vehicle_number: int,
        track_name: str,
        features_data: Optional[Dict]
    ) -> List:
        """Create title page elements"""
        elements = []

        # Title
        title = Paragraph(
            f"Racing Telemetry Analysis Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))

        # Vehicle and track info
        track_display = track_name.replace('-', ' ').title()
        info_data = [
            ['Vehicle Number:', f'#{vehicle_number}'],
            ['Track:', track_display],
            ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ]

        if features_data:
            info_data.append(['Total Features Analyzed:', str(features_data.get('total_features', 'N/A'))])
            info_data.append(['Laps Analyzed:', str(features_data.get('total_laps', 'N/A'))])

        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 11),
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#34495e')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))

        elements.append(info_table)
        elements.append(Spacer(1, 0.5*inch))

        return elements

    def _create_summary_section(self, features_data: Dict) -> List:
        """Create summary statistics section"""
        elements = []

        # Section header
        header = Paragraph("Analysis Summary", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.2*inch))

        # Summary statistics table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Features', str(features_data.get('total_features', 'N/A'))],
            ['Feature Categories', str(features_data.get('total_categories', 'N/A'))],
            ['Laps Analyzed', str(features_data.get('total_laps', 'N/A'))],
            ['Vehicles in Dataset', str(features_data.get('total_vehicles', 'N/A'))],
            ['Data Quality', f"{features_data.get('data_quality', 100)}%"],
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 2.5*inch])
        summary_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 12),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Data rows
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#34495e')),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica-Bold'),
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
        ]))

        elements.append(summary_table)
        elements.append(Spacer(1, 0.4*inch))

        return elements

    def _create_patterns_section(self, patterns_data: List[Dict]) -> List:
        """Create detected patterns section"""
        elements = []

        # Section header
        header = Paragraph("Detected Driving Patterns", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.15*inch))

        # Total savings summary
        total_savings = sum(p.get('impact_seconds', 0) for p in patterns_data)
        savings_text = Paragraph(
            f"<b>Total Potential Lap Time Savings: {total_savings:.3f} seconds</b>",
            self.styles['ReportBody']
        )
        elements.append(savings_text)
        elements.append(Spacer(1, 0.2*inch))

        # Each pattern
        for i, pattern in enumerate(patterns_data, 1):
            pattern_elements = self._create_pattern_card(i, pattern)
            elements.extend(pattern_elements)

        return elements

    def _create_pattern_card(self, rank: int, pattern: Dict) -> List:
        """Create individual pattern card"""
        elements = []

        # Pattern header
        name = pattern.get('pattern_name', 'Unknown Pattern')
        severity = pattern.get('severity', 'MEDIUM')
        impact = pattern.get('impact_seconds', 0)

        # Color based on severity
        severity_colors = {
            'HIGH': colors.HexColor('#dc3545'),
            'MEDIUM': colors.HexColor('#ffc107'),
            'LOW': colors.HexColor('#17a2b8')
        }
        severity_color = severity_colors.get(severity, colors.HexColor('#6c757d'))

        # Pattern title
        title_text = f"#{rank}: {name} [{severity}] - {impact:.3f}s per lap"
        title = Paragraph(title_text, self.styles['PatternName'])

        # Pattern details table
        # Note: Patterns from TelemetryAnalyzer use 'what_metrics', 'where_corners', 'when_laps', 'coaching'
        what_text = pattern.get('what_metrics', pattern.get('what_description', 'N/A'))
        where_text = pattern.get('where_corners', pattern.get('where_description', 'N/A'))
        when_text = pattern.get('when_laps', pattern.get('when_description', 'N/A'))
        coaching_text = pattern.get('coaching', pattern.get('coaching_recommendation', 'N/A'))

        # Convert list to string if needed
        if isinstance(what_text, list):
            what_text = ', '.join(str(x) for x in what_text) if what_text else 'N/A'
        if isinstance(where_text, list):
            where_text = ', '.join(str(x) for x in where_text) if where_text else 'N/A'
        if isinstance(when_text, list):
            when_text = ', '.join(str(x) for x in when_text) if when_text else 'N/A'

        details_data = [
            ['<b>What:</b>', str(what_text)],
            ['<b>Where:</b>', str(where_text)],
            ['<b>When:</b>', str(when_text)],
            ['<b>Coaching:</b>', str(coaching_text)],
        ]

        details_table = Table(details_data, colWidths=[1*inch, 5*inch])
        details_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
            ('FONT', (1, 0), (1, -1), 'Helvetica', 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#34495e')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (0, -1), 10),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
            ('BOX', (0, 0), (-1, -1), 2, severity_color),
        ]))

        elements.append(KeepTogether([title, Spacer(1, 0.1*inch), details_table]))
        elements.append(Spacer(1, 0.25*inch))

        return elements

    def _create_corners_section(self, corner_analyses: List[Dict]) -> List:
        """Create corner analysis section"""
        elements = []

        # Section header
        header = Paragraph("Corner-by-Corner Analysis", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.15*inch))

        # Summary
        total_corners = len(corner_analyses)
        total_opportunity = sum(c.get('time_delta', 0) for c in corner_analyses)
        summary_text = Paragraph(
            f"<b>{total_corners} corners analyzed with {total_opportunity:.3f}s total opportunity</b>",
            self.styles['ReportBody']
        )
        elements.append(summary_text)
        elements.append(Spacer(1, 0.2*inch))

        # Top 5 corners table
        top_corners = sorted(corner_analyses, key=lambda x: x.get('time_delta', 0), reverse=True)[:5]

        corner_data = [['Rank', 'Corner', 'Time Delta', 'Entry Speed Gap', 'Apex Speed Gap']]

        for i, corner in enumerate(top_corners, 1):
            corner_name = corner.get('corner_name', f"Turn {corner.get('corner_number', i)}")[:25]
            time_delta = f"{corner.get('time_delta', 0):.3f}s"
            entry_gap = corner.get('entry_speed_best', 0) - corner.get('entry_speed_avg', 0)
            apex_gap = corner.get('apex_speed_best', 0) - corner.get('apex_speed_avg', 0)

            corner_data.append([
                str(i),
                corner_name,
                time_delta,
                f"+{entry_gap:.1f} km/h",
                f"+{apex_gap:.1f} km/h"
            ])

        corner_table = Table(corner_data, colWidths=[0.6*inch, 2.5*inch, 1*inch, 1.2*inch, 1.2*inch])
        corner_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 11),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Data
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#34495e')),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (-1, -1), 'CENTER'),
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(corner_table)
        elements.append(Spacer(1, 0.3*inch))

        # Individual corner details (top 3 only for space)
        elements.append(Paragraph("Top 3 Corner Details", self.styles['SubSection']))
        elements.append(Spacer(1, 0.1*inch))

        for i, corner in enumerate(top_corners[:3], 1):
            corner_elements = self._create_corner_detail(i, corner)
            elements.extend(corner_elements)

        return elements

    def _create_corner_detail(self, rank: int, corner: Dict) -> List:
        """Create detailed corner analysis card"""
        elements = []

        corner_name = corner.get('corner_name', f"Turn {corner.get('corner_number', rank)}")
        time_delta = corner.get('time_delta', 0)

        # Corner title
        title_text = f"#{rank}: {corner_name} - {time_delta:.3f}s opportunity"
        title = Paragraph(title_text, self.styles['PatternName'])
        elements.append(title)

        # Metrics table
        metrics_data = [
            ['Metric', 'Your Average', 'Your Best', 'Gap'],
            [
                'Entry Speed',
                f"{corner.get('entry_speed_avg', 0):.1f} km/h",
                f"{corner.get('entry_speed_best', 0):.1f} km/h",
                f"+{corner.get('entry_speed_best', 0) - corner.get('entry_speed_avg', 0):.1f}"
            ],
            [
                'Apex Speed',
                f"{corner.get('apex_speed_avg', 0):.1f} km/h",
                f"{corner.get('apex_speed_best', 0):.1f} km/h",
                f"+{corner.get('apex_speed_best', 0) - corner.get('apex_speed_avg', 0):.1f}"
            ],
            [
                'Brake Pressure',
                f"{corner.get('brake_pressure_avg', 0):.1f} bar",
                f"{corner.get('brake_pressure_max', 0):.1f} bar",
                f"+{corner.get('brake_pressure_max', 0) - corner.get('brake_pressure_avg', 0):.1f}"
            ]
        ]

        metrics_table = Table(metrics_data, colWidths=[1.5*inch, 1.4*inch, 1.4*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
            ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(metrics_table)

        # Opportunities
        opportunities = corner.get('opportunities', [])
        if opportunities:
            opp_text = "<b>Opportunities:</b><br/>" + "<br/>".join([f"• {opp}" for opp in opportunities[:3]])
            opp_para = Paragraph(opp_text, self.styles['ReportBody'])
            elements.append(Spacer(1, 0.1*inch))
            elements.append(opp_para)

        elements.append(Spacer(1, 0.2*inch))

        return elements

    def _create_footer(self) -> List:
        """Create report footer"""
        elements = []

        elements.append(Spacer(1, 0.5*inch))

        footer_text = f"""
        <para align="center">
        <b>Racing Telemetry Analysis Dashboard</b><br/>
        Generated with GR Cup Racing Analytics Platform<br/>
        © 2025 | Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </para>
        """

        footer = Paragraph(footer_text, self.styles['Footer'])
        elements.append(footer)

        return elements


# Test function
if __name__ == '__main__':
    """Test PDF generation with sample data"""
    print("Testing PDF Report Generator...")

    generator = RacingReportGenerator()

    # Sample data
    sample_patterns = [
        {
            'pattern_name': 'Underutilizing Brake Pressure',
            'severity': 'HIGH',
            'impact_seconds': 0.50,
            'what_description': 'Not applying maximum brake pressure in heavy braking zones',
            'where_description': 'Turn 1, Turn 6, Turn 15',
            'when_description': 'Entry phase of slow corners',
            'coaching_recommendation': 'Brake harder and earlier, focus on threshold braking technique'
        },
        {
            'pattern_name': 'Conservative Throttle Application',
            'severity': 'MEDIUM',
            'impact_seconds': 0.30,
            'what_description': 'Delayed throttle application on corner exit',
            'where_description': 'Turns 3, 11, 19',
            'when_description': 'Exit phase after apex',
            'coaching_recommendation': 'Get on throttle earlier, trust the car\'s traction'
        }
    ]

    sample_corners = [
        {
            'corner_number': 1,
            'corner_name': 'Turn 1 (Uphill Left)',
            'entry_speed_avg': 166.6,
            'entry_speed_best': 170.2,
            'apex_speed_avg': 77.3,
            'apex_speed_best': 79.8,
            'brake_pressure_avg': 143.8,
            'brake_pressure_max': 150.1,
            'time_delta': 0.21,
            'opportunities': [
                'Brake 6.3 bar harder for better entry',
                'Carry 2.5 km/h more speed through apex',
                'Entry speed can increase 3.6 km/h with later braking'
            ]
        }
    ]

    sample_features = {
        'total_features': 138,
        'total_categories': 10,
        'total_laps': 12,
        'total_vehicles': 1,
        'data_quality': 100
    }

    # Generate PDF
    pdf_bytes = generator.generate_report(
        vehicle_number=2,
        track_name='circuit-of-the-americas',
        patterns_data=sample_patterns,
        corner_analyses=sample_corners,
        features_data=sample_features
    )

    # Save test PDF
    with open('test_racing_report.pdf', 'wb') as f:
        f.write(pdf_bytes)

    print(f"[SUCCESS] Test PDF generated: test_racing_report.pdf ({len(pdf_bytes)} bytes)")
