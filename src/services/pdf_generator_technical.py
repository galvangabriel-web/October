"""
Technical Deep-Dive PDF Report Generator for Racing Telemetry Analysis.

This module generates comprehensive, technical-depth PDF reports with detailed metrics,
sensor data, and complete coaching text. Designed for engineers and data analysts.

Author: Racing Telemetry Analysis System
"""

from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Any
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas


class TechnicalPDFGenerator:
    """
    Generates technical deep-dive PDF reports for racing telemetry analysis.

    Features:
    - Professional typography with proper text wrapping
    - Detailed technical tables with sensor names and metrics
    - Complete coaching text (no truncation)
    - Methodology section explaining analysis approach
    - Glossary of technical terms
    - 4-6 page comprehensive format
    """

    def __init__(self):
        """Initialize the PDF generator with styles and configuration."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Page configuration
        self.page_width = letter[0]
        self.page_height = letter[1]
        self.margin = 0.75 * inch
        self.usable_width = self.page_width - (2 * self.margin)

    def _setup_custom_styles(self):
        """Create custom paragraph styles for technical reports."""

        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=6,
            backColor=colors.HexColor('#ecf0f1')
        ))

        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))

        # Body text style
        self.styles.add(ParagraphStyle(
            name='TechnicalBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))

        # Coaching text style (important - needs space)
        self.styles.add(ParagraphStyle(
            name='CoachingText',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
            fontName='Helvetica',
            textColor=colors.HexColor('#e74c3c'),
            leftIndent=10,
            rightIndent=10,
            spaceAfter=8
        ))

        # Metadata style
        self.styles.add(ParagraphStyle(
            name='Metadata',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d'),
            fontName='Helvetica'
        ))

        # Caption style
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#95a5a6'),
            fontName='Helvetica-Oblique',
            alignment=TA_CENTER
        ))

    def _create_header_footer(self, canvas_obj, doc):
        """
        Add header and footer to each page.

        Args:
            canvas_obj: ReportLab canvas object
            doc: Document template object
        """
        canvas_obj.saveState()

        # Header
        canvas_obj.setFont('Helvetica-Bold', 9)
        canvas_obj.setFillColor(colors.HexColor('#34495e'))
        canvas_obj.drawString(
            self.margin,
            self.page_height - 0.5 * inch,
            "Racing Telemetry Analysis - Technical Deep-Dive Report"
        )

        # Footer
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(colors.HexColor('#7f8c8d'))
        canvas_obj.drawString(
            self.margin,
            0.5 * inch,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        canvas_obj.drawRightString(
            self.page_width - self.margin,
            0.5 * inch,
            f"Page {doc.page}"
        )

        canvas_obj.restoreState()

    def generate_report(
        self,
        vehicle_number: int,
        track_name: str,
        patterns_data: List[Dict[str, Any]],
        corner_analysis_data: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate a complete technical deep-dive PDF report.

        Args:
            vehicle_number: Race vehicle number
            track_name: Name of the racing track
            patterns_data: List of detected driving patterns with metrics
            corner_analysis_data: Optional corner-by-corner analysis data
            metadata: Optional metadata (upload time, file info, etc.)

        Returns:
            bytes: PDF file as bytes

        Example:
            >>> generator = TechnicalPDFGenerator()
            >>> pdf_bytes = generator.generate_report(
            ...     vehicle_number=2,
            ...     track_name='circuit-of-the-americas',
            ...     patterns_data=[{
            ...         'pattern_name': 'Underutilizing Brake Pressure',
            ...         'severity': 'High',
            ...         'impact_seconds': 0.2,
            ...         'what_metrics': ['pbrake_f', 'pbrake_r'],
            ...         'where_corners': [],
            ...         'when_laps': [1,2,3,4,5,6,7,8,9,10],
            ...         'coaching': "Full coaching text here..."
            ...     }]
            ... )
        """
        buffer = BytesIO()

        # Create document with custom page template
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=1.0 * inch,
            bottomMargin=0.75 * inch,
            title=f"Technical Analysis - Vehicle {vehicle_number}"
        )

        # Build content
        story = []

        # Page 1: Title, Metadata, Methodology
        story.extend(self._create_title_section(vehicle_number, track_name))
        story.extend(self._create_metadata_section(metadata))
        story.append(Spacer(1, 0.25 * inch))
        story.extend(self._create_methodology_section())
        story.append(PageBreak())

        # Pages 2-4: Pattern Analysis
        story.extend(self._create_patterns_section(patterns_data))

        # Page 5: Corner Analysis (if available)
        if corner_analysis_data:
            story.append(PageBreak())
            story.extend(self._create_corner_analysis_section(corner_analysis_data))

        # Page 6: Appendix
        story.append(PageBreak())
        story.extend(self._create_appendix_section())

        # Build PDF with header/footer
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def _create_title_section(self, vehicle_number: int, track_name: str) -> List:
        """Create title section with vehicle and track info."""
        elements = []

        # Main title
        title = Paragraph(
            f"Technical Deep-Dive Analysis Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.1 * inch))

        # Subtitle
        subtitle = Paragraph(
            f"Vehicle #{vehicle_number} | {self._format_track_name(track_name)}",
            self.styles['Heading2']
        )
        elements.append(subtitle)
        elements.append(Spacer(1, 0.2 * inch))

        return elements

    def _create_metadata_section(self, metadata: Optional[Dict[str, Any]]) -> List:
        """Create metadata table with analysis details."""
        elements = []

        # Section header
        header = Paragraph("Analysis Metadata", self.styles['SectionHeader'])
        elements.append(header)

        # Build metadata table
        if metadata is None:
            metadata = {}

        table_data = [
            ['Property', 'Value'],
            ['Analysis Date', metadata.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))],
            ['Analysis Time', metadata.get('analysis_time', datetime.now().strftime('%H:%M:%S'))],
            ['Data Source', metadata.get('data_source', 'Telemetry Upload')],
            ['Total Laps Analyzed', str(metadata.get('total_laps', 'N/A'))],
            ['Data Points', str(metadata.get('data_points', 'N/A'))],
            ['Analysis Engine', 'Cube Analysis v2.0 + Pattern Detection'],
            ['Report Version', 'Technical Deep-Dive v1.0']
        ]

        # Create table with proper styling
        table = Table(
            table_data,
            colWidths=[2.2 * inch, 4.3 * inch],
            hAlign='LEFT'
        )

        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),

            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2c3e50')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1), True),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),

            # Padding
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))

        elements.append(table)

        return elements

    def _create_methodology_section(self) -> List:
        """Create methodology section explaining analysis approach."""
        elements = []

        # Section header
        header = Paragraph("Analysis Methodology", self.styles['SectionHeader'])
        elements.append(header)

        # Methodology text
        methodology_text = """
        This technical report employs multi-dimensional telemetry analysis using the Cube Analysis Engine
        combined with advanced pattern detection algorithms. The analysis evaluates driver performance across
        three critical dimensions:
        """
        elements.append(Paragraph(methodology_text, self.styles['TechnicalBody']))
        elements.append(Spacer(1, 0.1 * inch))

        # Methodology breakdown table
        method_data = [
            ['Dimension', 'Sensors Analyzed', 'Key Metrics'],
            [
                'WHAT (Metrics)',
                'pbrake_f, pbrake_r, aps, accx_can, accy_can, Steering_Angle, speed',
                'Peak values, average usage, variance, consistency'
            ],
            [
                'WHERE (Location)',
                'gps_lat, gps_long, gps_alt, speed',
                'Corner identification, sector analysis, track position'
            ],
            [
                'WHEN (Timing)',
                'timestamp, lap_number, session_time',
                'Lap-by-lap trends, consistency, learning curve'
            ]
        ]

        method_table = Table(
            method_data,
            colWidths=[1.2 * inch, 2.6 * inch, 2.7 * inch],
            hAlign='LEFT'
        )

        method_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980b9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),

            # Data
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('WORDWRAP', (0, 0), (-1, -1), True),

            # Grid and styling
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(method_table)
        elements.append(Spacer(1, 0.15 * inch))

        # Additional notes
        notes_text = """
        <b>Statistical Approach:</b> Patterns are detected using threshold-based analysis combined with
        statistical deviation metrics. Severity ratings (High/Medium/Low) are assigned based on estimated
        lap time impact and consistency of the pattern across multiple laps. All coaching recommendations
        are data-driven and specific to observed telemetry deviations.
        """
        elements.append(Paragraph(notes_text, self.styles['TechnicalBody']))

        return elements

    def _create_patterns_section(self, patterns_data: List[Dict[str, Any]]) -> List:
        """Create detailed pattern analysis section."""
        elements = []

        # Section header
        header = Paragraph(
            f"Detected Driving Patterns ({len(patterns_data)} Total)",
            self.styles['SectionHeader']
        )
        elements.append(header)

        if not patterns_data:
            no_patterns = Paragraph(
                "No significant patterns detected. Driver performance appears optimal across all metrics.",
                self.styles['TechnicalBody']
            )
            elements.append(no_patterns)
            return elements

        # Create detailed analysis for each pattern
        for idx, pattern in enumerate(patterns_data, 1):
            # Keep each pattern together on same page
            pattern_elements = []

            # Pattern header
            pattern_title = f"{idx}. {pattern.get('pattern_name', 'Unknown Pattern')}"
            pattern_elements.append(
                Paragraph(pattern_title, self.styles['SubsectionHeader'])
            )

            # Overview table
            overview_data = [
                ['Property', 'Value'],
                ['Severity Level', pattern.get('severity', 'N/A')],
                ['Estimated Time Impact', f"{pattern.get('impact_seconds', 0):.3f} seconds/lap"],
                ['Detection Confidence', pattern.get('confidence', 'High')],
            ]

            overview_table = Table(
                overview_data,
                colWidths=[2.0 * inch, 4.5 * inch],
                hAlign='LEFT'
            )

            # Color code by severity
            severity_color = self._get_severity_color(pattern.get('severity', 'Low'))

            overview_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (1, 1), (1, 1), severity_color),
                ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('WORDWRAP', (0, 0), (-1, -1), True),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ]))

            pattern_elements.append(overview_table)
            pattern_elements.append(Spacer(1, 0.1 * inch))

            # WHAT dimension - metrics
            pattern_elements.append(
                Paragraph("<b>WHAT - Affected Metrics:</b>", self.styles['TechnicalBody'])
            )
            metrics_list = pattern.get('what_metrics', [])
            if metrics_list:
                metrics_text = ', '.join([f"<i>{m}</i>" for m in metrics_list])
                pattern_elements.append(
                    Paragraph(metrics_text, self.styles['TechnicalBody'])
                )
            else:
                pattern_elements.append(
                    Paragraph("<i>General pattern (multiple metrics)</i>", self.styles['TechnicalBody'])
                )
            pattern_elements.append(Spacer(1, 0.08 * inch))

            # WHERE dimension - corners
            pattern_elements.append(
                Paragraph("<b>WHERE - Affected Corners:</b>", self.styles['TechnicalBody'])
            )
            corners_list = pattern.get('where_corners', [])
            if corners_list:
                corners_text = ', '.join([f"Corner {c}" for c in corners_list])
                pattern_elements.append(
                    Paragraph(corners_text, self.styles['TechnicalBody'])
                )
            else:
                pattern_elements.append(
                    Paragraph("<i>Track-wide pattern (all sections)</i>", self.styles['TechnicalBody'])
                )
            pattern_elements.append(Spacer(1, 0.08 * inch))

            # WHEN dimension - laps
            pattern_elements.append(
                Paragraph("<b>WHEN - Affected Laps:</b>", self.styles['TechnicalBody'])
            )
            laps_list = pattern.get('when_laps', [])
            if laps_list:
                if len(laps_list) > 10:
                    laps_text = f"Laps {min(laps_list)}-{max(laps_list)} ({len(laps_list)} total laps)"
                else:
                    laps_text = f"Laps: {', '.join(map(str, laps_list))}"
                pattern_elements.append(
                    Paragraph(laps_text, self.styles['TechnicalBody'])
                )
            else:
                pattern_elements.append(
                    Paragraph("<i>Session-wide pattern</i>", self.styles['TechnicalBody'])
                )
            pattern_elements.append(Spacer(1, 0.15 * inch))

            # Coaching recommendation (FULL TEXT - no truncation)
            pattern_elements.append(
                Paragraph("<b>Coaching Recommendation:</b>", self.styles['TechnicalBody'])
            )
            coaching_text = pattern.get('coaching', 'No specific coaching available.')
            pattern_elements.append(
                Paragraph(coaching_text, self.styles['CoachingText'])
            )

            # Add separator between patterns
            if idx < len(patterns_data):
                pattern_elements.append(Spacer(1, 0.2 * inch))
                # Add thin line separator
                line_table = Table([['']], colWidths=[self.usable_width])
                line_table.setStyle(TableStyle([
                    ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor('#bdc3c7'))
                ]))
                pattern_elements.append(line_table)
                pattern_elements.append(Spacer(1, 0.15 * inch))

            # Keep pattern together on same page
            elements.append(KeepTogether(pattern_elements))

        return elements

    def _create_corner_analysis_section(self, corner_data: List[Dict[str, Any]]) -> List:
        """Create corner-by-corner analysis section."""
        elements = []

        # Section header
        header = Paragraph("Corner-by-Corner Analysis", self.styles['SectionHeader'])
        elements.append(header)

        intro_text = """
        This section provides detailed telemetry breakdown for each identified corner on the track.
        Metrics include entry/apex speeds, braking characteristics, and cornering forces.
        """
        elements.append(Paragraph(intro_text, self.styles['TechnicalBody']))
        elements.append(Spacer(1, 0.15 * inch))

        # Build corner analysis table
        table_data = [
            ['Corner', 'Entry Speed\n(km/h)', 'Apex Speed\n(km/h)', 'Peak Brake\n(bar)', 'Peak Lat G\n(g)', 'Notes']
        ]

        for corner in corner_data:
            corner_num = corner.get('corner_number', 'N/A')
            entry_speed = f"{corner.get('entry_speed', 0):.1f}"
            apex_speed = f"{corner.get('apex_speed', 0):.1f}"
            peak_brake = f"{corner.get('peak_brake_pressure', 0):.1f}"
            peak_lat_g = f"{corner.get('peak_lateral_g', 0):.2f}"
            notes = corner.get('notes', '')[:50] + '...' if len(corner.get('notes', '')) > 50 else corner.get('notes', '-')

            table_data.append([
                str(corner_num),
                entry_speed,
                apex_speed,
                peak_brake,
                peak_lat_g,
                notes
            ])

        corner_table = Table(
            table_data,
            colWidths=[0.6 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch, 0.85 * inch, 2.2 * inch],
            hAlign='LEFT'
        )

        corner_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),

            # Data
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (1, 1), (4, -1), 'RIGHT'),
            ('ALIGN', (5, 1), (5, -1), 'LEFT'),
            ('WORDWRAP', (0, 0), (-1, -1), True),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(corner_table)

        return elements

    def _create_appendix_section(self) -> List:
        """Create appendix with glossary and technical notes."""
        elements = []

        # Section header
        header = Paragraph("Appendix: Technical Glossary", self.styles['SectionHeader'])
        elements.append(header)

        # Glossary table
        glossary_data = [
            ['Term', 'Description', 'Units'],
            ['pbrake_f', 'Front brake pressure - hydraulic pressure applied to front brake calipers', 'bar'],
            ['pbrake_r', 'Rear brake pressure - hydraulic pressure applied to rear brake calipers', 'bar'],
            ['aps', 'Accelerator Pedal Position - throttle application percentage', '%'],
            ['accx_can', 'Longitudinal acceleration - forward/backward G-forces (CAN bus)', 'g'],
            ['accy_can', 'Lateral acceleration - side-to-side G-forces (CAN bus)', 'g'],
            ['Steering_Angle', 'Steering wheel angle - positive = right turn, negative = left turn', 'degrees'],
            ['speed', 'Vehicle speed measured via wheel sensors', 'km/h'],
            ['nmot', 'Engine RPM - motor revolutions per minute', 'rpm'],
            ['gear', 'Current transmission gear (1-6)', 'integer'],
            ['gps_lat/long/alt', 'GPS coordinates - latitude, longitude, altitude', 'deg/deg/m'],
        ]

        glossary_table = Table(
            glossary_data,
            colWidths=[1.2 * inch, 4.5 * inch, 0.8 * inch],
            hAlign='LEFT'
        )

        glossary_table.setStyle(TableStyle([
            # Header
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8e44ad')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),

            # Data
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('FONTNAME', (0, 1), (0, -1), 'Courier-Bold'),  # Monospace for sensor names
            ('WORDWRAP', (0, 0), (-1, -1), True),

            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(glossary_table)
        elements.append(Spacer(1, 0.2 * inch))

        # Technical notes
        notes_header = Paragraph("Analysis Notes", self.styles['SubsectionHeader'])
        elements.append(notes_header)

        notes = [
            "<b>Severity Levels:</b> High (>0.15s impact), Medium (0.05-0.15s), Low (<0.05s)",
            "<b>Data Quality:</b> All analyses require minimum 50 data points per lap for statistical validity",
            "<b>Corner Detection:</b> Corners identified using combined speed/steering/GPS analysis with machine learning validation",
            "<b>Pattern Confidence:</b> Based on consistency across laps and statistical significance (p<0.05)",
        ]

        for note in notes:
            elements.append(Paragraph(note, self.styles['TechnicalBody']))
            elements.append(Spacer(1, 0.08 * inch))

        # Footer disclaimer
        elements.append(Spacer(1, 0.3 * inch))
        disclaimer = """
        <i>This report is generated by automated telemetry analysis systems. While every effort is made to ensure
        accuracy, all coaching recommendations should be validated by qualified racing instructors. Track conditions,
        vehicle setup, and driver experience level significantly impact interpretation of these metrics.</i>
        """
        elements.append(Paragraph(disclaimer, self.styles['Caption']))

        return elements

    def _get_severity_color(self, severity: str) -> colors.Color:
        """Get color based on severity level."""
        severity_map = {
            'High': colors.HexColor('#e74c3c'),      # Red
            'Medium': colors.HexColor('#f39c12'),    # Orange
            'Low': colors.HexColor('#f1c40f'),       # Yellow
        }
        return severity_map.get(severity, colors.HexColor('#95a5a6'))  # Gray default

    def _format_track_name(self, track_name: str) -> str:
        """Format track name for display."""
        # Convert hyphenated names to Title Case
        return track_name.replace('-', ' ').title()


# Example usage and testing
if __name__ == '__main__':
    """
    Test the PDF generator with sample data.
    """

    # Sample pattern data
    sample_patterns = [
        {
            'pattern_name': 'Underutilizing Brake Pressure',
            'severity': 'High',
            'impact_seconds': 0.2,
            'confidence': 'High',
            'what_metrics': ['pbrake_f', 'pbrake_r'],
            'where_corners': [],
            'when_laps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'coaching': "You're only using an average of 13.1 bar when you've shown you can brake at 163.2 bar. Commit to higher brake pressure earlier in the braking zone. This is limiting your entry speed and overall lap time. Focus on building confidence in the braking system's capability to handle maximum pressure without lockup."
        },
        {
            'pattern_name': 'Inconsistent Throttle Application',
            'severity': 'Medium',
            'impact_seconds': 0.12,
            'confidence': 'Medium',
            'what_metrics': ['aps', 'accx_can'],
            'where_corners': [3, 5, 7],
            'when_laps': [4, 5, 6, 7, 8],
            'coaching': "Throttle application shows significant variation (std dev: 15.2%) in mid-corner phases. Work on smoother, more progressive throttle inputs to maintain rear grip and reduce mid-corner corrections. This will improve exit speeds and reduce tire wear."
        },
        {
            'pattern_name': 'Early Apex Entry',
            'severity': 'Low',
            'impact_seconds': 0.05,
            'confidence': 'High',
            'what_metrics': ['speed', 'gps_lat', 'gps_long'],
            'where_corners': [2, 8],
            'when_laps': [1, 2, 3],
            'coaching': "GPS data indicates apex positioning 2-3 meters earlier than optimal racing line in Corners 2 and 8. This forces early throttle lift and compromises exit speed. Practice delayed apex technique to maximize straight-line acceleration zones."
        }
    ]

    # Sample corner analysis data
    sample_corners = [
        {
            'corner_number': 1,
            'entry_speed': 145.3,
            'apex_speed': 89.2,
            'peak_brake_pressure': 135.6,
            'peak_lateral_g': 1.45,
            'notes': 'Strong braking, good apex speed'
        },
        {
            'corner_number': 2,
            'entry_speed': 132.1,
            'apex_speed': 72.5,
            'peak_brake_pressure': 98.3,
            'peak_lateral_g': 1.32,
            'notes': 'Early apex detected, low brake pressure'
        },
        {
            'corner_number': 3,
            'entry_speed': 158.7,
            'apex_speed': 95.3,
            'peak_brake_pressure': 142.1,
            'peak_lateral_g': 1.58,
            'notes': 'Optimal line, excellent execution'
        }
    ]

    # Sample metadata
    sample_metadata = {
        'analysis_date': '2025-10-30',
        'analysis_time': '14:35:22',
        'data_source': 'Telemetry Upload (CSV)',
        'total_laps': 10,
        'data_points': 125840,
    }

    # Generate PDF
    generator = TechnicalPDFGenerator()
    pdf_bytes = generator.generate_report(
        vehicle_number=2,
        track_name='circuit-of-the-americas',
        patterns_data=sample_patterns,
        corner_analysis_data=sample_corners,
        metadata=sample_metadata
    )

    # Save to file for testing
    output_path = Path(__file__).parent.parent.parent / 'output' / 'technical_report_test.pdf'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(pdf_bytes)

    print(f"[SUCCESS] Technical PDF generated successfully!")
    print(f"[FILE] Saved to: {output_path}")
    print(f"[SIZE] File size: {len(pdf_bytes) / 1024:.1f} KB")
    print(f"[PATTERNS] Patterns included: {len(sample_patterns)}")
    print(f"[CORNERS] Corners analyzed: {len(sample_corners)}")
