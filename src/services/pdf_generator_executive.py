"""
Executive Summary PDF Generator for Racing Telemetry Analysis

Generates clean, stakeholder-friendly PDF reports with:
- Professional cover page with branding
- Executive summary with KPI cards
- Top 3 improvement areas with coaching
- Color-coded severity indicators
- Plain language recommendations

Author: Racing Analysis Team
Version: 1.0.0
"""

from datetime import datetime
from io import BytesIO
from typing import List, Dict, Any, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, Frame, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class ExecutivePDFGenerator:
    """
    Executive-style PDF report generator for racing telemetry insights.

    Features:
    - Clean, non-technical language
    - KPI cards with Unicode icons
    - Color-coded severity levels
    - Professional layout (2-3 pages)
    - Actionable recommendations
    """

    # Color scheme
    COLOR_PRIMARY = colors.HexColor('#1e3a8a')  # Deep blue
    COLOR_SUCCESS = colors.HexColor('#16a34a')  # Green
    COLOR_WARNING = colors.HexColor('#d97706')  # Amber
    COLOR_DANGER = colors.HexColor('#dc2626')   # Red
    COLOR_LIGHT_BG = colors.HexColor('#f8fafc') # Light gray
    COLOR_BORDER = colors.HexColor('#cbd5e1')   # Border gray

    # Unicode icons
    ICON_SPEED = "🏁"
    ICON_BRAKE = "🛑"
    ICON_CORNER = "↗️"
    ICON_THROTTLE = "⚡"
    ICON_WARNING = "⚠️"
    ICON_SUCCESS = "✓"
    ICON_LAP = "🔄"
    ICON_IMPROVEMENT = "📈"

    def __init__(self, page_size=letter):
        """
        Initialize PDF generator.

        Args:
            page_size: Page size (default: letter, can use A4)
        """
        self.page_size = page_size
        self.width, self.height = page_size
        self.styles = self._create_styles()

    def _create_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles."""
        styles = getSampleStyleSheet()

        custom_styles = {
            'Title': ParagraphStyle(
                'CustomTitle',
                parent=styles['Title'],
                fontSize=28,
                textColor=self.COLOR_PRIMARY,
                spaceAfter=20,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ),
            'Heading1': ParagraphStyle(
                'CustomHeading1',
                parent=styles['Heading1'],
                fontSize=18,
                textColor=self.COLOR_PRIMARY,
                spaceAfter=12,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            ),
            'Heading2': ParagraphStyle(
                'CustomHeading2',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=self.COLOR_PRIMARY,
                spaceAfter=8,
                spaceBefore=8,
                fontName='Helvetica-Bold'
            ),
            'Body': ParagraphStyle(
                'CustomBody',
                parent=styles['BodyText'],
                fontSize=11,
                leading=14,
                alignment=TA_JUSTIFY,
                spaceAfter=8
            ),
            'KPI_Value': ParagraphStyle(
                'KPIValue',
                parent=styles['Normal'],
                fontSize=24,
                fontName='Helvetica-Bold',
                textColor=self.COLOR_PRIMARY,
                alignment=TA_CENTER
            ),
            'KPI_Label': ParagraphStyle(
                'KPILabel',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#64748b'),
                alignment=TA_CENTER
            ),
            'Coaching': ParagraphStyle(
                'Coaching',
                parent=styles['BodyText'],
                fontSize=11,
                leading=15,
                alignment=TA_LEFT,
                leftIndent=10,
                rightIndent=10,
                spaceAfter=10
            ),
            'Subtitle': ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#64748b'),
                alignment=TA_CENTER,
                spaceAfter=30
            )
        }

        return custom_styles

    def _get_severity_color(self, severity: str) -> colors.Color:
        """Get color for severity level."""
        severity_lower = severity.lower()
        if severity_lower in ['high', 'critical']:
            return self.COLOR_DANGER
        elif severity_lower in ['medium', 'moderate']:
            return self.COLOR_WARNING
        else:
            return self.COLOR_SUCCESS

    def _create_kpi_card(self, icon: str, value: str, label: str,
                         width: float = 2.0) -> Table:
        """
        Create a KPI card with icon, value, and label.

        Args:
            icon: Unicode icon
            value: Main value to display
            label: Description label
            width: Card width in inches

        Returns:
            Table object representing the KPI card
        """
        # Create paragraphs
        icon_para = Paragraph(f'<font size="24">{icon}</font>',
                             self.styles['KPI_Label'])
        value_para = Paragraph(f'<b>{value}</b>', self.styles['KPI_Value'])
        label_para = Paragraph(label, self.styles['KPI_Label'])

        # Build table
        data = [
            [icon_para],
            [value_para],
            [label_para]
        ]

        table = Table(data, colWidths=[width * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), self.COLOR_LIGHT_BG),
            ('BORDER', (0, 0), (-1, -1), 1, self.COLOR_BORDER),
            ('ROUNDEDCORNERS', [8, 8, 8, 8]),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))

        return table

    def _create_improvement_card(self, rank: int, pattern: Dict[str, Any],
                                 width: float = 7.0) -> Table:
        """
        Create an improvement area card with coaching.

        Args:
            rank: Priority rank (1, 2, 3)
            pattern: Pattern dictionary with name, severity, coaching, etc.
            width: Card width in inches

        Returns:
            Table object representing the improvement card
        """
        severity = pattern.get('severity', 'Medium')
        severity_color = self._get_severity_color(severity)
        impact = pattern.get('impact_seconds', 0.0)
        coaching = pattern.get('coaching', 'No coaching available')
        pattern_name = pattern.get('pattern_name', 'Unknown Pattern')

        # Header row with rank and severity
        rank_text = f'<font size="16"><b>#{rank}</b></font>'
        severity_badge = f'<font color="{severity_color.hexval()}"><b>{severity.upper()}</b></font>'

        header_para = Paragraph(
            f'{rank_text} &nbsp;&nbsp; {severity_badge}',
            self.styles['Heading2']
        )

        # Pattern name
        name_para = Paragraph(
            f'<b>{pattern_name}</b>',
            self.styles['Heading2']
        )

        # Impact
        impact_text = f'{self.ICON_IMPROVEMENT} <b>Potential Gain:</b> {abs(impact):.2f} seconds per lap'
        impact_para = Paragraph(impact_text, self.styles['Body'])

        # Coaching
        coaching_text = f'{self.ICON_LAP} <b>What to Do:</b><br/>{coaching}'
        coaching_para = Paragraph(coaching_text, self.styles['Coaching'])

        # Build table
        data = [
            [header_para],
            [name_para],
            [impact_para],
            [coaching_para]
        ]

        table = Table(data, colWidths=[width * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), severity_color),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('BORDER', (0, 0), (-1, -1), 2, severity_color),
            ('ROUNDEDCORNERS', [10, 10, 10, 10]),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 15),
            ('RIGHTPADDING', (0, 0), (-1, -1), 15),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ]))

        return table

    def _create_cover_page(self, vehicle_number: int, track_name: str,
                           session_date: Optional[str] = None) -> List:
        """
        Create cover page elements.

        Args:
            vehicle_number: Vehicle number
            track_name: Track name
            session_date: Session date (optional)

        Returns:
            List of flowable elements
        """
        elements = []

        # Spacer to center content
        elements.append(Spacer(1, 2 * inch))

        # Title
        title = Paragraph(
            f'Performance Analysis Report',
            self.styles['Title']
        )
        elements.append(title)

        # Subtitle
        track_display = track_name.replace('-', ' ').title()
        subtitle_text = f'Vehicle #{vehicle_number} • {track_display}'
        if session_date:
            subtitle_text += f' • {session_date}'

        subtitle = Paragraph(subtitle_text, self.styles['Subtitle'])
        elements.append(subtitle)

        elements.append(Spacer(1, 0.5 * inch))

        # Divider line
        divider_data = [['                                                                    ']]
        divider = Table(divider_data, colWidths=[6 * inch])
        divider.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 2, self.COLOR_PRIMARY),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(divider)

        return elements

    def _create_executive_summary(self, patterns_data: List[Dict[str, Any]]) -> List:
        """
        Create executive summary section with KPI cards.

        Args:
            patterns_data: List of pattern dictionaries

        Returns:
            List of flowable elements
        """
        elements = []

        # Section header
        header = Paragraph('Executive Summary', self.styles['Heading1'])
        elements.append(header)

        # Calculate KPIs
        total_patterns = len(patterns_data)
        high_priority = sum(1 for p in patterns_data
                           if p.get('severity', '').lower() in ['high', 'critical'])
        total_impact = sum(abs(p.get('impact_seconds', 0.0)) for p in patterns_data)
        avg_impact = total_impact / total_patterns if total_patterns > 0 else 0.0

        # Summary text
        summary_text = (
            f'This analysis identified <b>{total_patterns}</b> opportunities for improvement, '
            f'with <b>{high_priority}</b> requiring immediate attention. '
            f'Addressing these areas could improve lap times by up to '
            f'<b>{total_impact:.2f} seconds</b>.'
        )
        summary_para = Paragraph(summary_text, self.styles['Body'])
        elements.append(summary_para)
        elements.append(Spacer(1, 0.3 * inch))

        # KPI Cards in a row
        kpi_row = [
            self._create_kpi_card(
                self.ICON_WARNING,
                str(total_patterns),
                'Areas to Improve',
                width=2.2
            ),
            self._create_kpi_card(
                self.ICON_SPEED,
                f'{high_priority}',
                'High Priority',
                width=2.2
            ),
            self._create_kpi_card(
                self.ICON_IMPROVEMENT,
                f'{total_impact:.2f}s',
                'Total Potential Gain',
                width=2.2
            )
        ]

        kpi_table = Table([kpi_row], colWidths=[2.3*inch, 2.3*inch, 2.3*inch])
        kpi_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 0.4 * inch))

        return elements

    def _create_improvement_areas(self, patterns_data: List[Dict[str, Any]]) -> List:
        """
        Create top 3 improvement areas section.

        Args:
            patterns_data: List of pattern dictionaries

        Returns:
            List of flowable elements
        """
        elements = []

        # Section header
        header = Paragraph('Top Priority Improvements', self.styles['Heading1'])
        elements.append(header)
        elements.append(Spacer(1, 0.2 * inch))

        # Sort by severity and impact
        def sort_key(p):
            severity_map = {'high': 3, 'critical': 3, 'medium': 2, 'moderate': 2, 'low': 1}
            severity_score = severity_map.get(p.get('severity', '').lower(), 0)
            impact_score = abs(p.get('impact_seconds', 0.0))
            return (severity_score, impact_score)

        sorted_patterns = sorted(patterns_data, key=sort_key, reverse=True)

        # Create cards for top 3
        for rank, pattern in enumerate(sorted_patterns[:3], start=1):
            card = self._create_improvement_card(rank, pattern, width=7.0)
            elements.append(KeepTogether(card))

            if rank < min(3, len(sorted_patterns)):
                elements.append(Spacer(1, 0.25 * inch))

        return elements

    def _create_action_plan(self, patterns_data: List[Dict[str, Any]]) -> List:
        """
        Create action plan section.

        Args:
            patterns_data: List of pattern dictionaries

        Returns:
            List of flowable elements
        """
        elements = []

        # Section header
        header = Paragraph('Recommended Action Plan', self.styles['Heading1'])
        elements.append(header)
        elements.append(Spacer(1, 0.2 * inch))

        # Action steps
        actions = [
            f'{self.ICON_SUCCESS} Focus on the top 3 priority areas identified above',
            f'{self.ICON_LAP} Practice specific techniques during next 2-3 sessions',
            f'{self.ICON_IMPROVEMENT} Review telemetry after each session to track progress',
            f'{self.ICON_BRAKE} Work with coach to refine braking and corner entry techniques'
        ]

        for action in actions:
            action_para = Paragraph(f'• {action}', self.styles['Body'])
            elements.append(action_para)
            elements.append(Spacer(1, 0.1 * inch))

        return elements

    def _add_footer(self, canvas, doc):
        """
        Add footer to each page.

        Args:
            canvas: ReportLab canvas
            doc: Document template
        """
        canvas.saveState()

        # Footer line
        canvas.setStrokeColor(self.COLOR_BORDER)
        canvas.setLineWidth(1)
        canvas.line(0.75 * inch, 0.5 * inch,
                   self.width - 0.75 * inch, 0.5 * inch)

        # Footer text
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#64748b'))

        footer_text = f'Generated {datetime.now().strftime("%B %d, %Y at %I:%M %p")}'
        canvas.drawString(0.75 * inch, 0.35 * inch, footer_text)

        # Page number
        page_num = f'Page {doc.page}'
        canvas.drawRightString(self.width - 0.75 * inch, 0.35 * inch, page_num)

        canvas.restoreState()

    def generate_report(self,
                       vehicle_number: int,
                       track_name: str,
                       patterns_data: List[Dict[str, Any]],
                       session_date: Optional[str] = None) -> bytes:
        """
        Generate complete executive summary PDF report.

        Args:
            vehicle_number: Vehicle number
            track_name: Track name
            patterns_data: List of pattern dictionaries with keys:
                - pattern_name: str
                - severity: str ('High', 'Medium', 'Low')
                - impact_seconds: float
                - coaching: str
                - what_metrics: List[str] (optional)
                - where_corners: List[int] (optional)
                - when_laps: List[int] (optional)
            session_date: Session date string (optional)

        Returns:
            PDF bytes

        Example:
            >>> generator = ExecutivePDFGenerator()
            >>> patterns = [
            ...     {
            ...         'pattern_name': 'Underutilizing Brake Pressure',
            ...         'severity': 'High',
            ...         'impact_seconds': 0.2,
            ...         'coaching': 'Commit to higher brake pressure earlier...'
            ...     }
            ... ]
            >>> pdf_bytes = generator.generate_report(5, 'cota', patterns)
        """
        buffer = BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=self.page_size,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            title=f'Performance Analysis - Vehicle #{vehicle_number}',
            author='Racing Analysis Team'
        )

        # Build content
        elements = []

        # Page 1: Cover + Executive Summary
        elements.extend(self._create_cover_page(vehicle_number, track_name, session_date))
        elements.append(Spacer(1, 0.5 * inch))
        elements.extend(self._create_executive_summary(patterns_data))

        # Page 2: Top 3 Improvements
        elements.append(PageBreak())
        elements.extend(self._create_improvement_areas(patterns_data))

        # Page 3: Action Plan (if we have space)
        if len(patterns_data) <= 3:
            elements.append(Spacer(1, 0.5 * inch))
            elements.extend(self._create_action_plan(patterns_data))
        else:
            elements.append(PageBreak())
            elements.extend(self._create_action_plan(patterns_data))

        # Build PDF
        doc.build(elements, onFirstPage=self._add_footer,
                 onLaterPages=self._add_footer)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes


# Convenience function
def generate_executive_report(vehicle_number: int,
                             track_name: str,
                             patterns_data: List[Dict[str, Any]],
                             session_date: Optional[str] = None,
                             output_path: Optional[str] = None) -> bytes:
    """
    Generate executive summary PDF report.

    Args:
        vehicle_number: Vehicle number
        track_name: Track name
        patterns_data: List of pattern dictionaries
        session_date: Session date (optional)
        output_path: Path to save PDF (optional, if None returns bytes only)

    Returns:
        PDF bytes

    Example:
        >>> patterns = [{'pattern_name': 'Brake Too Late', 'severity': 'High', ...}]
        >>> pdf_bytes = generate_executive_report(5, 'cota', patterns)
        >>> # Or save to file
        >>> generate_executive_report(5, 'cota', patterns,
        ...                          output_path='report.pdf')
    """
    generator = ExecutivePDFGenerator()
    pdf_bytes = generator.generate_report(vehicle_number, track_name,
                                         patterns_data, session_date)

    if output_path:
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

    return pdf_bytes


if __name__ == '__main__':
    # Demo: Generate sample report
    sample_patterns = [
        {
            'pattern_name': 'Underutilizing Brake Pressure',
            'severity': 'High',
            'impact_seconds': 0.2,
            'what_metrics': ['pbrake_f', 'pbrake_r'],
            'where_corners': [],
            'when_laps': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'coaching': "You're only using an average of 13.1 bar when you've shown you can brake at 163.2 bar. Commit to higher brake pressure earlier in the braking zone to maximize deceleration and carry more speed into corners."
        },
        {
            'pattern_name': 'Inconsistent Throttle Application',
            'severity': 'Medium',
            'impact_seconds': 0.15,
            'what_metrics': ['aps'],
            'where_corners': [3, 5, 8],
            'when_laps': [4, 5, 6],
            'coaching': 'Your throttle inputs vary significantly lap-to-lap. Focus on smooth, progressive throttle application at corner exit to maximize traction and reduce lap time variation.'
        },
        {
            'pattern_name': 'Late Corner Apex',
            'severity': 'Medium',
            'impact_seconds': 0.12,
            'what_metrics': ['speed', 'Steering_Angle'],
            'where_corners': [1, 4, 7],
            'when_laps': [2, 3, 7, 8],
            'coaching': 'You are reaching the apex point later than optimal in several corners. Try turning in earlier and focusing on hitting the geometric apex to maximize corner exit speed.'
        },
        {
            'pattern_name': 'Excessive Steering Corrections',
            'severity': 'Low',
            'impact_seconds': 0.05,
            'what_metrics': ['Steering_Angle'],
            'where_corners': [2, 6],
            'when_laps': [1, 9, 10],
            'coaching': 'Minor steering oscillations detected mid-corner. Work on smoother initial turn-in and maintaining a steady arc through the corner to reduce scrubbing and improve stability.'
        }
    ]

    # Generate demo report
    print("Generating demo executive summary PDF...")
    pdf_bytes = generate_executive_report(
        vehicle_number=5,
        track_name='circuit-of-the-americas',
        patterns_data=sample_patterns,
        session_date='October 27, 2025',
        output_path='demo_executive_report.pdf'
    )

    print(f"✓ Demo report generated: demo_executive_report.pdf ({len(pdf_bytes):,} bytes)")
    print(f"  - {len(sample_patterns)} improvement areas identified")
    print(f"  - Total potential gain: {sum(p['impact_seconds'] for p in sample_patterns):.2f} seconds")
    print(f"  - High priority items: {sum(1 for p in sample_patterns if p['severity'] == 'High')}")
