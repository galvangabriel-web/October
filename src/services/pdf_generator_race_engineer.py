"""
Race Engineer PDF Report Generator - Strategic Coaching Edition

This module generates race engineer-style PDF reports with prioritized coaching
recommendations upfront, strategic assessments, and cross-correlation insights.

Author: Racing Telemetry Analysis System - Race Engineer AI
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


class RaceEngineerPDFGenerator:
    """
    Generates race engineer-style PDF reports with strategic coaching prioritization.

    Key Features:
    - Executive Coaching Summary upfront (all recommendations together)
    - Race Engineer's Strategic Assessment (prioritized action plan)
    - Cross-correlation insights from multiple data sources
    - Track-specific tactical recommendations
    """

    def __init__(self):
        """Initialize the PDF generator with race engineer styling."""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Page configuration
        self.page_width = letter[0]
        self.page_height = letter[1]
        self.margin = 0.75 * inch
        self.usable_width = self.page_width - (2 * self.margin)

    def _setup_custom_styles(self):
        """Create custom paragraph styles for race engineer reports."""

        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=26,
            textColor=colors.HexColor('#c0392b'),  # Racing red
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))

        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#c0392b'),  # Racing red
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=colors.HexColor('#e74c3c'),
            borderPadding=6,
            backColor=colors.HexColor('#fef5e7')  # Light yellow
        ))

        # Priority header style
        self.styles.add(ParagraphStyle(
            name='PriorityHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.white,
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            backColor=colors.HexColor('#c0392b'),
            borderPadding=8
        ))

        # Coach recommendation style (actionable)
        self.styles.add(ParagraphStyle(
            name='CoachingAction',
            parent=self.styles['BodyText'],
            fontSize=11,
            leading=16,
            alignment=TA_LEFT,
            fontName='Helvetica',
            textColor=colors.HexColor('#2c3e50'),
            leftIndent=15,
            rightIndent=10,
            spaceAfter=10,
            bulletIndent=10
        ))

        # Strategic insight style
        self.styles.add(ParagraphStyle(
            name='StrategicInsight',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
            textColor=colors.HexColor('#16a085'),  # Teal for insights
            leftIndent=10,
            rightIndent=10,
            spaceAfter=8
        ))

    def _create_header_footer(self, canvas_obj, doc):
        """Add header and footer to each page."""
        canvas_obj.saveState()

        # Header
        canvas_obj.setFont('Helvetica-Bold', 10)
        canvas_obj.setFillColor(colors.HexColor('#c0392b'))
        canvas_obj.drawString(
            self.margin,
            self.page_height - 0.5 * inch,
            "RACE ENGINEER REPORT - CONFIDENTIAL"
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
        metadata: Optional[Dict[str, Any]] = None,
        weather_context: Optional[Dict[str, Any]] = None,
        sector_benchmarks: Optional[Dict[str, Any]] = None
    ) -> bytes:
        """
        Generate a complete race engineer PDF report.

        Args:
            vehicle_number: Race vehicle number
            track_name: Name of the racing track
            patterns_data: List of detected driving patterns
            corner_analysis_data: Optional corner-by-corner data
            metadata: Optional metadata (laps, data points, etc.)
            weather_context: Optional weather correlation data
            sector_benchmarks: Optional sector benchmark comparisons

        Returns:
            bytes: PDF file as bytes
        """
        buffer = BytesIO()

        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=1.0 * inch,
            bottomMargin=0.75 * inch,
            title=f"Race Engineer Report - Vehicle {vehicle_number}"
        )

        # Build content
        story = []

        # Page 1: Title + Executive Coaching Summary
        story.extend(self._create_title_section(vehicle_number, track_name))
        story.extend(self._create_executive_coaching_summary(patterns_data))
        story.append(PageBreak())

        # Page 2: Race Engineer's Strategic Assessment
        story.extend(self._create_strategic_assessment(
            patterns_data,
            corner_analysis_data,
            weather_context,
            sector_benchmarks
        ))
        story.append(PageBreak())

        # Page 3+: Detailed Technical Analysis
        story.extend(self._create_detailed_patterns_section(patterns_data))

        # Corner analysis (if available)
        if corner_analysis_data:
            story.append(PageBreak())
            story.extend(self._create_corner_analysis_section(corner_analysis_data))

        # Appendix
        story.append(PageBreak())
        story.extend(self._create_appendix_section())

        # Build PDF
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)

        pdf_bytes = buffer.getvalue()
        buffer.close()

        return pdf_bytes

    def _create_title_section(self, vehicle_number: int, track_name: str) -> List:
        """Create title section."""
        elements = []

        # Main title
        title = Paragraph(
            f"Race Engineer Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.1 * inch))

        # Subtitle
        subtitle = Paragraph(
            f"<b>Vehicle #{vehicle_number}</b> | {self._format_track_name(track_name)}",
            self.styles['Heading2']
        )
        elements.append(subtitle)

        # Report date
        report_info = Paragraph(
            f"<i>Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            self.styles['Normal']
        )
        elements.append(report_info)
        elements.append(Spacer(1, 0.3 * inch))

        return elements

    def _create_executive_coaching_summary(self, patterns_data: List[Dict[str, Any]]) -> List:
        """Create executive coaching summary with all recommendations together."""
        elements = []

        # Section header
        header = Paragraph("Executive Coaching Summary", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.1 * inch))

        intro = Paragraph(
            "Based on comprehensive telemetry analysis, the following coaching recommendations "
            "are prioritized by lap time impact. Focus on HIGH priority items first for maximum performance gain.",
            self.styles['Normal']
        )
        elements.append(intro)
        elements.append(Spacer(1, 0.15 * inch))

        # Sort patterns by impact (high to low)
        sorted_patterns = sorted(
            patterns_data,
            key=lambda x: x.get('impact_seconds', 0),
            reverse=True
        )

        # Group by severity
        high_priority = [p for p in sorted_patterns if p.get('severity', '').upper() == 'HIGH']
        medium_priority = [p for p in sorted_patterns if p.get('severity', '').upper() == 'MEDIUM']
        low_priority = [p for p in sorted_patterns if p.get('severity', '').upper() == 'LOW']

        # HIGH PRIORITY
        if high_priority:
            elements.append(Paragraph("🔴 HIGH PRIORITY - Focus Here First", self.styles['PriorityHeader']))
            elements.append(Spacer(1, 0.1 * inch))

            for idx, pattern in enumerate(high_priority, 1):
                coaching = pattern.get('coaching', 'No coaching available.')
                impact = pattern.get('impact_seconds', 0)
                pattern_name = pattern.get('pattern_name', 'Unknown')

                # Create coaching card
                coaching_text = f"<b>{idx}. {pattern_name}</b> (Impact: +{impact:.3f}s/lap)<br/><br/>{coaching}"
                elements.append(Paragraph(coaching_text, self.styles['CoachingAction']))

            elements.append(Spacer(1, 0.15 * inch))

        # MEDIUM PRIORITY
        if medium_priority:
            elements.append(Paragraph("🟡 MEDIUM PRIORITY - Address Next", self.styles['PriorityHeader']))
            elements.append(Spacer(1, 0.1 * inch))

            for idx, pattern in enumerate(medium_priority, 1):
                coaching = pattern.get('coaching', 'No coaching available.')
                impact = pattern.get('impact_seconds', 0)
                pattern_name = pattern.get('pattern_name', 'Unknown')

                coaching_text = f"<b>{idx}. {pattern_name}</b> (Impact: +{impact:.3f}s/lap)<br/><br/>{coaching}"
                elements.append(Paragraph(coaching_text, self.styles['CoachingAction']))

            elements.append(Spacer(1, 0.15 * inch))

        # LOW PRIORITY
        if low_priority:
            elements.append(Paragraph("🟢 LOW PRIORITY - Fine-Tuning", self.styles['PriorityHeader']))
            elements.append(Spacer(1, 0.1 * inch))

            for idx, pattern in enumerate(low_priority, 1):
                coaching = pattern.get('coaching', 'No coaching available.')
                impact = pattern.get('impact_seconds', 0)
                pattern_name = pattern.get('pattern_name', 'Unknown')

                coaching_text = f"<b>{idx}. {pattern_name}</b> (Impact: +{impact:.3f}s/lap)<br/><br/>{coaching}"
                elements.append(Paragraph(coaching_text, self.styles['CoachingAction']))

        return elements

    def _create_strategic_assessment(
        self,
        patterns_data: List[Dict[str, Any]],
        corner_analysis_data: Optional[List[Dict[str, Any]]],
        weather_context: Optional[Dict[str, Any]],
        sector_benchmarks: Optional[Dict[str, Any]]
    ) -> List:
        """Create race engineer's strategic assessment."""
        elements = []

        # Section header
        header = Paragraph("Race Engineer's Strategic Assessment", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.15 * inch))

        # Calculate total potential gain
        total_gain = sum(p.get('impact_seconds', 0) for p in patterns_data)

        # Strategic overview
        overview_text = f"""
        <b>Performance Overview:</b><br/>
        Total lap time improvement potential: <b>{total_gain:.3f} seconds per lap</b><br/>
        Patterns detected: {len(patterns_data)}<br/>
        Critical focus areas: {len([p for p in patterns_data if p.get('severity', '').upper() == 'HIGH'])}
        """
        elements.append(Paragraph(overview_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.15 * inch))

        # Strategic recommendations
        elements.append(Paragraph("<b>STRATEGIC RECOMMENDATIONS:</b>", self.styles['Heading3']))
        elements.append(Spacer(1, 0.1 * inch))

        # Recommendation 1: Primary Focus
        primary_pattern = max(patterns_data, key=lambda x: x.get('impact_seconds', 0), default=None)
        if primary_pattern:
            rec1 = f"""
            <b>1. PRIMARY FOCUS THIS SESSION:</b><br/>
            Work on <b>{primary_pattern.get('pattern_name', 'primary pattern')}</b>.
            This single improvement can gain you {primary_pattern.get('impact_seconds', 0):.3f}s per lap.
            Focus {len(primary_pattern.get('when_laps', []))} laps specifically on this technique.
            """
            elements.append(Paragraph(rec1, self.styles['StrategicInsight']))

        # Recommendation 2: Session Plan
        rec2 = """
        <b>2. SESSION PLAN:</b><br/>
        • First 3 laps: Focus on HIGH priority items only<br/>
        • Next 3 laps: Combine HIGH + MEDIUM priority techniques<br/>
        • Final 4 laps: Full integration of all improvements<br/>
        • Use data from each stint to measure progress
        """
        elements.append(Paragraph(rec2, self.styles['StrategicInsight']))

        # Recommendation 3: Corner-specific
        if corner_analysis_data:
            worst_corners = sorted(
                corner_analysis_data,
                key=lambda x: x.get('peak_brake_pressure', 0)
            )[:2]
            corner_nums = [c.get('corner_number', '?') for c in worst_corners]
            rec3 = f"""
            <b>3. CORNER-SPECIFIC TARGETS:</b><br/>
            Focus improvement efforts on Corners {', '.join(map(str, corner_nums))} where brake
            confidence and line optimization will yield the biggest gains. These corners set up
            the longest straights on track.
            """
            elements.append(Paragraph(rec3, self.styles['StrategicInsight']))

        # Recommendation 4: Measurement criteria
        rec4 = """
        <b>4. SUCCESS CRITERIA:</b><br/>
        • Target: Reduce lap time by 0.2s within this session<br/>
        • Monitor: Brake pressure consistency (std dev < 10%)<br/>
        • Validate: Compare sector times lap-by-lap<br/>
        • Adjust: If no improvement after 5 laps, reassess technique
        """
        elements.append(Paragraph(rec4, self.styles['StrategicInsight']))

        elements.append(Spacer(1, 0.2 * inch))

        # What NOT to do (common mistakes)
        elements.append(Paragraph("<b>AVOID THESE MISTAKES:</b>", self.styles['Heading3']))
        elements.append(Spacer(1, 0.1 * inch))

        mistakes = """
        • Don't try to fix everything at once - prioritize<br/>
        • Don't change multiple techniques simultaneously - isolate variables<br/>
        • Don't ignore high-impact items to work on comfort zones<br/>
        • Don't skip warm-up laps - build confidence progressively
        """
        elements.append(Paragraph(mistakes, self.styles['Normal']))

        return elements

    def _create_detailed_patterns_section(self, patterns_data: List[Dict[str, Any]]) -> List:
        """Create detailed pattern analysis (technical reference)."""
        elements = []

        header = Paragraph("Detailed Technical Analysis", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.1 * inch))

        note = Paragraph(
            "<i>This section provides detailed technical breakdown of each pattern for reference. "
            "Focus on the Executive Summary and Strategic Assessment above for actionable guidance.</i>",
            self.styles['Normal']
        )
        elements.append(note)
        elements.append(Spacer(1, 0.15 * inch))

        # Sort by impact
        sorted_patterns = sorted(
            patterns_data,
            key=lambda x: x.get('impact_seconds', 0),
            reverse=True
        )

        for idx, pattern in enumerate(sorted_patterns, 1):
            pattern_elements = []

            # Pattern header
            pattern_title = f"{idx}. {pattern.get('pattern_name', 'Unknown Pattern')}"
            pattern_elements.append(Paragraph(pattern_title, self.styles['Heading3']))

            # Technical details table
            details_data = [
                ['Property', 'Value'],
                ['Severity', pattern.get('severity', 'N/A')],
                ['Lap Time Impact', f"{pattern.get('impact_seconds', 0):.3f}s per lap"],
                ['Affected Metrics', ', '.join(pattern.get('what_metrics', []))],
                ['Affected Corners', ', '.join(map(str, pattern.get('where_corners', []))) or 'Track-wide'],
                ['Affected Laps', f"{len(pattern.get('when_laps', []))} laps"],
            ]

            details_table = Table(details_data, colWidths=[2.0 * inch, 4.5 * inch])
            details_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))

            pattern_elements.append(details_table)
            pattern_elements.append(Spacer(1, 0.15 * inch))

            if idx < len(sorted_patterns):
                pattern_elements.append(Spacer(1, 0.1 * inch))

            elements.append(KeepTogether(pattern_elements))

        return elements

    def _create_corner_analysis_section(self, corner_data: List[Dict[str, Any]]) -> List:
        """Create corner-by-corner analysis section."""
        elements = []

        header = Paragraph("Corner-by-Corner Analysis", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.15 * inch))

        # Build corner table
        table_data = [
            ['Corner', 'Entry Speed\n(km/h)', 'Apex Speed\n(km/h)', 'Peak Brake\n(bar)', 'Peak Lat G\n(g)', 'Assessment']
        ]

        for corner in corner_data:
            corner_num = corner.get('corner_number', 'N/A')
            entry_speed = f"{corner.get('entry_speed', 0):.1f}"
            apex_speed = f"{corner.get('apex_speed', 0):.1f}"
            peak_brake = f"{corner.get('peak_brake_pressure', 0):.1f}"
            peak_lat_g = f"{corner.get('peak_lateral_g', 0):.2f}"
            notes = corner.get('notes', '')[:45] + '...' if len(corner.get('notes', '')) > 45 else corner.get('notes', '-')

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
            colWidths=[0.6 * inch, 0.95 * inch, 0.95 * inch, 0.95 * inch, 0.85 * inch, 2.2 * inch]
        )

        corner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ('TOPPADDING', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1), True),
        ]))

        elements.append(corner_table)

        return elements

    def _create_appendix_section(self) -> List:
        """Create appendix with notes."""
        elements = []

        header = Paragraph("Race Engineer Notes", self.styles['SectionHeader'])
        elements.append(header)
        elements.append(Spacer(1, 0.15 * inch))

        notes = [
            "<b>Coaching Philosophy:</b> Progressive improvement over single-session heroics",
            "<b>Data Confidence:</b> All recommendations based on multi-lap statistical analysis",
            "<b>Safety First:</b> Never compromise safety for lap time - build speed gradually",
            "<b>Context Matters:</b> Weather, tire condition, and traffic affect applicability",
        ]

        for note in notes:
            elements.append(Paragraph(note, self.styles['Normal']))
            elements.append(Spacer(1, 0.08 * inch))

        elements.append(Spacer(1, 0.3 * inch))

        disclaimer = """
        <i>This report combines telemetry analysis with race engineering best practices. All recommendations
        should be implemented progressively with qualified coaching supervision. Driver safety and comfort
        are paramount - never force techniques that don't feel natural.</i>
        """
        elements.append(Paragraph(disclaimer, self.styles['Normal']))

        return elements

    def _format_track_name(self, track_name: str) -> str:
        """Format track name for display."""
        return track_name.replace('-', ' ').title()


# For testing
if __name__ == '__main__':
    print("Race Engineer PDF Generator - Use via dashboard or integration script")
