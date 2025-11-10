"""
Consistency Analysis Module
============================

Calculates lap-to-lap consistency metrics for racing performance analysis.

Features:
- Overall consistency score (0-100)
- Corner-by-corner consistency analysis
- Standard deviation calculations for entry/apex/exit speeds
- Most/least consistent corners identification
- Coaching advice based on consistency level
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_consistency_metrics(corner_data_list: List[Dict]) -> Dict:
    """
    Calculate comprehensive consistency metrics across all laps

    Args:
        corner_data_list: List of corner data from all laps/vehicles

    Returns:
        Dictionary with consistency analysis:
        - overall_consistency: Score 0-100 (higher = more consistent)
        - by_corner: Dict mapping corner number to consistency metrics
        - most_consistent: Top 3 most consistent corners
        - least_consistent: Top 3 least consistent corners
        - total_laps: Number of laps analyzed
    """
    if not corner_data_list or len(corner_data_list) == 0:
        return {
            "overall_consistency": 0,
            "by_corner": {},
            "most_consistent": [],
            "least_consistent": [],
            "total_laps": 0
        }

    # Group corners by corner number
    corners_by_number = {}
    for corner in corner_data_list:
        corner_num = corner.get('corner_number', corner.get('corner_name', 'Unknown'))
        if corner_num not in corners_by_number:
            corners_by_number[corner_num] = []
        corners_by_number[corner_num].append(corner)

    # Calculate consistency for each corner
    consistency_scores = {}

    for corner_num, corners in corners_by_number.items():
        if len(corners) < 2:
            # Need at least 2 laps to calculate consistency
            continue

        # Extract speed metrics
        entry_speeds = [c.get('entry_speed', 0) for c in corners if 'entry_speed' in c]
        apex_speeds = [c.get('apex_speed', 0) for c in corners if 'apex_speed' in c]
        exit_speeds = [c.get('exit_speed', 0) for c in corners if 'exit_speed' in c]
        brake_points = [c.get('brake_point', 0) for c in corners if 'brake_point' in c]
        corner_times = [c.get('time_in_corner', 0) for c in corners if 'time_in_corner' in c]

        # Calculate standard deviations
        entry_std = np.std(entry_speeds) if len(entry_speeds) > 1 else 0.0
        apex_std = np.std(apex_speeds) if len(apex_speeds) > 1 else 0.0
        exit_std = np.std(exit_speeds) if len(exit_speeds) > 1 else 0.0
        brake_std = np.std(brake_points) if len(brake_points) > 1 else 0.0
        time_std = np.std(corner_times) if len(corner_times) > 1 else 0.0

        # Calculate means for context
        entry_mean = np.mean(entry_speeds) if entry_speeds else 0.0
        apex_mean = np.mean(apex_speeds) if apex_speeds else 0.0
        exit_mean = np.mean(exit_speeds) if exit_speeds else 0.0

        # Overall consistency score for this corner (0-100)
        # Lower std dev = better consistency
        # Formula: 100 - (average_std_dev_percentage)
        avg_speed = (entry_mean + apex_mean + exit_mean) / 3 if (entry_mean + apex_mean + exit_mean) > 0 else 1.0
        avg_std = (entry_std + apex_std + exit_std) / 3
        consistency_pct = max(0, 100 - (avg_std / avg_speed * 100)) if avg_speed > 0 else 0

        consistency_scores[corner_num] = {
            "entry_speed_std": entry_std,
            "entry_speed_mean": entry_mean,
            "apex_speed_std": apex_std,
            "apex_speed_mean": apex_mean,
            "exit_speed_std": exit_std,
            "exit_speed_mean": exit_mean,
            "brake_point_std": brake_std,
            "time_std": time_std,
            "consistency_score": consistency_pct,
            "num_laps": len(corners)
        }

    # Overall session consistency (average of all corners)
    all_consistency_scores = [c["consistency_score"] for c in consistency_scores.values()]
    overall_consistency = np.mean(all_consistency_scores) if all_consistency_scores else 0.0

    # Identify most/least consistent corners
    sorted_corners = sorted(
        consistency_scores.items(),
        key=lambda x: x[1]["consistency_score"]
    )

    # Least consistent (lowest scores) - top 3 improvement areas
    least_consistent = sorted_corners[:3] if len(sorted_corners) >= 3 else sorted_corners

    # Most consistent (highest scores) - top 3 strengths
    most_consistent = sorted_corners[-3:][::-1] if len(sorted_corners) >= 3 else sorted_corners[::-1]

    # Count unique laps
    unique_laps = set()
    for corner in corner_data_list:
        lap_num = corner.get('lap_number', corner.get('lap', 'Unknown'))
        unique_laps.add(lap_num)
    total_laps = len(unique_laps)

    return {
        "overall_consistency": overall_consistency,
        "by_corner": consistency_scores,
        "most_consistent": most_consistent,
        "least_consistent": least_consistent,
        "total_laps": total_laps
    }


def get_consistency_level(score: float) -> Tuple[str, str, str]:
    """
    Get consistency level description based on score

    Args:
        score: Consistency score 0-100

    Returns:
        Tuple of (level_name, color, icon)
    """
    if score >= 90:
        return ("EXCELLENT", "success", "fas fa-trophy")
    elif score >= 75:
        return ("GOOD", "primary", "fas fa-check-circle")
    elif score >= 60:
        return ("FAIR", "warning", "fas fa-exclamation-triangle")
    else:
        return ("NEEDS WORK", "danger", "fas fa-times-circle")


def get_consistency_coaching(score: float) -> str:
    """
    Generate coaching message based on consistency score

    Args:
        score: Consistency score 0-100

    Returns:
        Coaching message string
    """
    if score >= 90:
        return (
            "Excellent consistency! You're repeating your technique lap after lap. "
            "Your variation is minimal, showing great car control and racecraft. "
            "Focus on extracting more speed while maintaining this consistency level."
        )
    elif score >= 75:
        return (
            "Good consistency overall. You have a solid foundation with repeatable performance. "
            "Work on the corners where you have higher variation - use reference points "
            "(brake markers, turn-in points, apex curbing) to improve repeatability. "
            "Consistency in these areas will unlock more pace."
        )
    elif score >= 60:
        return (
            "Fair consistency with room for improvement. Focus on one corner at a time. "
            "Pick specific, visual reference points for brake, turn-in, and throttle application. "
            "Remember: Consistency comes before speed. Master the fundamentals first, "
            "then gradually increase pace while maintaining repeatability."
        )
    else:
        return (
            "Consistency needs work - this is currently limiting your pace potential. "
            "Pick ONE corner per session to master. Identify exact reference points: "
            "where you brake (distance board, marker), where you turn in (cone, curb), "
            "and where you apply throttle (apex, track-out point). Practice until you can "
            "hit the same speeds within 1-2 km/h lap after lap. Once consistent, speed will follow."
        )


def format_consistency_analysis(consistency_data: Dict) -> str:
    """
    Format consistency analysis as readable text summary

    Args:
        consistency_data: Output from calculate_consistency_metrics()

    Returns:
        Formatted summary string
    """
    score = consistency_data.get('overall_consistency', 0)
    level, _, _ = get_consistency_level(score)
    total_laps = consistency_data.get('total_laps', 0)

    summary = f"Overall Consistency: {score:.1f}/100 ({level})\n"
    summary += f"Analyzed {total_laps} laps\n\n"

    # Most consistent corners
    most_consistent = consistency_data.get('most_consistent', [])
    if most_consistent:
        summary += "Strengths (Most Consistent Corners):\n"
        for corner_num, metrics in most_consistent:
            score = metrics['consistency_score']
            apex_std = metrics['apex_speed_std']
            summary += f"  - Turn {corner_num}: {score:.0f}% (±{apex_std:.1f} km/h)\n"

    summary += "\n"

    # Least consistent corners
    least_consistent = consistency_data.get('least_consistent', [])
    if least_consistent:
        summary += "Focus Areas (Least Consistent Corners):\n"
        for corner_num, metrics in least_consistent:
            score = metrics['consistency_score']
            entry_std = metrics['entry_speed_std']
            apex_std = metrics['apex_speed_std']
            exit_std = metrics['exit_speed_std']
            summary += (
                f"  - Turn {corner_num}: {score:.0f}% "
                f"(±{entry_std:.1f}/±{apex_std:.1f}/±{exit_std:.1f} km/h E/A/X)\n"
            )

    return summary
