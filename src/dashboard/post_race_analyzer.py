"""
Post-Race Analysis Logic
========================

Analyzes prediction results to generate insights, detect anomalies, and provide
coaching recommendations.

Key Features:
- Session statistics calculation
- Anomaly detection (laps with large prediction errors)
- Cause inference (why the lap was slow)
- Coaching recommendations generation

Usage:
    analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
    analysis = analyzer.analyze_session(predictions_df)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SessionStatistics:
    """Container for session-level statistics"""
    total_laps: int
    avg_lap_time: float
    best_lap: float
    worst_lap: float
    std_lap_time: float
    model_mae: float
    model_rmse: float
    model_r2: float
    anomaly_count: int
    avg_error: float
    max_error: float
    consistency_rating: str  # Excellent, Good, Fair, Poor


class PostRaceAnalyzer:
    """Analyze post-race predictions and generate coaching insights"""

    def __init__(self, anomaly_threshold: float = 2.5):
        """
        Initialize analyzer

        Args:
            anomaly_threshold: Error threshold (seconds) for anomaly detection
                Default 2.5s means errors > 2.5s are flagged as anomalies
        """
        self.anomaly_threshold = anomaly_threshold

    def analyze_session(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Comprehensive session analysis

        Args:
            predictions_df: DataFrame with columns:
                - lap_number
                - vehicle_number
                - actual (lap time)
                - predicted (lap time)
                - error (actual - predicted)
                - abs_error

        Returns:
            Dict with keys:
                - statistics: SessionStatistics object
                - anomalies: DataFrame of problem laps
                - recommendations: List[str] of coaching suggestions
                - performance_trend: Dict with trend analysis
        """
        stats = self._calculate_statistics(predictions_df)
        anomalies = self._detect_anomalies(predictions_df)
        recommendations = self._generate_recommendations(predictions_df, anomalies)
        trend = self._analyze_performance_trend(predictions_df)

        return {
            'statistics': stats,
            'anomalies': anomalies,
            'recommendations': recommendations,
            'performance_trend': trend
        }

    def _calculate_statistics(self, df: pd.DataFrame) -> SessionStatistics:
        """
        Calculate comprehensive session statistics

        Args:
            df: Predictions DataFrame

        Returns:
            SessionStatistics object
        """
        # Basic lap statistics
        total_laps = len(df)
        avg_lap_time = df['actual'].mean()
        best_lap = df['actual'].min()
        worst_lap = df['actual'].max()
        std_lap_time = df['actual'].std()

        # Model performance
        model_mae = df['abs_error'].mean()
        model_rmse = np.sqrt((df['error'] ** 2).mean())

        # R² calculation
        ss_res = (df['error'] ** 2).sum()
        ss_tot = ((df['actual'] - df['actual'].mean()) ** 2).sum()
        model_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Anomaly detection
        anomaly_count = (df['abs_error'] > self.anomaly_threshold).sum()
        avg_error = df['error'].mean()
        max_error = df['abs_error'].max()

        # Consistency rating
        consistency_rating = self._rate_consistency(std_lap_time, avg_lap_time)

        return SessionStatistics(
            total_laps=total_laps,
            avg_lap_time=avg_lap_time,
            best_lap=best_lap,
            worst_lap=worst_lap,
            std_lap_time=std_lap_time,
            model_mae=model_mae,
            model_rmse=model_rmse,
            model_r2=model_r2,
            anomaly_count=anomaly_count,
            avg_error=avg_error,
            max_error=max_error,
            consistency_rating=consistency_rating
        )

    def _rate_consistency(self, std: float, mean: float) -> str:
        """
        Rate driver consistency based on lap time standard deviation

        Args:
            std: Standard deviation of lap times
            mean: Mean lap time

        Returns:
            Rating string: Excellent, Good, Fair, or Poor
        """
        coefficient_of_variation = (std / mean) * 100  # Percentage

        if coefficient_of_variation < 1.0:
            return "Excellent"
        elif coefficient_of_variation < 2.0:
            return "Good"
        elif coefficient_of_variation < 3.0:
            return "Fair"
        else:
            return "Poor"

    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect problem laps with large prediction errors

        Args:
            df: Predictions DataFrame

        Returns:
            DataFrame with anomalous laps, sorted by error magnitude
        """
        anomalies = df[df['abs_error'] > self.anomaly_threshold].copy()

        if len(anomalies) == 0:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'lap_number', 'vehicle_number', 'actual', 'predicted',
                'error', 'abs_error', 'likely_cause', 'severity'
            ])

        # Add cause inference
        anomalies['likely_cause'] = anomalies['error'].apply(self._infer_cause)

        # Add severity classification
        anomalies['severity'] = anomalies['abs_error'].apply(self._classify_severity)

        # Sort by error magnitude
        anomalies = anomalies.sort_values('abs_error', ascending=False)

        return anomalies[[
            'lap_number', 'vehicle_number', 'actual', 'predicted',
            'error', 'abs_error', 'likely_cause', 'severity'
        ]]

    def _infer_cause(self, error: float) -> str:
        """
        Infer likely cause of anomaly based on error magnitude and direction

        Args:
            error: Prediction error (actual - predicted)

        Returns:
            String describing likely cause
        """
        if error > 7.0:
            return "⚠️ Incident/Collision (very slow)"
        elif error > 5.0:
            return "⚠️ Traffic/Yellow Flag"
        elif error > 3.5:
            return "⚠️ Major Mistake (off-track/spin)"
        elif error > 2.5:
            return "⚠️ Driving Error (corner speed loss)"
        elif error < -3.0:
            return "✅ Exceptional Lap (much faster)"
        elif error < -2.0:
            return "✅ Strong Performance"
        else:
            return "ℹ️ Normal Variance"

    def _classify_severity(self, abs_error: float) -> str:
        """
        Classify anomaly severity

        Args:
            abs_error: Absolute prediction error

        Returns:
            Severity level: Critical, High, Medium, or Low
        """
        if abs_error > 7.0:
            return "Critical"
        elif abs_error > 5.0:
            return "High"
        elif abs_error > 3.5:
            return "Medium"
        else:
            return "Low"

    def _generate_recommendations(self,
                                 df: pd.DataFrame,
                                 anomalies: pd.DataFrame) -> List[str]:
        """
        Generate coaching recommendations based on session data

        Args:
            df: Full predictions DataFrame
            anomalies: Anomalous laps DataFrame

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # 1. Consistency analysis
        std = df['actual'].std()
        avg = df['actual'].mean()
        cv = (std / avg) * 100

        if cv > 3.0:
            recommendations.append(
                f"🎯 **Consistency Focus**: Lap time variability is high ({cv:.1f}%). "
                "Work on repeatable braking points and throttle application."
            )
        elif cv < 1.0:
            recommendations.append(
                "✅ **Excellent Consistency**: Very consistent lap times. "
                "Focus on extracting more pace while maintaining consistency."
            )

        # 2. Anomaly frequency
        anomaly_rate = len(anomalies) / len(df) * 100

        if anomaly_rate > 15:
            recommendations.append(
                f"⚠️ **High Error Rate**: {anomaly_rate:.1f}% of laps have large errors. "
                "Review video footage for recurring mistakes."
            )
        elif anomaly_rate > 0:
            recommendations.append(
                f"ℹ️ **Anomaly Review**: {len(anomalies)} laps flagged. "
                f"Check laps: {', '.join(map(str, anomalies['lap_number'].head(5).tolist()))}."
            )

        # 3. Performance trend analysis
        if len(df) >= 10:
            first_half = df.iloc[:len(df)//2]['actual'].mean()
            second_half = df.iloc[len(df)//2:]['actual'].mean()
            delta = second_half - first_half

            if delta > 1.0:
                recommendations.append(
                    f"📉 **Performance Degradation**: Lap times increased by {delta:.2f}s "
                    "in second half. Check tire pressures and driving technique."
                )
            elif delta < -1.0:
                recommendations.append(
                    f"📈 **Strong Improvement**: Lap times improved by {abs(delta):.2f}s "
                    "in second half. Good adaptation and learning."
                )

        # 4. Model accuracy check
        mae = df['abs_error'].mean()

        if mae > 3.0:
            recommendations.append(
                f"🤖 **Data Quality Issue**: Model MAE is {mae:.2f}s (expected <2s). "
                "Verify telemetry data quality and sensor calibration."
            )
        elif mae < 1.5:
            recommendations.append(
                f"✅ **Excellent Model Accuracy**: MAE = {mae:.2f}s. "
                "Predictions are highly reliable for this session."
            )

        # 5. Best lap analysis
        best_lap_idx = df['actual'].idxmin()
        best_lap = df.loc[best_lap_idx]

        recommendations.append(
            f"🏆 **Best Lap**: Lap {best_lap['lap_number']} - {best_lap['actual']:.3f}s. "
            "Analyze telemetry to understand what made this lap faster."
        )

        # 6. Specific anomaly coaching
        if len(anomalies) > 0:
            critical_anomalies = anomalies[anomalies['severity'].isin(['Critical', 'High'])]

            if len(critical_anomalies) > 0:
                lap_nums = ', '.join(map(str, critical_anomalies['lap_number'].head(3).tolist()))
                recommendations.append(
                    f"🔍 **Critical Laps to Review**: Laps {lap_nums} had major issues. "
                    "Watch onboard video and compare telemetry to best lap."
                )

        # 7. If no recommendations yet, add generic one
        if not recommendations:
            recommendations.append(
                "✅ **Solid Session**: No major issues detected. "
                "Continue building consistency and confidence."
            )

        return recommendations

    def _analyze_performance_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyze performance trend throughout session

        Args:
            df: Predictions DataFrame

        Returns:
            Dict with trend analysis:
                - direction: 'improving', 'degrading', or 'stable'
                - rate: Seconds per lap change
                - confidence: High, Medium, or Low
        """
        if len(df) < 5:
            return {
                'direction': 'unknown',
                'rate': 0.0,
                'confidence': 'Low',
                'message': 'Insufficient data for trend analysis'
            }

        # Linear regression on lap times
        x = np.arange(len(df))
        y = df['actual'].values

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Classify trend
        if slope < -0.05:
            direction = 'improving'
            message = f"Improving by {abs(slope):.3f}s per lap"
        elif slope > 0.05:
            direction = 'degrading'
            message = f"Degrading by {slope:.3f}s per lap"
        else:
            direction = 'stable'
            message = "Lap times stable throughout session"

        # Calculate R² to assess trend strength
        y_pred = np.polyfit(x, y, 1)[0] * x + np.polyfit(x, y, 1)[1]
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))

        if r2 > 0.5:
            confidence = 'High'
        elif r2 > 0.2:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        return {
            'direction': direction,
            'rate': slope,
            'confidence': confidence,
            'message': message,
            'r2': r2
        }

    def generate_summary_text(self, analysis: Dict) -> str:
        """
        Generate human-readable summary text

        Args:
            analysis: Output from analyze_session()

        Returns:
            Multi-line summary string
        """
        stats = analysis['statistics']
        anomalies = analysis['anomalies']
        trend = analysis['performance_trend']

        summary = f"""
📊 **SESSION SUMMARY**

🏁 **Lap Statistics:**
   • Total Laps: {stats.total_laps}
   • Average: {stats.avg_lap_time:.3f}s
   • Best: {stats.best_lap:.3f}s
   • Worst: {stats.worst_lap:.3f}s
   • Std Dev: {stats.std_lap_time:.3f}s
   • Consistency: {stats.consistency_rating}

🤖 **Model Performance:**
   • MAE: {stats.model_mae:.3f}s
   • RMSE: {stats.model_rmse:.3f}s
   • R²: {stats.model_r2:.1%}

⚠️ **Anomalies:**
   • Count: {stats.anomaly_count} laps
   • Largest Error: {stats.max_error:.3f}s

📈 **Performance Trend:**
   • Direction: {trend['direction'].capitalize()}
   • {trend['message']}
   • Confidence: {trend['confidence']}

💡 **Recommendations:**
{chr(10).join(f'   • {rec}' for rec in analysis['recommendations'])}
"""
        return summary


if __name__ == "__main__":
    """Example usage"""

    # Create sample predictions data
    sample_data = pd.DataFrame({
        'lap_number': range(1, 21),
        'vehicle_number': [5] * 20,
        'actual': [120.5, 119.8, 119.2, 118.9, 119.1, 119.3, 118.7, 119.0,
                   119.5, 124.2, 119.4, 119.1, 119.6, 119.8, 120.1, 119.7,
                   120.3, 120.8, 121.2, 121.5],
        'predicted': [120.0] * 20,
        'error': [0.5, -0.2, -0.8, -1.1, -0.9, -0.7, -1.3, -1.0, -0.5, 4.2,
                  -0.6, -0.9, -0.4, -0.2, 0.1, -0.3, 0.3, 0.8, 1.2, 1.5],
        'abs_error': [0.5, 0.2, 0.8, 1.1, 0.9, 0.7, 1.3, 1.0, 0.5, 4.2,
                      0.6, 0.9, 0.4, 0.2, 0.1, 0.3, 0.3, 0.8, 1.2, 1.5]
    })

    # Analyze session
    analyzer = PostRaceAnalyzer(anomaly_threshold=2.5)
    analysis = analyzer.analyze_session(sample_data)

    # Print summary
    print(analyzer.generate_summary_text(analysis))

    # Print anomaly details
    if len(analysis['anomalies']) > 0:
        print("\n🔍 **ANOMALY DETAILS:**")
        print(analysis['anomalies'].to_string(index=False))
