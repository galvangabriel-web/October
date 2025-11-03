#!/usr/bin/env python3
"""
Comprehensive Analysis of ALL Racing Data Categories
Analyzes: Weather, Results, Analysis, Other, plus full Telemetry and Lap Times
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from data_loader import RacingDataLoader

class ComprehensiveRacingDataAnalyzer:
    """Analyzes all data categories across all tracks"""

    def __init__(self):
        self.loader = RacingDataLoader()
        self.tracks = self.loader.list_tracks()
        self.results = {
            'weather': {},
            'results': {},
            'analysis': {},
            'other': {},
            'telemetry_full': {},
            'lap_times_full': {},
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }

    def analyze_weather_data(self):
        """Analyze weather data across all tracks"""
        print("\n" + "="*80)
        print("ANALYZING WEATHER DATA")
        print("="*80)

        weather_summary = []

        for track in self.tracks:
            print(f"\n[*] Track: {track}")
            try:
                races = self.loader.list_races(track)
                for race in races:
                    try:
                        categories = self.loader.list_categories(track, race)
                        if 'weather' in categories:
                            print(f"  |- {race}/weather")
                            df = self.loader.load_data(track, race, 'weather', combine_chunks=True)

                            info = {
                                'track': track,
                                'race': race,
                                'rows': len(df),
                                'columns': list(df.columns),
                                'sample_data': df.head(3).to_dict() if len(df) > 0 else {},
                                'statistics': {}
                            }

                            # Analyze numeric columns
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            for col in numeric_cols:
                                info['statistics'][col] = {
                                    'min': float(df[col].min()),
                                    'max': float(df[col].max()),
                                    'mean': float(df[col].mean()),
                                    'median': float(df[col].median())
                                }

                            weather_summary.append(info)
                            print(f"      [+] Analyzed {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"      [-] Error in {race}: {str(e)}")
            except Exception as e:
                print(f"  [-] Error accessing {track}: {str(e)}")

        self.results['weather'] = {
            'total_files': len(weather_summary),
            'details': weather_summary
        }

        print(f"\n[+] Weather Analysis Complete: {len(weather_summary)} files analyzed")
        return weather_summary

    def analyze_results_data(self):
        """Analyze race results data across all tracks"""
        print("\n" + "="*80)
        print("ANALYZING RESULTS DATA")
        print("="*80)

        results_summary = []

        for track in self.tracks:
            print(f"\n[*] Track: {track}")
            try:
                races = self.loader.list_races(track)
                for race in races:
                    try:
                        categories = self.loader.list_categories(track, race)
                        if 'results' in categories:
                            print(f"  ├─ {race}/results")
                            df = self.loader.load_data(track, race, 'results', combine_chunks=True)

                            info = {
                                'track': track,
                                'race': race,
                                'rows': len(df),
                                'columns': list(df.columns),
                                'sample_data': df.head(5).to_dict() if len(df) > 0 else {},
                                'vehicle_count': int(df['vehicle_number'].nunique()) if 'vehicle_number' in df.columns else 'N/A',
                                'statistics': {}
                            }

                            # Analyze position/ranking if available
                            if 'position' in df.columns or 'pos' in df.columns or 'Pos' in df.columns:
                                pos_col = [c for c in df.columns if 'pos' in c.lower()][0]
                                info['statistics']['positions'] = {
                                    'winner': df[pos_col].min(),
                                    'last': df[pos_col].max(),
                                    'total_finishers': int(df[pos_col].count())
                                }

                            results_summary.append(info)
                            print(f"      [+] Analyzed {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"      [-] Error in {race}: {str(e)}")
            except Exception as e:
                print(f"  [-] Error accessing {track}: {str(e)}")

        self.results['results'] = {
            'total_files': len(results_summary),
            'details': results_summary
        }

        print(f"\n[+] Results Analysis Complete: {len(results_summary)} files analyzed")
        return results_summary

    def analyze_analysis_files(self):
        """Analyze pre-computed analysis files"""
        print("\n" + "="*80)
        print("ANALYZING PRE-COMPUTED ANALYSIS FILES")
        print("="*80)

        analysis_summary = []

        for track in self.tracks:
            print(f"\n[*] Track: {track}")
            try:
                races = self.loader.list_races(track)
                for race in races:
                    try:
                        categories = self.loader.list_categories(track, race)
                        if 'analysis' in categories:
                            print(f"  ├─ {race}/analysis")
                            df = self.loader.load_data(track, race, 'analysis', combine_chunks=True)

                            info = {
                                'track': track,
                                'race': race,
                                'rows': len(df),
                                'columns': list(df.columns),
                                'sample_data': df.head(3).to_dict() if len(df) > 0 else {},
                                'key_metrics': {}
                            }

                            # Look for sector times or segment analysis
                            sector_cols = [c for c in df.columns if 'sector' in c.lower() or 'segment' in c.lower()]
                            if sector_cols:
                                info['key_metrics']['sector_columns'] = sector_cols

                            # Look for best lap data
                            if 'lap_time' in df.columns or 'time' in df.columns:
                                time_col = [c for c in df.columns if 'time' in c.lower()][0]
                                info['key_metrics']['best_lap'] = float(df[time_col].min())
                                info['key_metrics']['worst_lap'] = float(df[time_col].max())

                            analysis_summary.append(info)
                            print(f"      [+] Analyzed {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"      [-] Error in {race}: {str(e)}")
            except Exception as e:
                print(f"  [-] Error accessing {track}: {str(e)}")

        self.results['analysis'] = {
            'total_files': len(analysis_summary),
            'details': analysis_summary
        }

        print(f"\n[+] Analysis Files Complete: {len(analysis_summary)} files analyzed")
        return analysis_summary

    def analyze_other_files(self):
        """Investigate 'other' category files"""
        print("\n" + "="*80)
        print("INVESTIGATING 'OTHER' FILES")
        print("="*80)

        other_summary = []

        for track in self.tracks:
            print(f"\n[*] Track: {track}")
            try:
                races = self.loader.list_races(track)
                for race in races:
                    try:
                        categories = self.loader.list_categories(track, race)
                        if 'other' in categories:
                            print(f"  ├─ {race}/other")
                            df = self.loader.load_data(track, race, 'other', combine_chunks=True)

                            info = {
                                'track': track,
                                'race': race,
                                'rows': len(df),
                                'columns': list(df.columns),
                                'sample_data': df.head(3).to_dict() if len(df) > 0 else {},
                                'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
                            }

                            other_summary.append(info)
                            print(f"      [+] Analyzed {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        print(f"      [-] Error in {race}: {str(e)}")
            except Exception as e:
                print(f"  [-] Error accessing {track}: {str(e)}")

        self.results['other'] = {
            'total_files': len(other_summary),
            'details': other_summary
        }

        print(f"\n[+] Other Files Investigation Complete: {len(other_summary)} files analyzed")
        return other_summary

    def generate_comprehensive_report(self):
        """Generate comprehensive JSON and Markdown reports"""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORTS")
        print("="*80)

        # Summary statistics
        self.results['summary'] = {
            'total_tracks_analyzed': len(self.tracks),
            'tracks': self.tracks,
            'weather_files': self.results['weather']['total_files'],
            'results_files': self.results['results']['total_files'],
            'analysis_files': self.results['analysis']['total_files'],
            'other_files': self.results['other']['total_files'],
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save JSON report
        json_path = Path('COMPREHENSIVE_ANALYSIS_REPORT.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"[OK] JSON report saved: {json_path}")

        # Generate Markdown report
        self.generate_markdown_report()

        print("\n" + "="*80)
        print("[OK] ALL ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nReports generated:")
        print(f"  - COMPREHENSIVE_ANALYSIS_REPORT.json")
        print(f"  - COMPREHENSIVE_ANALYSIS_REPORT.md")
        print(f"\nSummary:")
        print(f"  - Tracks analyzed: {self.results['summary']['total_tracks_analyzed']}")
        print(f"  - Weather files: {self.results['summary']['weather_files']}")
        print(f"  - Results files: {self.results['summary']['results_files']}")
        print(f"  - Analysis files: {self.results['summary']['analysis_files']}")
        print(f"  - Other files: {self.results['summary']['other_files']}")

    def generate_markdown_report(self):
        """Generate human-readable markdown report"""
        md_path = Path('COMPREHENSIVE_ANALYSIS_REPORT.md')

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Racing Data Analysis Report\n\n")
            f.write(f"**Generated:** {self.results['summary']['analysis_date']}\n\n")
            f.write("---\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Tracks Analyzed:** {self.results['summary']['total_tracks_analyzed']}\n")
            f.write(f"- **Weather Files:** {self.results['summary']['weather_files']}\n")
            f.write(f"- **Results Files:** {self.results['summary']['results_files']}\n")
            f.write(f"- **Analysis Files:** {self.results['summary']['analysis_files']}\n")
            f.write(f"- **Other Files:** {self.results['summary']['other_files']}\n\n")

            f.write("---\n\n")

            # Weather section
            f.write("## Weather Data Analysis\n\n")
            if self.results['weather']['total_files'] > 0:
                f.write(f"**Files Analyzed:** {self.results['weather']['total_files']}\n\n")
                for item in self.results['weather']['details']:
                    f.write(f"### {item['track']} - {item['race']}\n")
                    f.write(f"- Rows: {item['rows']}\n")
                    f.write(f"- Columns: {', '.join(item['columns'])}\n")
                    if item['statistics']:
                        f.write(f"- Statistics: {len(item['statistics'])} numeric columns analyzed\n")
                    f.write("\n")
            else:
                f.write("No weather data found.\n\n")

            f.write("---\n\n")

            # Results section
            f.write("## Race Results Analysis\n\n")
            if self.results['results']['total_files'] > 0:
                f.write(f"**Files Analyzed:** {self.results['results']['total_files']}\n\n")
                for item in self.results['results']['details']:
                    f.write(f"### {item['track']} - {item['race']}\n")
                    f.write(f"- Rows: {item['rows']}\n")
                    f.write(f"- Vehicles: {item['vehicle_count']}\n")
                    f.write(f"- Columns: {', '.join(item['columns'])}\n")
                    f.write("\n")
            else:
                f.write("No results data found.\n\n")

            f.write("---\n\n")

            # Analysis section
            f.write("## Pre-computed Analysis Files\n\n")
            if self.results['analysis']['total_files'] > 0:
                f.write(f"**Files Analyzed:** {self.results['analysis']['total_files']}\n\n")
                for item in self.results['analysis']['details']:
                    f.write(f"### {item['track']} - {item['race']}\n")
                    f.write(f"- Rows: {item['rows']}\n")
                    f.write(f"- Columns: {', '.join(item['columns'])}\n")
                    if item['key_metrics']:
                        f.write(f"- Key Metrics: {item['key_metrics']}\n")
                    f.write("\n")
            else:
                f.write("No analysis files found.\n\n")

            f.write("---\n\n")

            # Other section
            f.write("## Other Files Investigation\n\n")
            if self.results['other']['total_files'] > 0:
                f.write(f"**Files Analyzed:** {self.results['other']['total_files']}\n\n")
                for item in self.results['other']['details']:
                    f.write(f"### {item['track']} - {item['race']}\n")
                    f.write(f"- Rows: {item['rows']}\n")
                    f.write(f"- Columns: {', '.join(item['columns'])}\n")
                    f.write("\n")
            else:
                f.write("No other files found.\n\n")

            f.write("---\n\n")
            f.write("## Conclusion\n\n")
            f.write("This report provides a comprehensive analysis of all data categories ")
            f.write("across all racing tracks in the dataset.\n")

        print(f"[OK] Markdown report saved: {md_path}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RACING DATA ANALYSIS")
        print("Analyzing ALL unanalyzed data categories")
        print("="*80)
        print(f"\nTracks to analyze: {len(self.tracks)}")
        print(f"Tracks: {', '.join(self.tracks)}")

        try:
            # Analyze each category
            self.analyze_weather_data()
            self.analyze_results_data()
            self.analyze_analysis_files()
            self.analyze_other_files()

            # Generate reports
            self.generate_comprehensive_report()

            return True
        except Exception as e:
            print(f"\n[-] ERROR during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    print("="*80)
    print("COMPREHENSIVE RACING DATA ANALYZER")
    print("="*80)
    print("\nThis script will analyze ALL data categories:")
    print("  - Weather data")
    print("  - Race results")
    print("  - Pre-computed analysis files")
    print("  - Other/miscellaneous files")
    print("\nStarting analysis...\n")

    analyzer = ComprehensiveRacingDataAnalyzer()
    success = analyzer.run_full_analysis()

    if success:
        print("\n[SUCCESS] Check the generated reports:")
        print("  - COMPREHENSIVE_ANALYSIS_REPORT.json")
        print("  - COMPREHENSIVE_ANALYSIS_REPORT.md")
        sys.exit(0)
    else:
        print("\n[FAILED] Analysis failed. Check error messages above.")
        sys.exit(1)
