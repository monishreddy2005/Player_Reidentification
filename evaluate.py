#!/usr/bin/env python3
"""
Evaluation script for Player Re-identification System

This script provides evaluation metrics and analysis tools for assessing
the performance of the player tracking system.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
import os


class TrackingEvaluator:
    """
    Evaluates player tracking performance using various metrics.
    """
    
    def __init__(self, results_file: str):
        """
        Initialize evaluator with tracking results.
        
        Args:
            results_file: Path to JSON file with tracking results
        """
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.total_frames = len(self.results)
        self.all_players = self._extract_all_players()
    
    def _extract_all_players(self) -> Dict[int, List[int]]:
        """Extract all player IDs and their appearance frames."""
        players = {}
        
        for frame_idx, frame_data in enumerate(self.results):
            for player in frame_data:
                player_id = player['id']
                if player_id not in players:
                    players[player_id] = []
                players[player_id].append(frame_idx)
        
        return players
    
    def calculate_detection_rate(self) -> float:
        """Calculate overall detection rate."""
        total_detections = sum(len(frame_data) for frame_data in self.results)
        frames_with_detections = sum(1 for frame_data in self.results if len(frame_data) > 0)
        
        if self.total_frames == 0:
            return 0.0
        
        return frames_with_detections / self.total_frames
    
    def calculate_id_consistency(self) -> Dict[str, float]:
        """Calculate ID consistency metrics."""
        metrics = {}
        
        # Average tracking duration
        durations = [len(frames) for frames in self.all_players.values()]
        metrics['avg_tracking_duration'] = np.mean(durations) if durations else 0
        metrics['max_tracking_duration'] = max(durations) if durations else 0
        
        # ID persistence (how long IDs last on average)
        total_gaps = 0
        total_sequences = 0
        
        for player_id, frames in self.all_players.items():
            if len(frames) < 2:
                continue
                
            # Calculate gaps in tracking
            sorted_frames = sorted(frames)
            gaps = []
            for i in range(1, len(sorted_frames)):
                gap = sorted_frames[i] - sorted_frames[i-1] - 1
                if gap > 0:
                    gaps.append(gap)
            
            total_gaps += sum(gaps)
            total_sequences += len(gaps) if gaps else 1
        
        metrics['avg_gap_size'] = total_gaps / max(1, total_sequences)
        metrics['total_unique_ids'] = len(self.all_players)
        
        return metrics
    
    def calculate_temporal_consistency(self) -> float:
        """Calculate temporal consistency score."""
        consistency_scores = []
        
        for player_id, frames in self.all_players.items():
            if len(frames) < 2:
                continue
            
            # Calculate consistency as ratio of actual frames to span
            min_frame = min(frames)
            max_frame = max(frames)
            span = max_frame - min_frame + 1
            consistency = len(frames) / span
            consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0
    
    def detect_id_switches(self) -> List[Dict]:
        """Detect potential ID switches based on spatial jumps."""
        switches = []
        
        for frame_idx in range(1, len(self.results)):
            prev_frame = {p['id']: p['center'] for p in self.results[frame_idx - 1]}
            curr_frame = {p['id']: p['center'] for p in self.results[frame_idx]}
            
            for player_id in curr_frame:
                if player_id in prev_frame:
                    # Calculate movement distance
                    prev_pos = prev_frame[player_id]
                    curr_pos = curr_frame[player_id]
                    distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                                     (curr_pos[1] - prev_pos[1])**2)
                    
                    # Flag large jumps as potential ID switches
                    if distance > 200:  # Threshold for suspicious movement
                        switches.append({
                            'frame': frame_idx,
                            'player_id': player_id,
                            'distance': distance,
                            'prev_pos': prev_pos,
                            'curr_pos': curr_pos
                        })
        
        return switches
    
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            'general_stats': {
                'total_frames': self.total_frames,
                'total_detections': sum(len(frame_data) for frame_data in self.results),
                'avg_players_per_frame': np.mean([len(frame_data) for frame_data in self.results]),
                'max_players_in_frame': max([len(frame_data) for frame_data in self.results]),
                'detection_rate': self.calculate_detection_rate()
            },
            'tracking_metrics': self.calculate_id_consistency(),
            'temporal_consistency': self.calculate_temporal_consistency(),
            'potential_id_switches': len(self.detect_id_switches())
        }
        
        return report
    
    def plot_tracking_timeline(self, output_path: str = "tracking_timeline.png"):
        """Plot timeline showing player appearances."""
        if not self.all_players:
            print("No players to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        player_ids = sorted(self.all_players.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(player_ids)))
        
        for i, (player_id, frames) in enumerate([(pid, self.all_players[pid]) for pid in player_ids]):
            y_pos = i
            for frame in frames:
                plt.scatter(frame, y_pos, c=[colors[i]], s=20, alpha=0.7)
        
        plt.xlabel('Frame Number')
        plt.ylabel('Player ID')
        plt.title('Player Tracking Timeline')
        plt.yticks(range(len(player_ids)), [f"Player {pid}" for pid in player_ids])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Timeline plot saved to: {output_path}")
    
    def plot_detection_statistics(self, output_path: str = "detection_stats.png"):
        """Plot detection statistics over time."""
        frame_counts = [len(frame_data) for frame_data in self.results]
        
        plt.figure(figsize=(12, 6))
        
        # Plot detections per frame
        plt.subplot(2, 1, 1)
        plt.plot(frame_counts, linewidth=2)
        plt.title('Detections per Frame')
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Players Detected')
        plt.grid(True, alpha=0.3)
        
        # Plot histogram of detection counts
        plt.subplot(2, 1, 2)
        plt.hist(frame_counts, bins=max(1, max(frame_counts)), alpha=0.7, edgecolor='black')
        plt.title('Distribution of Detection Counts')
        plt.xlabel('Number of Players')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detection statistics plot saved to: {output_path}")
    
    def export_evaluation_report(self, output_path: str = "evaluation_report.json"):
        """Export detailed evaluation report."""
        report = self.generate_summary_report()
        
        # Add detailed player information
        report['player_details'] = {}
        for player_id, frames in self.all_players.items():
            report['player_details'][f'player_{player_id}'] = {
                'total_appearances': len(frames),
                'first_frame': min(frames),
                'last_frame': max(frames),
                'tracking_span': max(frames) - min(frames) + 1,
                'consistency_ratio': len(frames) / (max(frames) - min(frames) + 1)
            }
        
        # Add ID switch details
        switches = self.detect_id_switches()
        report['id_switches'] = switches
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report exported to: {output_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Player Re-identification Performance")
    parser.add_argument("--results", required=True, help="Path to tracking results JSON file")
    parser.add_argument("--output-dir", default="evaluation_output", 
                       help="Directory for output files")
    parser.add_argument("--plots", action="store_true", 
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = TrackingEvaluator(args.results)
    
    # Generate summary report
    print("Generating evaluation report...")
    report = evaluator.generate_summary_report()
    
    # Print summary to console
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"Total frames processed: {report['general_stats']['total_frames']}")
    print(f"Total detections: {report['general_stats']['total_detections']}")
    print(f"Average players per frame: {report['general_stats']['avg_players_per_frame']:.2f}")
    print(f"Detection rate: {report['general_stats']['detection_rate']:.2%}")
    print(f"Unique player IDs: {report['tracking_metrics']['total_unique_ids']}")
    print(f"Average tracking duration: {report['tracking_metrics']['avg_tracking_duration']:.1f} frames")
    print(f"Temporal consistency: {report['temporal_consistency']:.2%}")
    print(f"Potential ID switches: {report['potential_id_switches']}")
    
    # Export detailed report
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    evaluator.export_evaluation_report(report_path)
    
    # Generate plots if requested
    if args.plots:
        print("\nGenerating visualization plots...")
        timeline_path = os.path.join(args.output_dir, "tracking_timeline.png")
        stats_path = os.path.join(args.output_dir, "detection_stats.png")
        
        evaluator.plot_tracking_timeline(timeline_path)
        evaluator.plot_detection_statistics(stats_path)
    
    print(f"\nEvaluation complete! Output saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())