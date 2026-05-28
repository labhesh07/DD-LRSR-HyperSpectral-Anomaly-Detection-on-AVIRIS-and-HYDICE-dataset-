# coding=utf-8
"""
Result Export Module
Exports results in multiple formats (JSON, CSV, images)
"""
import numpy as np
import json
import csv
import os
from datetime import datetime
import scipy.io as sio


class ResultExporter:
    """Export results in various formats"""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def export_metrics(self, metrics_dict, filename=None):
        """Export metrics to JSON"""
        if filename is None:
            filename = f'metrics_{self.timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert numpy types to native Python types
        export_dict = {}
        for key, value in metrics_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                export_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                export_dict[key] = value.tolist()
            else:
                export_dict[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(export_dict, f, indent=4)
        
        print(f"Metrics exported to: {filepath}")
        return filepath
    
    def export_anomaly_scores(self, target2d, rows, cols, filename=None):
        """Export anomaly scores to CSV"""
        if filename is None:
            filename = f'anomaly_scores_{self.timestamp}.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Calculate scores
        scores = np.zeros((rows * cols,))
        for i in range(rows * cols):
            scores[i] = np.linalg.norm(target2d[:, i])
        
        # Normalize
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        scores_2d = scores.reshape(rows, cols)
        
        # Save as CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Row', 'Col', 'Anomaly_Score'])
            for i in range(rows):
                for j in range(cols):
                    writer.writerow([i, j, scores_2d[i, j]])
        
        print(f"Anomaly scores exported to: {filepath}")
        return filepath
    
    def export_summary_report(self, metrics_dict, performance_tracker, config=None, filename=None):
        """Export comprehensive summary report"""
        if filename is None:
            filename = f'summary_report_{self.timestamp}.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        report = {
            'timestamp': self.timestamp,
            'experiment_date': datetime.now().isoformat(),
            'metrics': {},
            'performance': {},
            'configuration': config.config if config else {}
        }
        
        # Add metrics
        for key, value in metrics_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                report['metrics'][key] = float(value)
            elif isinstance(value, np.ndarray):
                report['metrics'][key] = value.tolist()
            else:
                report['metrics'][key] = value
        
        # Add performance timings
        if performance_tracker:
            report['performance'] = {
                'total_time': float(performance_tracker.end_time - performance_tracker.start_time) if performance_tracker.end_time else None,
                'stage_timings': {k: float(v) for k, v in performance_tracker.timings.items()}
            }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Summary report exported to: {filepath}")
        return filepath
    
    def export_matlab_compatible(self, target2d, background2d, rows, cols, bands, filename=None):
        """Export results in MATLAB-compatible format"""
        if filename is None:
            filename = f'results_{self.timestamp}.mat'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to 3D
        target3d = target2d.reshape(bands, rows * cols).T.reshape(rows, cols, bands)
        background3d = background2d.reshape(bands, rows * cols).T.reshape(rows, cols, bands)
        
        sio.savemat(filepath, {
            'target2d': target2d,
            'background2d': background2d,
            'target3d': target3d,
            'background3d': background3d,
            'rows': rows,
            'cols': cols,
            'bands': bands
        })
        
        print(f"MATLAB-compatible results exported to: {filepath}")
        return filepath

