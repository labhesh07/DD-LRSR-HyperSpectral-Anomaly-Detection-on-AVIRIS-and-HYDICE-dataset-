# coding=utf-8
"""
Performance Tracking Module
Tracks execution time and provides progress indicators
"""
import time
from datetime import datetime
from contextlib import contextmanager


class PerformanceTracker:
    """Track performance metrics for different stages"""
    
    def __init__(self):
        self.timings = {}
        self.start_time = None
        self.end_time = None
        
    def start(self):
        """Start overall timing"""
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Experiment started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")
        
    def end(self):
        """End overall timing"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        print(f"\n{'='*60}")
        print(f"Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"{'='*60}\n")
        return total_time
    
    @contextmanager
    def track(self, stage_name):
        """Context manager to track time for a specific stage"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {stage_name}...")
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.timings[stage_name] = elapsed
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {stage_name} ({elapsed:.2f}s)")
    
    def get_report(self):
        """Get formatted timing report"""
        if not self.timings:
            return "No timing data available"
        
        report = "\n" + "="*60 + "\n"
        report += "PERFORMANCE BREAKDOWN\n"
        report += "="*60 + "\n"
        
        total_tracked = sum(self.timings.values())
        for stage, elapsed in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (elapsed / total_tracked * 100) if total_tracked > 0 else 0
            report += f"{stage:30s}: {elapsed:8.2f}s ({percentage:5.1f}%)\n"
        
        report += "-"*60 + "\n"
        report += f"{'Total Tracked':30s}: {total_tracked:8.2f}s\n"
        report += "="*60 + "\n"
        
        return report
    
    def print_report(self):
        """Print timing report"""
        print(self.get_report())

