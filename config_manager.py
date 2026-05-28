# coding=utf-8
"""
Configuration Management Module
Allows loading parameters from YAML/JSON files
"""
import json
import os
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class ConfigManager:
    """Manage configuration parameters"""
    
    def __init__(self, config_file=None):
        self.config = {}
        if config_file:
            self.load(config_file)
        else:
            self.set_defaults()
    
    def set_defaults(self):
        """Set default configuration parameters"""
        self.config = {
            'data': {
                'input_file': 'Sandiego.mat',
                'groundtruth_file': 'PlaneGT.mat',
                'crop_rows': 100,
                'crop_cols': 100,
                'remove_bands': {
                    'ranges': [[0, 6], [32, 35], [93, 97], [106, 113], [152, 166], [220, 224]]
                }
            },
            'dictionary_construction': {
                'win_size': 3,
                'cluster_num': 10,
                'K': 10,
                'selected_dic_percent': 0.05,
                'target_dic_num': 200
            },
            'lrsr': {
                'beta': 0.001,
                'lmda': 0.01,
                'max_iterations': 500,
                'convergence_error': 0.000001
            },
            'output': {
                'save_results': True,
                'save_plots': True,
                'output_dir': 'results',
                'export_formats': ['mat', 'json']
            }
        }
    
    def load(self, config_file):
        """Load configuration from file"""
        if not os.path.exists(config_file):
            print(f"Config file {config_file} not found. Using defaults.")
            self.set_defaults()
            return
        
        ext = os.path.splitext(config_file)[1].lower()
        
        try:
            with open(config_file, 'r') as f:
                if ext in ['.yaml', '.yml']:
                    if HAS_YAML:
                        self.config = yaml.safe_load(f)
                    else:
                        print("PyYAML not installed. Install with: pip install pyyaml")
                        print("Falling back to defaults.")
                        self.set_defaults()
                elif ext == '.json':
                    self.config = json.load(f)
                else:
                    print(f"Unsupported config file format: {ext}. Using defaults.")
                    self.set_defaults()
        except Exception as e:
            print(f"Error loading config file: {e}. Using defaults.")
            self.set_defaults()
    
    def save(self, config_file):
        """Save current configuration to file"""
        ext = os.path.splitext(config_file)[1].lower()
        
        with open(config_file, 'w') as f:
            if ext in ['.yaml', '.yml']:
                if HAS_YAML:
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                else:
                    raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
            elif ext == '.json':
                json.dump(self.config, f, indent=4)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'data.input_file')"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[keys[-1]] = value

