"""
TermiAI - Utilities Module
Helper functions, logging, configuration management, and platform utilities
"""

import os
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any


class ConfigManager:
    """Manages application configuration"""

    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration manager"""
        self.config_file = Path(config_file)
        self.default_config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Ollama/LLM settings
            "ollama_host": "http://localhost:11434",
            "ollama_model": "phi3",
            "llama_timeout": 30,

            # Whisper settings
            "whisper_model_size": "tiny.en",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",

            # Execution settings
            "confirmation_required": True,
            "max_execution_time": 30,
            "allow_dangerous_commands": False,
            "dry_run_mode": False,

            # Audio settings
            "audio_timeout": 10,
            "audio_sample_rate": 16000,
            "audio_channels": 1,

            # Logging settings
            "log_level": "INFO",
            "log_to_file": True,
            "max_log_entries": 1000,

            # Safety settings
            "safety_level": "normal",  # strict, normal, permissive
            "auto_execute_safe_commands": False,

            # UI settings
            "show_transcription": True,
            "show_command_explanation": False,
            "color_output": True
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)

                # Merge with defaults
                config = self.default_config.copy()
                config.update(user_config)

                print(f"✅ Configuration loaded from {self.config_file}")
                return config

            except Exception as e:
                print(f"❌ Error loading config: {e}")
                print("🔧 Using default configuration")

        # Create default config file
        self.save_config(self.default_config)
        return self.default_config.copy()

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def update_config(self, key: str, value: Any) -> bool:
        """Update a single configuration value"""
        try:
            config = self.load_config()
            config[key] = value
            self.save_config(config)
            return True
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False

    def reset_config(self):
        """Reset configuration to defaults"""
        self.save_config(self.default_config)
        print("🔄 Configuration reset to defaults")


class Logger:
    """Handles application logging and history"""

    def __init__(self, log_dir: str = "logs"):
        """Initialize logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # History file
        self.history_file = self.log_dir / "history.log"

        # Set up Python logging
        self.setup_logging()

        # In-memory history for quick access
        self.interaction_history = []
        self.command_history = []

        # Load existing history
        self.load_history()

    def setup_logging(self):
        """Set up Python logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.log_dir / "termiai.log"),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("TermiAI")

    def log_interaction(self, user_input: str, command: str, executed: bool, input_type: str = "voice"):
        """Log user interaction"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "command": command,
            "executed": executed,
            "input_type": input_type,  # "voice" or "text"
            "platform": platform.system()
        }

        self.interaction_history.append(entry)
        self._save_to_file(entry, "interaction")

        # Keep only recent entries in memory
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def log_command_execution(self, command: str, status: str, output: str = ""):
        """Log command execution details"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "status": status,  # "started", "success", "failed", "error"
            "output": output[:500],  # Limit output length
            "platform": platform.system()
        }

        self.command_history.append(entry)
        self._save_to_file(entry, "command")

        # Keep only recent entries in memory
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

    def log_error(self, error_message: str, context: str = ""):
        """Log error messages"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "message": error_message,
            "context": context
        }

        self._save_to_file(entry, "error")
        self.logger.error(f"{context}: {error_message}" if context else error_message)

    def log_info(self, message: str, context: str = ""):
        """Log informational messages"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "info",
            "message": message,
            "context": context
        }

        self._save_to_file(entry, "info")
        self.logger.info(f"{context}: {message}" if context else message)

    def _save_to_file(self, entry: Dict, entry_type: str):
        """Save log entry to file"""
        try:
            log_line = f"[{entry_type.upper()}] {json.dumps(entry)}\n"

            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(log_line)

        except Exception as e:
            print(f"❌ Error saving to log file: {e}")

    def load_history(self):
        """Load history from file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse log entry
                    if line.startswith('[INTERACTION]'):
                        entry_json = line[13:].strip()
                        entry = json.loads(entry_json)
                        self.interaction_history.append(entry)
                    elif line.startswith('[COMMAND]'):
                        entry_json = line[9:].strip()
                        entry = json.loads(entry_json)
                        self.command_history.append(entry)

            # Keep only recent entries
            self.interaction_history = self.interaction_history[-100:]
            self.command_history = self.command_history[-100:]

        except Exception as e:
            print(f"❌ Error loading history: {e}")

    def get_recent_history(self, limit: int = 10) -> List[Dict]:
        """Get recent interaction history"""
        return self.interaction_history[-limit:]

    def get_command_history(self, limit: int = 10) -> List[Dict]:
        """Get recent command execution history"""
        return self.command_history[-limit:]

    def clear_history(self):
        """Clear all history"""
        self.interaction_history.clear()
        self.command_history.clear()

        # Clear log file
        try:
            self.history_file.unlink(missing_ok=True)
            print("🗑️  History cleared")
        except Exception as e:
            print(f"❌ Error clearing history: {e}")

    def export_history(self, output_file: str = None) -> str:
        """Export history to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"termiai_history_{timestamp}.json"

        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "interaction_history": self.interaction_history,
                "command_history": self.command_history
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            print(f"✅ History exported to {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error exporting history: {e}")
            return ""


class PlatformUtils:
    """Platform-specific utilities"""

    def __init__(self):
        """Initialize platform utilities"""
        self.platform = platform.system().lower()
        self.platform_info = self._get_platform_info()

    def _get_platform_info(self) -> Dict[str, str]:
        """Get detailed platform information"""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }

    def get_platform(self) -> str:
        """Get simplified platform name"""
        platform_map = {
            "darwin": "macOS",
            "linux": "Linux",
            "windows": "Windows"
        }
        return platform_map.get(self.platform, self.platform)

    def get_shell_command(self) -> str:
        """Get default shell command for platform"""
        if self.platform == "windows":
            return "cmd"
        else:
            return "bash"

    def get_path_separator(self) -> str:
        """Get path separator for platform"""
        return "\\" if self.platform == "windows" else "/"

    def normalize_path(self, path: str) -> str:
        """Normalize path for current platform"""
        if self.platform == "windows":
            return path.replace("/", "\\")
        else:
            return path.replace("\\", "/")

    def get_home_directory(self) -> str:
        """Get user home directory"""
        return str(Path.home())

    def get_temp_directory(self) -> str:
        """Get system temp directory"""
        import tempfile
        return tempfile.gettempdir()

    def is_admin(self) -> bool:
        """Check if running with admin privileges"""
        try:
            if self.platform == "windows":
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except:
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = self.platform_info.copy()

        # Add runtime info
        info.update({
            "home_directory": self.get_home_directory(),
            "temp_directory": self.get_temp_directory(),
            "current_directory": os.getcwd(),
            "is_admin": self.is_admin(),
            "shell": self.get_shell_command(),
            "path_separator": self.get_path_separator()
        })

        # Add disk space info
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.getcwd())
            info["disk_space"] = {
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2)
            }
        except:
            info["disk_space"] = "unavailable"

        return info


class ColorUtils:
    """Utilities for colored terminal output"""

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'gray': '\033[90m'
    }

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Add color to text"""
        if color in cls.COLORS:
            return f"{cls.COLORS[color]}{text}{cls.COLORS['reset']}"
        return text

    @classmethod
    def success(cls, text: str) -> str:
        """Green text for success messages"""
        return cls.colorize(text, 'green')

    @classmethod
    def error(cls, text: str) -> str:
        """Red text for error messages"""
        return cls.colorize(text, 'red')

    @classmethod
    def warning(cls, text: str) -> str:
        """Yellow text for warning messages"""
        return cls.colorize(text, 'yellow')

    @classmethod
    def info(cls, text: str) -> str:
        """Blue text for info messages"""
        return cls.colorize(text, 'blue')

    @classmethod
    def highlight(cls, text: str) -> str:
        """Cyan text for highlighted content"""
        return cls.colorize(text, 'cyan')


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def validate_url(url: str) -> bool:
    """Validate if string is a valid URL"""
    import re
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return pattern.match(url) is not None


# Test utilities
def test_utils():
    """Test utility functions"""
    print("🧪 Testing TermiAI Utilities...")

    # Test ConfigManager
    print("\n📝 Testing ConfigManager...")
    config_manager = ConfigManager("test_config.json")
    config = config_manager.load_config()
    print(f"✅ Config loaded: {len(config)} settings")

    # Test Logger
    print("\n📋 Testing Logger...")
    logger = Logger("test_logs")
    logger.log_info("Test message", "utils_test")
    logger.log_interaction("test input", "test command", True)
    print("✅ Logger test completed")

    # Test PlatformUtils
    print("\n🖥️  Testing PlatformUtils...")
    platform_utils = PlatformUtils()
    print(f"Platform: {platform_utils.get_platform()}")
    print(f"Home: {platform_utils.get_home_directory()}")
    print(f"Shell: {platform_utils.get_shell_command()}")

    # Test ColorUtils
    print("\n🎨 Testing ColorUtils...")
    print(ColorUtils.success("✅ Success message"))
    print(ColorUtils.error("❌ Error message"))
