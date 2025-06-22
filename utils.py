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
            "color_output": True,
            
            # Version info
            "version": "1.0.0",
            "last_updated": datetime.now().strftime("%Y-%m-%d")
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
            # Update last_updated timestamp
            config["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
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

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration values"""
        try:
            # Check required keys
            required_keys = ["ollama_host", "ollama_model", "whisper_model_size"]
            for key in required_keys:
                if key not in config:
                    print(f"❌ Missing required config key: {key}")
                    return False

            # Validate specific values
            if config.get("safety_level") not in ["strict", "normal", "permissive"]:
                print("❌ Invalid safety_level. Must be: strict, normal, or permissive")
                return False

            if config.get("max_execution_time", 0) <= 0:
                print("❌ max_execution_time must be greater than 0")
                return False

            return True

        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False


class Logger:
    """Handles application logging and history"""

    def __init__(self, log_dir: str = "logs"):
        """Initialize logger"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # History file
        self.history_file = self.log_dir / "history.log"
        self.error_file = self.log_dir / "errors.log"
        self.info_file = self.log_dir / "info.log"

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

        # Create formatters
        formatter = logging.Formatter(log_format)

        # Configure root logger
        self.logger = logging.getLogger("TermiAI")
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        file_handler = logging.FileHandler(self.log_dir / "termiai.log")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Console handler (optional - can be disabled)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        self.logger.addHandler(console_handler)

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
            "output": output[:500] if output else "",  # Limit output length
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

        # Also save to dedicated error file
        try:
            with open(self.error_file, 'a', encoding='utf-8') as f:
                f.write(f"{entry['timestamp']} - {context}: {error_message}\n")
        except:
            pass

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

        # Also save to dedicated info file
        try:
            with open(self.info_file, 'a', encoding='utf-8') as f:
                f.write(f"{entry['timestamp']} - {context}: {message}\n")
        except:
            pass

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

            print(f"✅ Loaded {len(self.interaction_history)} interactions and {len(self.command_history)} commands from history")

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

        # Clear log files
        files_to_clear = [self.history_file, self.error_file, self.info_file]
        for file_path in files_to_clear:
            try:
                file_path.unlink(missing_ok=True)
            except Exception as e:
                print(f"❌ Error clearing {file_path}: {e}")

        print("🗑️  History cleared")

    def export_history(self, output_file: str = None) -> str:
        """Export history to JSON file"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"termiai_history_{timestamp}.json"

        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "platform": platform.system(),
                "interaction_history": self.interaction_history,
                "command_history": self.command_history,
                "total_interactions": len(self.interaction_history),
                "total_commands": len(self.command_history)
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2)

            print(f"✅ History exported to {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error exporting history: {e}")
            return ""

    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "total_interactions": len(self.interaction_history),
            "total_commands": len(self.command_history),
            "successful_commands": sum(1 for cmd in self.command_history if cmd.get("status") == "success"),
            "failed_commands": sum(1 for cmd in self.command_history if cmd.get("status") in ["failed", "error"]),
            "voice_interactions": sum(1 for interaction in self.interaction_history if interaction.get("input_type") == "voice"),
            "text_interactions": sum(1 for interaction in self.interaction_history if interaction.get("input_type") == "text"),
        }

        if stats["total_commands"] > 0:
            stats["success_rate"] = round((stats["successful_commands"] / stats["total_commands"]) * 100, 2)
        else:
            stats["success_rate"] = 0

        return stats

    def rotate_logs(self, max_size_mb: int = 10):
        """Rotate log files if they get too large"""
        max_size_bytes = max_size_mb * 1024 * 1024
        
        files_to_check = [self.history_file, self.error_file, self.info_file]
        
        for log_file in files_to_check:
            if log_file.exists() and log_file.stat().st_size > max_size_bytes:
                # Create backup
                backup_name = f"{log_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{log_file.suffix}"
                backup_path = log_file.parent / backup_name
                
                try:
                    log_file.rename(backup_path)
                    print(f"📦 Rotated log file: {log_file} -> {backup_path}")
                except Exception as e:
                    print(f"❌ Error rotating log file {log_file}: {e}")


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
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }

    def get_platform(self) -> str:
        """Get simplified platform name"""
        platform_map = {
            "darwin": "macOS",
            "linux": "Linux",
            "windows": "Windows"
        }
        return platform_map.get(self.platform, self.platform.capitalize())

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

    def get_documents_folder(self) -> str:
        """Get user documents folder"""
        if self.platform == "windows":
            return os.path.join(self.get_home_directory(), "Documents")
        else:
            return os.path.join(self.get_home_directory(), "Documents")

    def get_downloads_folder(self) -> str:
        """Get user downloads folder"""
        return os.path.join(self.get_home_directory(), "Downloads")

    def get_desktop_folder(self) -> str:
        """Get user desktop folder"""
        return os.path.join(self.get_home_directory(), "Desktop")

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
            "path_separator": self.get_path_separator(),
            "documents_folder": self.get_documents_folder(),
            "downloads_folder": self.get_downloads_folder(),
            "desktop_folder": self.get_desktop_folder()
        })

        # Add disk space info
        try:
            import shutil
            total, used, free = shutil.disk_usage(os.getcwd())
            info["disk_space"] = {
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "usage_percent": round((used / total) * 100, 2)
            }
        except:
            info["disk_space"] = "unavailable"

        # Add memory info (if available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory"] = {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "used_percent": memory.percent
            }
        except ImportError:
            info["memory"] = "psutil not available"
        except:
            info["memory"] = "unavailable"

        return info

    def get_env_vars(self) -> Dict[str, str]:
        """Get important environment variables"""
        important_vars = ["PATH", "HOME", "USER", "SHELL", "TERM", "PYTHON_PATH"]
        if self.platform == "windows":
            important_vars.extend(["USERPROFILE", "USERNAME", "COMPUTERNAME", "OS"])

        return {var: os.environ.get(var, "Not set") for var in important_vars}


class ColorUtils:
    """Utilities for colored terminal output"""

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'underline': '\033[4m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'gray': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m'
    }

    @classmethod
    def colorize(cls, text: str, color: str, bold: bool = False) -> str:
        """Add color to text"""
        if not cls.is_color_supported():
            return text
            
        color_code = cls.COLORS.get(color, '')
        bold_code = cls.COLORS['bold'] if bold else ''
        reset_code = cls.COLORS['reset']
        
        return f"{bold_code}{color_code}{text}{reset_code}"

    @classmethod
    def is_color_supported(cls) -> bool:
        """Check if terminal supports colors"""
        return hasattr(os, 'isatty') and os.isatty(1)

    @classmethod
    def success(cls, text: str, bold: bool = True) -> str:
        """Green text for success messages"""
        return cls.colorize(text, 'bright_green', bold)

    @classmethod
    def error(cls, text: str, bold: bool = True) -> str:
        """Red text for error messages"""
        return cls.colorize(text, 'bright_red', bold)

    @classmethod
    def warning(cls, text: str, bold: bool = True) -> str:
        """Yellow text for warning messages"""
        return cls.colorize(text, 'bright_yellow', bold)

    @classmethod
    def info(cls, text: str, bold: bool = False) -> str:
        """Blue text for info messages"""
        return cls.colorize(text, 'bright_blue', bold)

    @classmethod
    def highlight(cls, text: str, bold: bool = True) -> str:
        """Cyan text for highlighted content"""
        return cls.colorize(text, 'bright_cyan', bold)

    @classmethod
    def dim(cls, text: str) -> str:
        """Dim text for less important content"""
        return cls.colorize(text, 'gray', False)


class FileUtils:
    """File and directory utilities"""

    @staticmethod
    def ensure_directory(directory: str) -> bool:
        """Ensure directory exists, create if it doesn't"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {e}")
            return False

    @staticmethod
    def safe_filename(filename: str) -> str:
        """Create a safe filename by removing invalid characters"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    @staticmethod
    def get_file_size(filepath: str) -> int:
        """Get file size in bytes"""
        try:
            return Path(filepath).stat().st_size
        except:
            return 0

    @staticmethod
    def backup_file(filepath: str) -> str:
        """Create a backup of a file"""
        try:
            original = Path(filepath)
            if not original.exists():
                return ""

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{original.stem}_backup_{timestamp}{original.suffix}"
            backup_path = original.parent / backup_name
            
            import shutil
            shutil.copy2(original, backup_path)
            
            return str(backup_path)
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return ""

    @staticmethod
    def cleanup_old_files(directory: str, days_old: int = 30, pattern: str = "*"):
        """Clean up old files in a directory"""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now() - timedelta(days=days_old)
            directory_path = Path(directory)
            
            if not directory_path.exists():
                return
                
            deleted_count = 0
            for file_path in directory_path.glob(pattern):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted_count += 1
                        
            if deleted_count > 0:
                print(f"🗑️  Cleaned up {deleted_count} old files from {directory}")
                
        except Exception as e:
            print(f"❌ Error cleaning up files: {e}")


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


def get_system_command_examples() -> Dict[str, List[str]]:
    """Get platform-specific command examples"""
    system = platform.system().lower()
    
    if system == "windows":
        return {
            "File Operations": [
                "dir - List directory contents",
                "mkdir folder_name - Create directory",
                "rmdir folder_name - Remove directory",
                "copy file1.txt file2.txt - Copy file",
                "move file1.txt new_location\\ - Move file",
                "del file.txt - Delete file"
            ],
            "System Info": [
                "systeminfo - Display system information",
                "tasklist - List running processes",
                "ipconfig - Display network configuration",
                "date /t - Display current date",
                "time /t - Display current time"
            ],
            "Navigation": [
                "cd folder_name - Change directory",
                "cd .. - Go up one directory",
                "cd \\ - Go to root directory"
            ]
        }
    else:
        return {
            "File Operations": [
                "ls -la - List directory contents with details",
                "mkdir folder_name - Create directory",
                "rmdir folder_name - Remove empty directory",
                "cp file1.txt file2.txt - Copy file",
                "mv file1.txt new_location/ - Move file",
                "rm file.txt - Delete file"
            ],
            "System Info": [
                "ps aux - List running processes",
                "top - Display running processes",
                "df -h - Display disk usage",
                "free -h - Display memory usage",
                "uname -a - Display system information",
                "date - Display current date and time"
            ],
            "Navigation": [
                "cd folder_name - Change directory",
                "cd .. - Go up one directory",
                "cd ~ - Go to home directory",
                "pwd - Print working directory"
            ]
        }


# Test utilities
def test_utils():
    """Test utility functions"""
    print("🧪 Testing TermiAI Utilities...")

    # Test ConfigManager
    print("\n📝 Testing ConfigManager...")
    config_manager = ConfigManager("test_config.json")
    config = config_manager.load_config()
    print(f"✅ Config loaded: {len(config)} settings")
    
    # Test config validation
    is_valid = config_manager.validate_config(config)
    print(f"✅ Config validation: {'Passed' if is_valid else 'Failed'}")

    # Test Logger
    print("\n📋 Testing Logger...")
    logger = Logger("test_logs")
    logger.log_info("Test message", "utils_test")
    logger.log_interaction("test input", "test command", True, "text")
    logger.log_command_execution("echo test", "success", "test output")
    
    stats = logger.get_log_stats()
    print(f"✅ Logger stats: {stats}")

    # Test PlatformUtils
    print("\n🖥️  Testing PlatformUtils...")
    platform_utils = PlatformUtils()
    print(f"Platform: {platform_utils.get_platform()}")
    print(f"Home: {platform_utils.get_home_directory()}")
    print(f"Shell: {platform_utils.get_shell_command()}")
    print(f"Admin: {platform_utils.is_admin()}")

    # Test ColorUtils
    print("\n🎨 Testing ColorUtils...")
    print(ColorUtils.success("✅ Success message"))
    print(ColorUtils.error("❌ Error message"))
    print(ColorUtils.warning("⚠️  Warning message"))
    print(ColorUtils.info("ℹ️  Info message"))
    print(ColorUtils.highlight("🔥 Highlighted message"))

    # Test FileUtils
    print("\n📁 Testing FileUtils...")
    FileUtils.ensure_directory("test_directory")
    safe_name = FileUtils.safe_filename("unsafe<>filename?.txt")
    print(f"Safe filename: {safe_name}")

    # Test utility functions
    print("\n🔧 Testing utility functions...")
    print(f"File size: {format_file_size(1024 * 1024 * 5)}")
    print(f"Duration: {format_duration(125.5)}")
    print(f"URL valid: {validate_url('https://example.com')}")
