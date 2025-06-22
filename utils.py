"""
TermiAI - Utilities Module
Helper functions, logging, configuration management, and platform utilities
"""

import os
import json
import logging
import platform
import re
import shutil
import tempfile
import ctypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


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
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # Merge with defaults
                config = self.default_config.copy()
                config.update(user_config)

                # Validate loaded config
                if not self.validate_config(config):
                    print("⚠️  Config validation failed, using defaults with user overrides")
                    config = self._merge_safe_config(user_config)

                print(f"✅ Configuration loaded from {self.config_file}")
                return config

            except json.JSONDecodeError as e:
                print(f"❌ Error parsing config JSON: {e}")
                print("🔧 Using default configuration")
            except Exception as e:
                print(f"❌ Error loading config: {e}")
                print("🔧 Using default configuration")

        # Create default config file
        self.save_config(self.default_config)
        return self.default_config.copy()

    def _merge_safe_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge user config with defaults, skipping invalid values"""
        config = self.default_config.copy()
        
        for key, value in user_config.items():
            if key in self.default_config:
                # Type checking
                default_type = type(self.default_config[key])
                if isinstance(value, default_type):
                    config[key] = value
                else:
                    print(f"⚠️  Invalid type for {key}, using default")
            else:
                print(f"⚠️  Unknown config key: {key}, ignoring")
        
        return config

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            # Update last_updated timestamp
            config["last_updated"] = datetime.now().strftime("%Y-%m-%d")
            
            # Create backup of existing config
            if self.config_file.exists():
                backup_path = self.config_file.with_suffix('.json.backup')
                shutil.copy2(self.config_file, backup_path)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")

    def update_config(self, key: str, value: Any) -> bool:
        """Update a single configuration value"""
        try:
            config = self.load_config()
            
            # Validate the specific key-value pair
            if not self._validate_config_item(key, value):
                print(f"❌ Invalid value for {key}")
                return False
                
            config[key] = value
            self.save_config(config)
            return True
        except Exception as e:
            print(f"❌ Error updating config: {e}")
            return False

    def _validate_config_item(self, key: str, value: Any) -> bool:
        """Validate a single configuration item"""
        validators = {
            "safety_level": lambda v: v in ["strict", "normal", "permissive"],
            "max_execution_time": lambda v: isinstance(v, (int, float)) and v > 0,
            "audio_timeout": lambda v: isinstance(v, (int, float)) and v > 0,
            "audio_sample_rate": lambda v: isinstance(v, int) and v > 0,
            "audio_channels": lambda v: isinstance(v, int) and v in [1, 2],
            "whisper_model_size": lambda v: v in ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large"],
            "whisper_device": lambda v: v in ["cpu", "cuda", "auto"],
            "log_level": lambda v: v in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "ollama_host": lambda v: isinstance(v, str) and (v.startswith("http://") or v.startswith("https://")),
        }
        
        if key in validators:
            return validators[key](value)
        return True

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

            # Validate specific values using individual validators
            for key, value in config.items():
                if not self._validate_config_item(key, value):
                    print(f"❌ Invalid value for {key}: {value}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Config validation error: {e}")
            return False

    def export_config(self, export_path: Optional[str] = None) -> str:
        """Export current configuration to a file"""
        if not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = f"config_backup_{timestamp}.json"
        
        try:
            config = self.load_config()
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration exported to {export_path}")
            return export_path
        except Exception as e:
            print(f"❌ Error exporting config: {e}")
            return ""

    def import_config(self, import_path: str) -> bool:
        """Import configuration from a file"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            if self.validate_config(imported_config):
                self.save_config(imported_config)
                print(f"✅ Configuration imported from {import_path}")
                return True
            else:
                print(f"❌ Invalid configuration in {import_path}")
                return False
        except Exception as e:
            print(f"❌ Error importing config: {e}")
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
        self.session_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Set up Python logging
        self.setup_logging()

        # In-memory history for quick access
        self.interaction_history = []
        self.command_history = []
        self.session_start = datetime.now()

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

        # File handler for main log
        file_handler = logging.FileHandler(self.log_dir / "termiai.log", encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

        # Session-specific handler
        session_handler = logging.FileHandler(self.session_file, encoding='utf-8')
        session_handler.setFormatter(formatter)
        session_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(session_handler)

        # Console handler (optional - can be disabled)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        self.logger.addHandler(console_handler)

    def log_interaction(self, user_input: str, command: str, executed: bool, input_type: str = "voice"):
        """Log user interaction"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input.strip(),
            "command": command.strip(),
            "executed": executed,
            "input_type": input_type,  # "voice" or "text"
            "platform": platform.system(),
            "session_id": id(self)  # Simple session identifier
        }

        self.interaction_history.append(entry)
        self._save_to_file(entry, "interaction")

        # Keep only recent entries in memory
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]

    def log_command_execution(self, command: str, status: str, output: str = "", execution_time: float = 0):
        """Log command execution details"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command.strip(),
            "status": status,  # "started", "success", "failed", "error", "timeout"
            "output": self._truncate_output(output),
            "execution_time": execution_time,
            "platform": platform.system(),
            "session_id": id(self)
        }

        self.command_history.append(entry)
        self._save_to_file(entry, "command")

        # Keep only recent entries in memory
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

    def _truncate_output(self, output: str, max_length: int = 1000) -> str:
        """Truncate command output for logging"""
        if not output:
            return ""
        
        output = output.strip()
        if len(output) <= max_length:
            return output
        
        return output[:max_length] + f"\n... [truncated {len(output) - max_length} characters]"

    def log_error(self, error_message: str, context: str = "", exception: Optional[Exception] = None):
        """Log error messages"""
        error_details = str(error_message)
        if exception:
            error_details += f" | Exception: {type(exception).__name__}: {str(exception)}"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "message": error_details,
            "context": context,
            "session_id": id(self)
        }

        self._save_to_file(entry, "error")
        self.logger.error(f"{context}: {error_details}" if context else error_details)

        # Also save to dedicated error file
        try:
            with open(self.error_file, 'a', encoding='utf-8') as f:
                f.write(f"{entry['timestamp']} - {context}: {error_details}\n")
        except Exception:
            pass  # Avoid infinite error loops

    def log_info(self, message: str, context: str = ""):
        """Log informational messages"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "info",
            "message": str(message),
            "context": context,
            "session_id": id(self)
        }

        self._save_to_file(entry, "info")
        self.logger.info(f"{context}: {message}" if context else message)

        # Also save to dedicated info file
        try:
            with open(self.info_file, 'a', encoding='utf-8') as f:
                f.write(f"{entry['timestamp']} - {context}: {message}\n")
        except Exception:
            pass

    def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log system events like startup, shutdown, config changes"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "system_event",
            "event_type": event_type,
            "details": details,
            "session_id": id(self)
        }
        
        self._save_to_file(entry, "system")
        self.logger.info(f"System event: {event_type} - {details}")

    def _save_to_file(self, entry: Dict, entry_type: str):
        """Save log entry to file"""
        try:
            log_line = f"[{entry_type.upper()}] {json.dumps(entry, ensure_ascii=False)}\n"

            with open(self.history_file, 'a', encoding='utf-8') as f:
                f.write(log_line)

        except Exception as e:
            # Use print to avoid infinite recursion
            print(f"❌ Error saving to log file: {e}")

    def load_history(self):
        """Load history from file"""
        if not self.history_file.exists():
            return

        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        # Parse log entry
                        if line.startswith('[INTERACTION]'):
                            entry_json = line[13:].strip()
                            entry = json.loads(entry_json)
                            self.interaction_history.append(entry)
                        elif line.startswith('[COMMAND]'):
                            entry_json = line[9:].strip()
                            entry = json.loads(entry_json)
                            self.command_history.append(entry)
                    except json.JSONDecodeError:
                        print(f"⚠️  Skipping malformed log entry at line {line_num}")
                        continue

            # Keep only recent entries
            self.interaction_history = self.interaction_history[-100:]
            self.command_history = self.command_history[-100:]

            print(f"✅ Loaded {len(self.interaction_history)} interactions and {len(self.command_history)} commands from history")

        except Exception as e:
            print(f"❌ Error loading history: {e}")

    def get_recent_history(self, limit: int = 10) -> List[Dict]:
        """Get recent interaction history"""
        return self.interaction_history[-limit:] if self.interaction_history else []

    def get_command_history(self, limit: int = 10) -> List[Dict]:
        """Get recent command execution history"""
        return self.command_history[-limit:] if self.command_history else []

    def search_history(self, query: str, history_type: str = "all") -> List[Dict]:
        """Search through history"""
        results = []
        query_lower = query.lower()
        
        if history_type in ["all", "interaction"]:
            for entry in self.interaction_history:
                if (query_lower in entry.get("user_input", "").lower() or 
                    query_lower in entry.get("command", "").lower()):
                    results.append(entry)
        
        if history_type in ["all", "command"]:
            for entry in self.command_history:
                if (query_lower in entry.get("command", "").lower() or 
                    query_lower in entry.get("output", "").lower()):
                    results.append(entry)
        
        return sorted(results, key=lambda x: x.get("timestamp", ""))

    def clear_history(self):
        """Clear all history"""
        self.interaction_history.clear()
        self.command_history.clear()

        # Clear log files
        files_to_clear = [self.history_file, self.error_file, self.info_file]
        for file_path in files_to_clear:
            try:
                if file_path.exists():
                    file_path.unlink()
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
                "session_start": self.session_start.isoformat(),
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "interaction_history": self.interaction_history,
                "command_history": self.command_history,
                "statistics": self.get_log_stats(),
                "total_interactions": len(self.interaction_history),
                "total_commands": len(self.command_history)
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✅ History exported to {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error exporting history: {e}")
            return ""

    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "session_duration": str(datetime.now() - self.session_start),
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

        # Add average execution time
        execution_times = [cmd.get("execution_time", 0) for cmd in self.command_history if cmd.get("execution_time")]
        if execution_times:
            stats["avg_execution_time"] = round(sum(execution_times) / len(execution_times), 2)
        else:
            stats["avg_execution_time"] = 0

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
                    shutil.move(str(log_file), str(backup_path))
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
        info = {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "architecture": platform.architecture()[0]
        }
        
        # Add platform-specific info
        if self.platform == "linux":
            try:
                # Try to get Linux distribution info
                with open("/etc/os-release", "r") as f:
                    os_release = f.read()
                    for line in os_release.split("\n"):
                        if line.startswith("PRETTY_NAME="):
                            info["distribution"] = line.split("=")[1].strip('"')
                            break
            except:
                info["distribution"] = "Unknown Linux"
        
        elif self.platform == "darwin":
            info["macos_version"] = platform.mac_ver()[0]
        
        elif self.platform == "windows":
            info["windows_version"] = platform.win32_ver()[0]
            info["windows_edition"] = platform.win32_edition()
        
        return info

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
            # Check if PowerShell is available
            if shutil.which("powershell"):
                return "powershell"
            return "cmd"
        else:
            # Check available shells
            shells = ["/bin/zsh", "/bin/bash", "/bin/sh"]
            for shell in shells:
                if Path(shell).exists():
                    return shell
            return "bash"  # fallback

    def get_path_separator(self) -> str:
        """Get path separator for platform"""
        return "\\" if self.platform == "windows" else "/"

    def normalize_path(self, path: str) -> str:
        """Normalize path for current platform"""
        normalized = Path(path).as_posix() if self.platform != "windows" else str(Path(path)).replace("/", "\\")
        return normalized

    def get_home_directory(self) -> str:
        """Get user home directory"""
        return str(Path.home())

    def get_temp_directory(self) -> str:
        """Get system temp directory"""
        return tempfile.gettempdir()

    def get_documents_folder(self) -> str:
        """Get user documents folder"""
        home = Path.home()
        documents_paths = {
            "windows": home / "Documents",
            "darwin": home / "Documents",
            "linux": home / "Documents"
        }
        return str(documents_paths.get(self.platform, home / "Documents"))

    def get_downloads_folder(self) -> str:
        """Get user downloads folder"""
        home = Path.home()
        downloads_paths = {
            "windows": home / "Downloads",
            "darwin": home / "Downloads", 
            "linux": home / "Downloads"
        }
        return str(downloads_paths.get(self.platform, home / "Downloads"))

    def get_desktop_folder(self) -> str:
        """Get user desktop folder"""
        home = Path.home()
        desktop_paths = {
            "windows": home / "Desktop",
            "darwin": home / "Desktop",
            "linux": home / "Desktop"
        }
        return str(desktop_paths.get(self.platform, home / "Desktop"))

    def is_admin(self) -> bool:
        """Check if running with admin privileges"""
        try:
            if self.platform == "windows":
                return bool(ctypes.windll.shell32.IsUserAnAdmin())
            else:
                return os.geteuid() == 0
        except Exception:
            return False

    def get_available_drives(self) -> List[str]:
        """Get available drives (Windows) or mount points (Unix)"""
        drives = []
        
        if self.platform == "windows":
            # Get Windows drive letters
            import string
            for letter in string.ascii_uppercase:
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append(drive)
        else:
            # Get Unix mount points
            try:
                with open("/proc/mounts", "r") as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1].startswith("/"):
                            mount_point = parts[1]
                            if not mount_point.startswith(("/proc", "/sys", "/dev")):
                                drives.append(mount_point)
                drives = sorted(list(set(drives)))
            except:
                drives = ["/"]
        
        return drives

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
            "desktop_folder": self.get_desktop_folder(),
            "available_drives": self.get_available_drives()
        })

        # Add disk space info
        try:
            total, used, free = shutil.disk_usage(os.getcwd())
            info["disk_space"] = {
                "total_gb": round(total / (1024 ** 3), 2),
                "used_gb": round(used / (1024 ** 3), 2),
                "free_gb": round(free / (1024 ** 3), 2),
                "usage_percent": round((used / total) * 100, 2)
            }
        except Exception:
            info["disk_space"] = "unavailable"

        # Add memory info (if psutil is available)
        try:
            import psutil
            memory = psutil.virtual_memory()
            info["memory"] = {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "used_percent": memory.percent
            }
            
            # CPU info
            info["cpu"] = {
                "count": psutil.cpu_count(),
                "usage_percent": psutil.cpu_percent(interval=1)
            }
        except ImportError:
            info["memory"] = "psutil not available"
            info["cpu"] = "psutil not available"
        except Exception:
            info["memory"] = "unavailable"
            info["cpu"] = "unavailable"

        return info

    def get_env_vars(self) -> Dict[str, str]:
        """Get important environment variables"""
        important_vars = ["PATH", "HOME", "USER", "SHELL", "TERM", "PYTHONPATH"]
        
        if self.platform == "windows":
            important_vars.extend(["USERPROFILE", "USERNAME", "COMPUTERNAME", "OS", "PROCESSOR_ARCHITECTURE"])
        else:
            important_vars.extend(["PWD", "OLDPWD", "DISPLAY"])

        return {var: os.environ.get(var, "Not set") for var in important_vars}

    def get_network_info(self) -> Dict[str, Any]:
        """Get basic network information"""
        network_info = {}
        
        try:
            import socket
            hostname = socket.gethostname()
            network_info["hostname"] = hostname
            
            try:
                # Get local IP address
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
                network_info["local_ip"] = local_ip
            except Exception:
                network_
