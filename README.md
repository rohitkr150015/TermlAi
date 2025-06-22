# TermlAi
# TermlAi 🎙️🤖

**Voice-Controlled Terminal Assistant with AI Command Generation**

TermlAi is an intelligent terminal assistant that converts natural language voice commands into executable terminal commands using Faster-Whisper for speech recognition and Ollama (phi3) for command generation.

## ✨ Features

### 🎙️ Voice Recognition
- **Faster-Whisper Integration**: High-quality, local speech-to-text processing
- **Real-time Audio Processing**: Record and transcribe voice commands instantly  
- **Multi-platform Audio Support**: Works on Windows, macOS, and Linux
- **Configurable Models**: Choose from different Whisper model sizes for accuracy vs speed

### 🧠 AI-Powered Command Generation
- **Ollama + phi3 Integration**: Local LLM for secure command generation
- **Context-Aware Commands**: Platform-specific command generation
- **Natural Language Processing**: Convert everyday language to terminal commands
- **Command Explanation**: Get explanations for generated commands

### 🛡️ Security & Safety
- **Command Validation**: Built-in safety checks for dangerous commands
- **Confirmation System**: Optional user confirmation before execution
- **Dry Run Mode**: Test commands without executing them
- **Configurable Safety Levels**: Strict, normal, or permissive modes
- **Command Blacklisting**: Prevents execution of destructive operations

### 📊 Comprehensive Logging
- **Interaction Tracking**: Log all voice inputs and generated commands
- **Command History**: Track execution success/failure with timestamps
- **Error Logging**: Detailed error tracking and debugging information
- **System Monitoring**: Performance and usage statistics
- **Multiple Log Formats**: JSON and text logs for different use cases

### ⚙️ Advanced Configuration
- **Flexible Settings**: Customize all aspects through config.json
- **Platform Detection**: Automatic platform-specific optimizations
- **Performance Tuning**: Adjustable timeouts, model sizes, and processing options
- **Audio Configuration**: Fine-tune microphone and processing settings

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for models and logs
- **Microphone**: Any USB or built-in microphone

### Dependencies
- **Ollama**: Local LLM server
- **Faster-Whisper**: Speech recognition
- **PyAudio**: Audio recording
- **aiohttp**: Async HTTP client

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/TermlAi.git
cd TermlAi
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
**Windows/macOS:**
- Download from [ollama.ai](https://ollama.ai/)
- Run the installer

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 4. Install phi3 Model
```bash
ollama pull phi3
```

### 5. Start Ollama Server
```bash
ollama serve
```

### 6. Configure TermlAi
Edit `config.json` to customize settings:
```json
{
  "ollama_host": "http://localhost:11434",
  "ollama_model": "phi3",
  "whisper_model_size": "tiny.en",
  "confirmation_required": true,
  "safety_level": "normal"
}
```

## 🎯 Usage

### Basic Usage
```bash
python main.py
```

### Interactive Commands
- **Press Enter**: Start voice recording
- **Speak naturally**: "list files in current directory"
- **Confirm/Edit**: Review generated command before execution
- **Type commands**: Direct text input also supported

### Voice Command Examples
```
🗣️ "list all files in documents folder"
💻 Generated: ls -la ~/Documents

🗣️ "create a folder called projects"  
💻 Generated: mkdir projects

🗣️ "show system information"
💻 Generated: uname -a

🗣️ "go to desktop directory"
💻 Generated: cd ~/Desktop
```

### Special Commands
- **"history"**: View recent command history
- **"config"**: Display current configuration
- **"exit"** or **"quit"**: Stop TermlAi

## ⚙️ Configuration

### Main Configuration (`config.json`)
```json
{
  "ollama_host": "http://localhost:11434",
  "ollama_model": "phi3",
  "llama_timeout": 30,
  "whisper_model_size": "tiny.en",
  "whisper_device": "cpu",
  "whisper_compute_type": "int8",
  "confirmation_required": true,
  "max_execution_time": 30,
  "allow_dangerous_commands": false,
  "dry_run_mode": false,
  "log_level": "INFO",
  "safety_level": "normal",
  "color_output": true
}
```

### Whisper Model Options
- **tiny.en**: Fastest, least accurate (~39MB)
- **base.en**: Balanced speed/accuracy (~74MB)  
- **small.en**: Good accuracy (~244MB)
- **medium.en**: Better accuracy (~769MB)
- **large**: Best accuracy (~1550MB)

### Safety Levels
- **strict**: Maximum security, 10s timeout, no dangerous commands
- **normal**: Balanced security, 30s timeout, basic protection
- **permissive**: Minimal restrictions, 60s timeout, advanced users

## 📁 Project Structure

```
TermlAi/
├── main.py              # Main application entry point
├── voice_input.py       # Voice recording & Whisper transcription
├── llm_interface.py     # Ollama API communication
├── executor.py          # Safe command execution
├── utils.py             # Logging, config, and utilities
├── config.json          # Configuration settings
├── requirements.txt     # Python dependencies
├── logs/               # Log files directory
│   ├── history.log     # Command history (text)
│   ├── interactions.log # User interactions (JSON)
│   ├── commands.log    # Command execution (JSON)
│   ├── errors.log      # Error tracking (JSON)
│   └── system.log      # System information (JSON)
└── README.md           # This file
```

## 📊 Logging System

TermlAi maintains comprehensive logs for monitoring and debugging:

### Log Files
- **`history.log`**: Human-readable command history
- **`interactions.log`**: Detailed interaction data (JSON)
- **`commands.log`**: Command execution details (JSON)
- **`errors.log`**: Error tracking and debugging (JSON)
- **`system.log`**: System information and startup data (JSON)

### Log Management
- **Automatic Rotation**: Prevents log files from growing too large
- **Configurable Retention**: Set maximum number of log entries
- **Performance Monitoring**: Track success rates and execution times
- **Error Analysis**: Detailed error categorization and tracking

## 🛡️ Security Features

### Command Safety
- **Pattern Matching**: Blocks known dangerous command patterns
- **Syntax Validation**: Prevents malformed commands
- **Execution Timeouts**: Prevents runaway processes
- **User Confirmation**: Review commands before execution

### Dangerous Command Protection
Automatically blocks commands like:
- `rm -rf /` (destructive deletion)
- `format c:` (disk formatting)
- `shutdown` (system shutdown)
- `chmod 777` (permission changes)
- And many more...

### Privacy
- **Local Processing**: All speech recognition happens locally
- **No Data Collection**: No personal data sent to external servers
- **Secure Communication**: Local-only LLM processing

## 🔧 Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Start Ollama server
ollama serve

# Check if phi3 is installed
ollama list

# Install phi3 if missing
ollama pull phi3
```

**Microphone Not Working**
```python
# Test microphone in Python
python -c "
from voice_input import VoiceProcessor
processor = VoiceProcessor({})
print('Mic test:', processor.test_microphone())
processor.list_audio_devices()
"
```

**Audio Dependencies Missing**
```bash
# Linux
sudo apt-get install portaudio19-dev python3-pyaudio

# macOS
brew install portaudio
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

**Model Loading Errors**
```bash
# Download specific Whisper model
python -c "
from faster_whisper import WhisperModel
model = WhisperModel('tiny.en')
print('Model loaded successfully')
"
```

### Performance Optimization

**Speed up Whisperrs:**
- Use smaller models (tiny.en, base.en)
- Set device to "cpu" for consistency
- Use "int8" compute type for lower memory usage

**Reduce Latency:**
- Lower audio timeout settings
- Use faster Whisper models
- Optimize Ollama model parameters

## 🔄 Updates & Maintenance

### Updating Models
```bash
# Update phi3 model
ollama pull phi3

# Check for newer models
ollama list
```

### Log Maintenance
```python
# Clear old logs
python -c "
from utils import Logger
logger = Logger()
logger.clear_logs()
print('Logs cleared')
"
```

### Configuration Reset
```python
# Reset to default configuration
python -c "
from utils import ConfigManager
config_manager = ConfigManager()
config_manager.reset_config()
print('Config reset')
"
```

## 📈 Performance Monitoring

### Statistics Tracking
- **Command Success Rate**: Monitor execution success/failure
- **Response Time**: Track voice-to-execution latency
- **Model Performance**: Whisper transcription accuracy
- **System Resource Usage**: Memory and CPU monitoring

### Health Checks
```python
# Run system health check
python -c "
from utils import Logger, PlatformUtils
logger = Logger()
platform_utils = PlatformUtils()

print('Log Stats:', logger.get_log_stats())
print('System Info:', platform_utils.get_system_info())
print('Platform:', platform_utils.get_platform())
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama Team**: For the excellent local LLM platform
- **OpenAI**: For the Whisper speech recognition model
- **Faster-Whisper**: For the optimized Whisper implementation
- **Python Community**: For the incredible ecosystem of libraries

## 📞 Support

For issues, questions, or contributions:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check this README and code comments
- **Community**: Join discussions in GitHub Discussions

---

**Madefor developers who love talking to their terminals**
