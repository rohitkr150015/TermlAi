
"""
TermiAI - LLM Interface Module
Handles communication with Ollama API using phi3 model for command generation
"""
import json
import asyncio
import platform
from typing import Optional, Dict, Any

try:
    import aiohttp
except ImportError:
    print("❌ Missing aiohttp dependency")
    print("Install with: pip install aiohttp")
    exit(1)


class LLMProcessor:
    """Handles LLM interactions with Ollama + phi3"""

    def __init__(self, config: dict):
        """Initialize LLM processor with Ollama configuration"""
        self.config = config
        self.ollama_host = config.get('ollama_host', 'http://localhost:11434')
        self.model_name = config.get('ollama_model', 'phi3')
        self.timeout = config.get('llama_timeout', 30)

        # Get system information for context
        self.system_platform = platform.system().lower()
        self.system_info = self._get_system_info()

        print(f"🧠 LLM Processor initialized - Model: {self.model_name}")

    def _get_system_info(self) -> dict:
        """Get system information for command generation context"""
        return {
            'platform': self.system_platform,
            'shell': self._get_default_shell(),
            'home_dir': self._get_home_directory(),
            'current_dir': self._get_current_directory()
        }

    def _get_default_shell(self) -> str:
        """Get the default shell for the current platform"""
        if self.system_platform == 'windows':
            return 'cmd'
        else:
            return 'bash'

    def _get_home_directory(self) -> str:
        """Get user home directory path"""
        import os
        return os.path.expanduser('~')

    def _get_current_directory(self) -> str:
        """Get current working directory"""
        import os
        return os.getcwd()

    async def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_host}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model['name'] for model in data.get('models', [])]

                        if self.model_name in models:
                            print(f"✅ Connected to Ollama - Model '{self.model_name}' available")
                            return True
                        else:
                            print(f"❌ Model '{self.model_name}' not found. Available models: {models}")
                            print(f"💡 Install with: ollama pull {self.model_name}")
                            return False
                    else:
                        print(f"❌ Ollama connection failed - Status: {response.status}")
                        return False

        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
            return False

    async def text_to_command(self, user_input: str) -> Optional[str]:
        """Convert natural language to terminal command using Ollama + phi3"""
        try:
            # Create system prompt with context
            system_prompt = self._create_system_prompt()

            # Create user prompt
            user_prompt = self._create_user_prompt(user_input)

            # Send request to Ollama
            command = await self._query_ollama(system_prompt, user_prompt)

            if command:
                # Clean and validate the command
                return self._clean_command(command)
            else:
                return "# Could not generate command - please try rephrasing"

        except Exception as e:
            print(f"❌ LLM processing error: {e}")
            return f"# Error: {str(e)}"

    def _create_system_prompt(self) -> str:
        """Create system prompt with platform-specific context"""
        return f"""You are TermiAI, a terminal command generator. Convert natural language requests to shell commands.

SYSTEM CONTEXT:
- Platform: {self.system_info['platform']}
- Shell: {self.system_info['shell']}
- Home Directory: {self.system_info['home_dir']}
- Current Directory: {self.system_info['current_dir']}

RULES:
1. Return ONLY the command, no explanations or markdown
2. Use platform-appropriate commands:
   - Linux/macOS: bash/zsh commands (ls, mkdir, cd, etc.)
   - Windows: cmd commands (dir, mkdir, cd, etc.)
3. Handle common shortcuts:
   - "documents" → ~/Documents or %USERPROFILE%\\Documents
   - "desktop" → ~/Desktop or %USERPROFILE%\\Desktop
   - "downloads" → ~/Downloads or %USERPROFILE%\\Downloads
4. For ambiguous requests, choose the most common interpretation
5. For dangerous commands, return: # DANGEROUS: [explanation]
6. For unclear requests, return: # UNCLEAR: [suggestion]

COMMAND EXAMPLES:
- "list files" → ls -la (Linux/Mac) or dir (Windows)
- "create folder projects" → mkdir projects
- "go to documents" → cd ~/Documents (Linux/Mac) or cd %USERPROFILE%\\Documents (Windows)
- "show disk space" → df -h (Linux/Mac) or dir (Windows)
- "current directory" → pwd (Linux/Mac) or cd (Windows)"""

    def _create_user_prompt(self, user_input: str) -> str:
        """Create user prompt for command generation"""
        return f"Convert this request to a shell command: {user_input}"

    async def _query_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Send query to Ollama API"""
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent command generation
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 100  # Limit response length
                }
            }

            timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                        f"{self.ollama_host}/api/chat",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                ) as response:

                    if response.status == 200:
                        data = await response.json()
                        command = data.get('message', {}).get('content', '').strip()
                        return command
                    else:
                        error_text = await response.text()
                        print(f"❌ Ollama API error {response.status}: {error_text}")
                        return None

        except asyncio.TimeoutError:
            print("❌ Ollama request timed out")
            return None
        except Exception as e:
            print(f"❌ Ollama query error: {e}")
            return None

    def _clean_command(self, command: str) -> str:
        """Clean and validate the generated command"""
        # Remove common markdown formatting
        command = command.strip()

        # Remove code block markers
        if command.startswith('```'):
            lines = command.split('\n')
            if len(lines) > 1:
                command = '\n'.join(lines[1:-1]) if lines[-1].strip() == '```' else '\n'.join(lines[1:])
            else:
                command = command.replace('```', '')

        # Remove backticks
        command = command.replace('`', '')

        # Remove extra whitespace
        command = ' '.join(command.split())

        # Platform-specific path corrections
        if self.system_platform == 'windows':
            # Convert Unix-style paths to Windows paths
            command = command.replace('~/', '%USERPROFILE%\\')
            command = command.replace('/', '\\')

        return command

    async def explain_command(self, command: str) -> str:
        """Get explanation for a command"""
        try:
            system_prompt = f"""You are a helpful terminal command explainer. 
Explain what the given command does in simple terms.
Platform: {self.system_info['platform']}
Keep explanations concise and user-friendly."""

            user_prompt = f"Explain this command: {command}"

            explanation = await self._query_ollama(system_prompt, user_prompt)
            return explanation or "Could not generate explanation"

        except Exception as e:
            return f"Error generating explanation: {e}"

    async def suggest_alternative(self, failed_command: str, error_message: str) -> str:
        """Suggest alternative command when one fails"""
        try:
            system_prompt = f"""You are a terminal command troubleshooter.
Given a failed command and error message, suggest a corrected version.
Platform: {self.system_info['platform']}
Return ONLY the corrected command, no explanations."""

            user_prompt = f"Command failed: {failed_command}\nError: {error_message}\nSuggest correction:"

            suggestion = await self._query_ollama(system_prompt, user_prompt)
            return self._clean_command(suggestion) if suggestion else "# Could not suggest alternative"

        except Exception as e:
            return f"# Error suggesting alternative: {e}"

    def get_model_info(self) -> dict:
        """Get information about the current LLM configuration"""
        return {
            'model_name': self.model_name,
            'ollama_host': self.ollama_host,
            'timeout': self.timeout,
            'system_platform': self.system_platform,
            'system_info': self.system_info
        }


# Utility function for testing
async def test_llm_processor():
    """Test the LLM processor functionality"""
    config = {
        'ollama_host': 'http://localhost:11434',
        'ollama_model': 'phi3',
        'llama_timeout': 30
    }

    processor = LLMProcessor(config)

    # Test connection
    if not await processor.test_connection():
        print("❌ LLM connection test failed")
        return

    print("✅ LLM connection test passed")

    # Test command generation
    test_inputs = [
        "list files in current directory",
        "create a folder named test_folder",
        "show system information",
        "go to documents folder",
        "delete file named test.txt"
    ]

    for test_input in test_inputs:
        print(f"\n🧠 Testing: '{test_input}'")
        command = await processor.text_to_command(test_input)
        print(f"💻 Generated: {command}")

        if command and not command.startswith('#'):
            explanation = await processor.explain_command(command)
            print(f"📝 Explanation: {explanation}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_llm_processor())