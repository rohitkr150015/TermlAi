"""
TermiAI - Main Entry Point
Voice-controlled terminal assistant with Ollama + phi3 and Faster-Whisper
"""
import os
import sys
import json
import asyncio
from pathlib import Path

from voice_input import VoiceProcessor
from llm_interface import LLMProcessor
from executor import CommandExecutor
from utils import Logger, ConfigManager, PlatformUtils


class TermiAI:
    """Main TermiAI application class"""

    def __init__(self):
        """Initialize TermiAI components"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.logger = Logger()
        self.platform_utils = PlatformUtils()

        # Initialize components
        print("🤖 Initializing TermiAI...")
        self.voice_processor = VoiceProcessor(self.config)
        self.llm_processor = LLMProcessor(self.config)
        self.command_executor = CommandExecutor(self.config, self.logger)

        print(f"✅ TermiAI initialized on {self.platform_utils.get_platform()}")
        self._print_welcome()

    def _print_welcome(self):
        """Print welcome message and instructions"""
        print("\n" + "=" * 60)
        print("🎙️  TERMIAI - Voice-Controlled Terminal Assistant")
        print("=" * 60)
        print("🔧 Using: Faster-Whisper + Ollama (phi3)")
        print("🎯 Commands:")
        print("   • Press ENTER to start listening")
        print("   • Say 'history' to view recent commands")
        print("   • Say 'exit' or 'quit' to stop")
        print("   • Type 'config' to view settings")
        print("=" * 60)

    async def run_interactive_session(self):
        """Main interactive session loop"""
        print("\n🚀 Starting interactive session...")

        # Test Ollama connection
        if not await self.llm_processor.test_connection():
            print("❌ Failed to connect to Ollama. Please ensure Ollama is running.")
            print("💡 Start Ollama with: ollama serve")
            return

        while True:
            try:
                # Wait for user input
                user_input = input("\n🎙️  Press ENTER to speak (or type command): ").strip()

                # Handle typed commands
                if user_input:
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        break
                    elif user_input.lower() == 'config':
                        self._show_config()
                        continue
                    elif user_input.lower() == 'history':
                        self._show_history()
                        continue
                    else:
                        # Process typed command directly
                        await self._process_command(user_input, input_type="text")
                        continue

                # Voice input mode
                print("🎙️  Listening... (speak now)")
                speech_text = await self.voice_processor.listen_and_transcribe()

                if not speech_text:
                    print("❌ No speech detected or transcription failed")
                    continue

                print(f"🗣️  You said: '{speech_text}'")

                # Check for exit commands
                if any(word in speech_text.lower() for word in ['exit', 'quit', 'stop', 'bye']):
                    print("👋 Goodbye!")
                    break

                # Process the voice command
                await self._process_command(speech_text, input_type="voice")

            except KeyboardInterrupt:
                print("\n👋 Session terminated by user")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                self.logger.log_error(f"Main loop error: {e}")
                continue

    async def _process_command(self, user_input: str, input_type: str = "voice"):
        """Process user input and execute command"""
        try:
            # Handle special commands
            if 'history' in user_input.lower():
                self._show_history()
                return

            # Generate command using LLM
            print("🧠 Generating command...")
            command = await self.llm_processor.text_to_command(user_input)

            if not command:
                print("❌ Could not generate command")
                return

            if command.startswith('#'):
                print(f"💭 {command}")
                return

            print(f"💻 Generated command: {command}")

            # Get user confirmation
            if self.config.get('confirmation_required', True):
                confirmation = input("Execute this command? (y/n/e=edit): ").strip().lower()

                if confirmation in ['n', 'no']:
                    print("❌ Command cancelled")
                    self.logger.log_interaction(user_input, command, False, input_type)
                    return
                elif confirmation in ['e', 'edit']:
                    edited_command = input(f"Edit command [{command}]: ").strip()
                    if edited_command:
                        command = edited_command
                        print(f"💻 Updated command: {command}")

            # Execute command
            success, output = await self.command_executor.execute_command(command)

            if success:
                print("✅ Command executed successfully")
                if output.strip():
                    print(f"📄 Output:\n{output}")
            else:
                print(f"❌ Command failed: {output}")

            # Log interaction
            self.logger.log_interaction(user_input, command, success, input_type)

        except Exception as e:
            print(f"❌ Error processing command: {e}")
            self.logger.log_error(f"Command processing error: {e}")

    def _show_config(self):
        """Display current configuration"""
        print("\n🔧 Current Configuration:")
        for key, value in self.config.items():
            print(f"   {key}: {value}")

    def _show_history(self):
        """Display command history"""
        history = self.logger.get_recent_history(10)
        if not history:
            print("📝 No command history found")
            return

        print("\n📜 Recent Commands:")
        for entry in history:
            status = "✅" if entry.get('executed', False) else "❌"
            timestamp = entry.get('timestamp', '')[:19]
            input_text = entry.get('user_input', '')
            command = entry.get('command', '')
            input_type = entry.get('input_type', 'unknown')

            print(f"{status} [{timestamp}] ({input_type}) {input_text} → {command}")


def main():
    """Main entry point"""
    try:
        # Ensure logs directory exists
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Create and run TermiAI
        app = TermiAI()
        asyncio.run(app.run_interactive_session())

    except KeyboardInterrupt:
        print("\n👋 TermiAI shutting down...")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()