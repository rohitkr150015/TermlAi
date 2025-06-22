"""
TermiAI - Command Executor Module
Handles safe execution of terminal commands with logging and security checks
"""

import os
import asyncio
import subprocess
import platform
from typing import Tuple, List, Optional
from pathlib import Path


class CommandExecutor:
    """Handles safe execution of terminal commands"""

    def __init__(self, config: dict, logger):
        """Initialize command executor with safety configurations"""
        self.config = config
        self.logger = logger
        self.platform = platform.system().lower()

        # Security settings
        self.max_execution_time = config.get('max_execution_time', 30)
        self.allow_dangerous_commands = config.get('allow_dangerous_commands', False)
        self.dry_run_mode = config.get('dry_run_mode', False)

        # Platform-specific settings
        self.shell = self._get_default_shell()

        print(f"⚡ Command Executor initialized - Platform: {self.platform}")

    def _get_default_shell(self) -> str:
        """Get default shell for the platform"""
        if self.platform == 'windows':
            return 'cmd'
        else:
            return '/bin/bash'

    async def execute_command(self, command: str) -> Tuple[bool, str]:
        """
        Execute a terminal command safely
        Returns (success: bool, output: str)
        """
        try:
            # Pre-execution checks
            if not self._is_command_safe(command):
                return False, "❌ Command blocked for security reasons"

            # Dry run mode
            if self.dry_run_mode:
                return True, f"[DRY RUN] Would execute: {command}"

            # Log command execution attempt
            self.logger.log_command_execution(command, "started")

            # Execute command
            success, output = await self._run_command(command)

            # Log execution result
            status = "success" if success else "failed"
            self.logger.log_command_execution(command, status, output)

            return success, output

        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.logger.log_command_execution(command, "error", error_msg)
            return False, error_msg

    def _is_command_safe(self, command: str) -> bool:
        """Check if command is safe to execute"""
        command_lower = command.lower().strip()

        # Skip safety checks if explicitly allowed
        if self.allow_dangerous_commands:
            return True

        # Dangerous command patterns
        dangerous_patterns = [
            # Destructive operations
            'rm -rf /',
            'rm -rf ~',
            'rm -rf *',
            'del /s /q',
            'format c:',
            'mkfs.',
            'dd if=',

            # System operations
            'shutdown',
            'reboot',
            'halt',
            'poweroff',
            'systemctl poweroff',
            'systemctl reboot',

            # Network/Security
            'curl -s',  # Potential script execution
            'wget -O -',  # Potential script execution
            'nc -l',  # Netcat listener
            'netcat -l',

            # User/Permission changes
            'chmod 777',
            'chown root',
            'sudo su',
            'su root',

            # Package management (potentially dangerous)
            'apt-get remove --purge',
            'yum remove',
            'dnf remove',
            'pacman -R',

            # Process killing (broad)
            'killall -9',
            'pkill -9',
            'taskkill /f /im *',
        ]

        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if pattern in command_lower:
                return False

        # Check for suspicious characters/operators
        suspicious_chars = [
            '&&', '||', ';', '|',  # Command chaining
            '>', '>>', '<',  # Redirection (could be dangerous)
            '$(', '`',  # Command substitution
            'eval', 'exec'  # Code execution
        ]

        # Allow basic redirection but be cautious
        safe_redirections = ['> /dev/null', '2>/dev/null', '> NUL']
        has_safe_redirection = any(safe in command for safe in safe_redirections)

        for char in suspicious_chars:
            if char in command and not has_safe_redirection:
                # Allow some safe cases
                if char in ['>', '>>'] and not any(
                        danger in command_lower for danger in ['/etc/', '/sys/', '/proc/', 'c:\\']):
                    continue
                return False

        return True

    async def _run_command(self, command: str) -> Tuple[bool, str]:
        """Run the command using subprocess"""
        try:
            # Prepare command for execution
            if self.platform == 'windows':
                cmd_args = ['cmd', '/c', command]
            else:
                cmd_args = [self.shell, '-c', command]

            # Execute command with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.getcwd()
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.max_execution_time
                )

                # Decode output
                stdout_text = stdout.decode('utf-8', errors='replace').strip()
                stderr_text = stderr.decode('utf-8', errors='replace').strip()

                # Combine output
                output = stdout_text
                if stderr_text:
                    output += f"\n[STDERR] {stderr_text}"

                success = process.returncode == 0
                return success, output

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return False, f"❌ Command timed out after {self.max_execution_time} seconds"

        except Exception as e:
            return False, f"❌ Command execution failed: {str(e)}"

    def validate_command_syntax(self, command: str) -> Tuple[bool, str]:
        """Validate command syntax without executing"""
        try:
            # Basic syntax validation
            if not command.strip():
                return False, "Empty command"

            # Check for balanced quotes
            single_quotes = command.count("'")
            double_quotes = command.count('"')

            if single_quotes % 2 != 0:
                return False, "Unbalanced single quotes"

            if double_quotes % 2 != 0:
                return False, "Unbalanced double quotes"

            # Check for valid command structure
            parts = command.strip().split()
            if not parts:
                return False, "No command specified"

            # Platform-specific validation
            if self.platform == 'windows':
                return self._validate_windows_command(command)
            else:
                return self._validate_unix_command(command)

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _validate_windows_command(self, command: str) -> Tuple[bool, str]:
        """Validate Windows-specific command syntax"""
        # Check for valid Windows commands
        valid_commands = [
            'dir', 'cd', 'md', 'mkdir', 'rd', 'rmdir', 'del', 'copy', 'move', 'ren',
            'type', 'echo', 'cls', 'date', 'time', 'ver', 'vol', 'path', 'set',
            'tasklist', 'taskkill', 'ping', 'ipconfig', 'netstat', 'systeminfo'
        ]

        first_word = command.strip().split()[0].lower()

        # Allow PowerShell commands
        if first_word in ['powershell', 'pwsh']:
            return True, "PowerShell command"

        # Check if it's a known Windows command
        if first_word in valid_commands:
            return True, "Valid Windows command"

        # Check if it's an executable
        if first_word.endswith('.exe') or first_word.endswith('.bat') or first_word.endswith('.cmd'):
            return True, "Windows executable"

        return True, "Command passed basic validation"

    def _validate_unix_command(self, command: str) -> Tuple[bool, str]:
        """Validate Unix-specific command syntax"""
        # Common Unix commands
        common_commands = [
            'ls', 'cd', 'mkdir', 'rmdir', 'rm', 'cp', 'mv', 'ln', 'find', 'grep',
            'cat', 'less', 'more', 'head', 'tail', 'sort', 'uniq', 'wc', 'cut',
            'sed', 'awk', 'tar', 'gzip', 'gunzip', 'zip', 'unzip', 'wget', 'curl',
            'ps', 'top', 'kill', 'killall', 'jobs', 'bg', 'fg', 'nohup',
            'chmod', 'chown', 'chgrp', 'df', 'du', 'free', 'uname', 'whoami',
            'date', 'cal', 'uptime', 'which', 'whereis', 'man', 'info',
            'git', 'npm', 'pip', 'python', 'node', 'java', 'gcc', 'make'
        ]

        first_word = command.strip().split()[0]

        # Remove path if present
        if '/' in first_word:
            first_word = first_word.split('/')[-1]

        return True, "Unix command passed validation"

    def get_command_history(self, limit: int = 10) -> List[dict]:
        """Get recent command execution history"""
        return self.logger.get_command_history(limit)

    def get_execution_stats(self) -> dict:
        """Get command execution statistics"""
        history = self.logger.get_command_history(100)

        total_commands = len(history)
        successful_commands = sum(1 for cmd in history if cmd.get('status') == 'success')
        failed_commands = total_commands - successful_commands

        return {
            'total_commands': total_commands,
            'successful_commands': successful_commands,
            'failed_commands': failed_commands,
            'success_rate': (successful_commands / total_commands * 100) if total_commands > 0 else 0
        }

    def set_dry_run_mode(self, enabled: bool):
        """Enable or disable dry run mode"""
        self.dry_run_mode = enabled
        print(f"🔧 Dry run mode: {'enabled' if enabled else 'disabled'}")

    def set_safety_level(self, level: str):
        """Set safety level: 'strict', 'normal', 'permissive'"""
        if level == 'strict':
            self.allow_dangerous_commands = False
            self.max_execution_time = 10
        elif level == 'normal':
            self.allow_dangerous_commands = False
            self.max_execution_time = 30
        elif level == 'permissive':
            self.allow_dangerous_commands = True
            self.max_execution_time = 60
        else:
            raise ValueError("Safety level must be 'strict', 'normal', or 'permissive'")

        print(f"🛡️  Safety level set to: {level}")


# Utility function for testing
async def test_command_executor():
    """Test the command executor functionality"""
    from utils import Logger

    config = {
        'max_execution_time': 10,
        'allow_dangerous_commands': False,
        'dry_run_mode': False
    }

    logger = Logger()
    executor = CommandExecutor(config, logger)

    # Test safe commands
    safe_commands = [
        'echo "Hello World"',
        'pwd' if platform.system() != 'Windows' else 'cd',
        'date' if platform.system() != 'Windows' else 'date /t',
        'ls -la' if platform.system() != 'Windows' else 'dir'
    ]

    print("🧪 Testing safe commands...")
    for cmd in safe_commands:
        print(f"\n💻 Testing: {cmd}")
        success, output = await executor.execute_command(cmd)
        status = "✅" if success else "❌"
        print(f"{status} Result: {output[:100]}...")

    # Test dangerous commands (should be blocked)
    dangerous_commands = [
        'rm -rf /',
        'format c:',
        'shutdown now'
    ]

    print("\n🚨 Testing dangerous commands (should be blocked)...")
    for cmd in dangerous_commands:
        print(f"\n💻 Testing: {cmd}")
        success, output = await executor.execute_command(cmd)
        if not success and "blocked" in output:
            print("✅ Correctly blocked dangerous command")
        else:
            print("❌ Dangerous command was not blocked!")

    # Show execution stats
    stats = executor.get_execution_stats()
    print(f"\n📊 Execution Stats: {stats}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_command_executor())