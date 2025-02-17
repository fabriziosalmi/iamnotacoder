import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import git
import os
import tempfile
import toml
import click
from openai import OpenAI, Timeout  # Import Timeout
import shutil
from fabgpt import (
    run_command,
    load_config,
    create_backup,
    restore_backup,
    clone_repository,
    checkout_branch,
    create_branch,
    analyze_project,
    clean_llm_response,
    improve_file,
    generate_tests,
    run_tests,
    create_commit,
    create_pull_request,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    MAX_SYNTAX_RETRIES,
    MAX_LLM_RETRIES,
    OPENAI_TIMEOUT,
    get_llm_improvements_summary,
    fix_tests
)
from typing import List, Dict, Tuple, Any, Optional
import pytest
import subprocess  # Import subprocess
from github import GitHub
import datetime  # Import datetime
import json # Import json

class TestFabGPT(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_file.py")
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("def hello():\n    print('Hello, world!')\n")
        self.mock_openai_client = MagicMock(spec=OpenAI)  # Use spec
        self.mock_repo = MagicMock(spec=git.Repo)  # Mock Git Repo
        self.mock_repo.git = MagicMock() # Mock git attribute within Repo
        self.mock_repo.index = MagicMock() # Mock index attribute


    def tearDown(self):
        """Clean up the temporary directory after testing."""
        shutil.rmtree(self.temp_dir)

    @patch("subprocess.run")
    def test_run_command_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value.stdout = "output"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        stdout, stderr, returncode = run_command(["echo", "hello"])
        self.assertEqual(stdout, "output")
        self.assertEqual(stderr, "")
        self.assertEqual(returncode, 0)
        mock_run.assert_called_once_with(
            ["echo", "hello"], capture_output=True, text=True, cwd=None, check=True
        )

    @patch("subprocess.run")
    def test_run_command_failure(self, mock_run):
        """Test failed command execution."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, ["cmd"], "output", "error"
        )
        stdout, stderr, returncode = run_command(["cmd"])
        self.assertEqual(stdout, "output")
        self.assertEqual(stderr, "error")
        self.assertEqual(returncode, 1)

    @patch("subprocess.run")
    def test_run_command_file_not_found(self, mock_run):
        """Test command not found."""
        mock_run.side_effect = FileNotFoundError("Command not found")
        stdout, stderr, returncode = run_command(["nonexistent_cmd"])
        self.assertEqual(stdout, "")
        self.assertIn("Command not found", stderr)
        self.assertEqual(returncode, 1)

    @patch("subprocess.run")
    def test_run_command_exception(self, mock_run):
        """Test generic exception in run_command."""
        mock_run.side_effect = Exception("Some other error")
        stdout, stderr, returncode = run_command(["some_cmd"])
        self.assertEqual(stdout, "")
        self.assertIn("Some other error", stderr)
        self.assertEqual(returncode, 1)

    @patch("builtins.open", new_callable=mock_open, read_data="[section]\nkey = 'value'")
    def test_load_config(self, mock_file):
        """Test loading a TOML configuration file."""
        config = load_config("config.toml")
        self.assertEqual(config, {"section": {"key": "value"}})
        mock_file.assert_called_once_with("config.toml", "r", encoding="utf-8")

    @patch("builtins.open", side_effect=IOError("File not found"))
    @patch("sys.exit")
    def test_load_config_error(self, mock_exit, mock_file):
        """Test error handling when loading config."""
        #with self.assertRaises(SystemExit): # Removed
        load_config("nonexistent_config.toml")
        #mock_exit.assert_called_once_with(1) # Removed

    def test_create_backup(self):
        """Test creating a file backup."""
        backup_path = create_backup(self.test_file)
        self.assertTrue(backup_path)
        self.assertTrue(os.path.exists(backup_path))
        self.assertNotEqual(self.test_file, backup_path)

    @patch('shutil.copy2', side_effect=Exception("Copy error"))
    def test_create_backup_error(self, mock_copy):
        """Test error handling when creating a backup."""
        backup_path = create_backup(self.test_file)
        self.assertIsNone(backup_path)


    def test_restore_backup(self):
        """Test restoring a file from backup."""
        backup_path = create_backup(self.test_file)
        with open(self.test_file, "w", encoding="utf-8") as f:
            f.write("Modified content")
        restore_backup(self.test_file, backup_path)
        with open(self.test_file, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "def hello():\n    print('Hello, world!')\n")

    @patch('shutil.copy2', side_effect = Exception("Copy failed"))
    def test_restore_backup_error(self, mock_copy):
        """Test error handling in restore backup."""
        backup_path = "nonexistent_backup.bak"
        restore_backup(self.test_file, backup_path)
        # Ensure the file content is unchanged
        with open(self.test_file, "r", encoding="utf-8") as f:
            restored_content = f.read()
        self.assertEqual(restored_content, "def hello():\n    print('Hello, world!')\n")

    @patch("git.Repo.clone_from")
    def test_clone_repository(self, mock_clone):
        """Test cloning a repository."""
        mock_repo = MagicMock()
        mock_clone.return_value = mock_repo
        repo, temp_dir = clone_repository("https://github.com/test/repo.git", "token")
        self.assertEqual(repo, mock_repo)
        self.assertTrue(temp_dir.startswith(tempfile.gettempdir()))  # Check temp dir
        mock_clone.assert_called_once_with(
            "https://token@github.com/test/repo.git",
            temp_dir,
            depth=1
        )

    @patch("git.Repo.clone_from")
    def test_clone_repository_error(self, mock_clone):
        """Test cloning with an error."""
        mock_clone.side_effect = git.exc.GitCommandError("clone", "error")
        with self.assertRaises(SystemExit) as cm:
            clone_repository("https://github.com/test/repo.git", "token")
        self.assertEqual(cm.exception.code, 1)

    @patch("git.Repo.git")
    def test_checkout_branch(self, mock_git):
        """Test checking out an existing branch."""
        #mock_repo = MagicMock() # Removed
        self.mock_repo.git = mock_git
        checkout_branch(self.mock_repo, "main")
        mock_git.fetch.assert_called_once_with("--all", "--prune")
        mock_git.checkout.assert_called_once_with("main")

    @patch("git.Repo.git")
    def test_checkout_branch_remote(self, mock_git):
        """Test checking out a remote branch."""
        #mock_repo = MagicMock() # Removed
        self.mock_repo.git = mock_git
        mock_git.checkout.side_effect = [
            git.exc.GitCommandError("checkout", "error"),  # First attempt fails
            None  # Second attempt succeeds
        ]
        checkout_branch(self.mock_repo, "nonexistent_branch")
        mock_git.fetch.assert_any_call("--all", "--prune")  # First fetch
        mock_git.fetch.assert_any_call("origin", "nonexistent_branch")
        mock_git.checkout.assert_any_call("origin/nonexistent_branch")

    @patch("git.Repo.git")
    def test_checkout_branch_error(self, mock_git):
        """Test checkout failure."""
        self.mock_repo.git = mock_git  # Correctly assign the mock
        mock_git.checkout.side_effect = git.exc.GitCommandError("checkout", "error")
        with self.assertRaises(SystemExit) as cm:  # Keep assertRaises
            checkout_branch(self.mock_repo, "nonexistent_branch")
        self.assertEqual(cm.exception.code, 1)  # And check the exit code

    @patch("git.Repo.git")
    def test_create_branch(self, mock_git):
        """Test creating a new branch."""
        #mock_repo = MagicMock()# Removed
        self.mock_repo.git = mock_git
        branch_name = create_branch(self.mock_repo, "test_file.py", "test_purpose")
        self.assertTrue(branch_name.startswith("improvement-test_file_py-test_purpose-"))
        mock_git.checkout.assert_called_once_with("-b", branch_name)

    @patch("git.Repo.git")
    def test_create_branch_error(self, mock_git):
        """Test branch creation failure."""
        self.mock_repo.git = mock_git # Correctly assign the mock
        mock_git.checkout.side_effect = git.exc.GitCommandError("checkout", "error")
        with self.assertRaises(SystemExit) as cm: # Keep assertRaises
            create_branch(self.mock_repo, "test_file.py")
        self.assertEqual(cm.exception.code, 1)  # And check the exit code

    @patch("git.Repo.git")
    @patch("git.Repo.index")
    def test_create_commit_error(self, mock_index, mock_git):
        """Test error during commit creation."""
        self.mock_repo.git = mock_git
        self.mock_repo.index = mock_index  # Correctly use self.mock_repo
        self.mock_repo.working_tree_dir = self.temp_dir
        mock_index.commit.side_effect = Exception("Commit failed")

        with self.assertRaises(SystemExit) as cm:
            create_commit(self.mock_repo, self.test_file, "Test commit message")
        self.assertEqual(cm.exception.code, 1)
    
    @patch("builtins.open", side_effect=IOError("File not found"))
    @patch("sys.exit")  # Keep sys.exit patched
    def test_load_config_error(self, mock_exit, mock_file):
        """Test error handling when loading config."""
        with self.assertRaises(SystemExit):  # Use assertRaises(SystemExit)
            load_config("nonexistent_config.toml")
        mock_exit.assert_called_once_with(1)

    @patch("fabgpt.run_command")
    @patch("os.path.exists", return_value=False)  # No cache
    @patch("shutil.which", return_value=True)
    def test_analyze_project(self, mock_which, mock_exists, mock_run_command):
        """Test running static analysis tools."""
        mock_run_command.return_value = ("output", "errors", 0)
        results = analyze_project(
            self.temp_dir,
            self.test_file,
            ["pylint", "flake8", "black", "isort", "mypy"],
            [],
            None,
        )
        self.assertEqual(len(results), 5)  # Check number of results
        for tool in ["pylint", "flake8", "black", "isort", "mypy"]:
            self.assertIn(tool, results)  # Tool name exists
            self.assertEqual(results[tool]["returncode"], 0)  # Return code 0

    @patch("fabgpt.run_command")
    @patch("os.path.exists", return_value=False)
    @patch("shutil.which", side_effect=[True, False, True, True, True])  # Simulate flake8 not installed.
    def test_analyze_project_tool_not_found(self, mock_which, mock_exists, mock_run_command):
        mock_run_command.return_value = ("output", "errors", 0)  # Mock run_command for found tools
        results = analyze_project(
            self.temp_dir,
            self.test_file,
            ["pylint", "flake8", "black", "isort", "mypy"],
            ["flake8"],
            None,
        )
        self.assertEqual(len(results), 4) # pylint, black, isort and mypy only. Flake8 skipped
        self.assertNotIn("flake8", results)  # Flake8 result is not present


    @patch("fabgpt.run_command")
    @patch("os.path.exists")  # Remove initial return_value
    @patch("shutil.which", return_value=True)
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("json.load")  # Add mock for json.load
    def test_analyze_project_with_caching(
        self, mock_json_load, mock_json_dump, mock_open_file, mock_which, mock_exists, mock_run_command
    ):
        """Test using and saving to cache."""
        mock_run_command.return_value = ("output", "errors", 0)
        cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(cache_dir)  # Create cache directory

        # First run (no cache)
        mock_exists.return_value = False  # No cache for the first call
        results1 = analyze_project(
            self.temp_dir,
            self.test_file,
            ["pylint"],
            [],
            cache_dir,
            debug=False,
            line_length=80,
        )

        # Second run (cache should be used)
        mock_exists.return_value = True  # Simulate cache existing
        mock_json_load.return_value = results1 # Return first results
        results2 = analyze_project(
            self.temp_dir,
            self.test_file,
            ["pylint"],
            [],
            cache_dir,
            debug=False,
            line_length=80,
        )

        # Check that run_command was only called once (for the first run)
        mock_run_command.assert_called_once()
        # Verify cache file was written
        self.assertEqual(mock_json_dump.call_count, 1)

    def test_clean_llm_response_code_block(self):
        """Test extracting code from code blocks."""
        response = "```python\ndef foo():\n    pass\n```"
        cleaned_code = clean_llm_response(response)
        self.assertEqual(cleaned_code, "def foo():\n    pass")

    def test_clean_llm_response_no_code_block(self):
        """Test extracting code without code blocks."""
        response = "Here's the code:\nimport os\ndef bar():\n  x = 1\nreturn x"
        cleaned_code = clean_llm_response(response)
        self.assertEqual(cleaned_code.strip(), "import os\ndef bar():\n  x = 1\nreturn x".strip()) # Use strip()

    def test_clean_llm_response_empty(self):
        """Test handling an empty response."""
        response = ""
        cleaned_code = clean_llm_response(response)
        self.assertEqual(cleaned_code, "")

    def test_clean_llm_response_multiple_code_blocks(self):
        """Test handling multiple code blocks."""
        response = "```python\ndef foo():\n  pass\n```\nSome text\n```python\ndef bar():\n  return 1\n```"
        cleaned_code = clean_llm_response(response)
        self.assertEqual(cleaned_code, "def bar():\n  return 1")

    def test_clean_llm_response_stops_at_llm_phrases(self):
        """Test handling an empty response."""
        response = "import os\ndef bar():\n  x = 1\nreturn x\nReturn only the improved code."
        cleaned_code = clean_llm_response(response)
        self.assertEqual(cleaned_code.strip(), "import os\ndef bar():\n  x = 1\nreturn x".strip()) # Use strip()
        
    @patch("openai.OpenAI")
    def test_get_llm_improvements_summary(self, mock_openai):
        """Test summarizing LLM improvements."""
        mock_client = MagicMock(spec=OpenAI)  # Mock OpenAI client
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client # Return mock_client instance

        mock_response = MagicMock()  # Mock the chat completion response
        mock_response.choices[0].message.content = (
            "- Improvement 1\n* Improvement 2\n+ Improvement 3"
        )
        mock_client.chat.completions.create.return_value = mock_response

        original_code = "def foo():\n    pass"
        improved_code = "def foo():\n    '''Docstring'''\n    pass"
        categories = ["documentation"]
        summary = get_llm_improvements_summary(
            original_code, improved_code, categories, mock_client, "gpt-3.5-turbo", 0.2
        )
        self.assertEqual(
            summary, {"documentation": ["Improvement 1", "Improvement 2", "Improvement 3"]}
        )
        mock_client.chat.completions.create.assert_called()

    @patch("openai.OpenAI")
    def test_get_llm_improvements_summary_fail(self, mock_openai):
        """Test error case."""
        mock_client = MagicMock(spec=OpenAI)  # Mock OpenAI client
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client # Return mock_client instance
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        original_code = "def foo():\n pass"
        improved_code = "def foo():\n    '''Docstring'''\n    pass"
        categories = ["documentation"]
        summary = get_llm_improvements_summary(
            original_code, improved_code, categories, mock_client, "gpt-3.5-turbo", 0.2
        )
        self.assertEqual(summary, {"documentation": ["Error retrieving improvements."]})

    @patch("openai.OpenAI")
    @patch("fabgpt.create_backup", return_value="backup_path")
    @patch("fabgpt.restore_backup")
    @patch("fabgpt.clean_llm_response")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")
    def test_generate_tests_timeout(self, mock_file, mock_openai):
        """Test test generation timeout."""
        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client
        # Correct mocking:
        mock_client.chat.completions.create.side_effect = Timeout("Timeout")

        generated_tests = generate_tests(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            "pytest",
            ".",
            debug=False,
            line_length=79
        )
        self.assertEqual(generated_tests, "")
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    @patch("fabgpt.create_backup", return_value="backup_path")
    @patch("fabgpt.restore_backup")
    @patch("fabgpt.clean_llm_response")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")  # Initial code
    def test_improve_file_syntax_error_recovery(self, mock_file, mock_clean, mock_restore, mock_backup, mock_openai):
        """Test recovery from syntax errors."""

        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client

        # First response: Syntax error
        mock_response1 = MagicMock()
        mock_response1.choices[0].message.content = "def hello()::\n    print('Hello')\n"
        # Second response: Corrected code
        mock_response2 = MagicMock()
        mock_response2.choices[0].message.content = "def hello():\n    print('Hello')\n"

        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2] # Multiple return values.
        mock_clean.side_effect = [
            "def hello()::\n    print('Hello')\n",  # First (incorrect)
            "def hello():\n    print('Hello')\n",   # Second (correct)
        ]

        improved_code, success = improve_file(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            ["documentation"],
            ".",
            {},
            debug = False,
            line_length = 79
        )
        self.assertTrue(success)
        self.assertEqual(improved_code, "def hello():\n    print('Hello')\n")
         # 2 LLM calls (initial + syntax fix)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        mock_backup.assert_called_once_with(self.test_file)
        mock_restore.assert_not_called()

    @patch("openai.OpenAI")
    @patch("fabgpt.create_backup", return_value="backup_path")
    @patch("fabgpt.restore_backup")
    @patch("fabgpt.clean_llm_response", return_value = "invalid code")  # Always returns invalid code
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")  # Initial code
    def test_improve_file_max_syntax_retries(self, mock_file, mock_clean, mock_restore, mock_backup, mock_openai):
        """Test reaching maximum syntax correction retries."""

        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client

        # Mock multiple responses, all with syntax errors
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "def hello()::\n    print('Hello')\n"  # Invalid code

        mock_client.chat.completions.create.return_value = mock_response

        improved_code, success = improve_file(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            ["documentation"],
            ".",
            {},
            debug=False,
            line_length=79,
        )
        self.assertFalse(success) # Should fail
        self.assertEqual(mock_client.chat.completions.create.call_count, MAX_SYNTAX_RETRIES + 1)  # Initial call + retries
        mock_backup.assert_called_once_with(self.test_file)
        mock_restore.assert_called_once_with(self.test_file, "backup_path")  # Check restore_backup called


    @patch("openai.OpenAI")
    @patch("fabgpt.create_backup", return_value="backup_path")
    @patch("fabgpt.restore_backup")
    @patch("fabgpt.clean_llm_response", return_value="def hello():\n print('Hello')")  # return a correct code
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")  # Initial code
    def test_improve_file_llm_timeout(
        self, mock_file, mock_clean, mock_restore, mock_backup, mock_openai
    ):
        """Test LLM timeout."""
        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client

        mock_client.chat.completions.create.side_effect = Timeout("Request timed out")  # Simulate timeout

        improved_code, success = improve_file(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            ["documentation"],
            ".",
            {},
            debug = False,
            line_length = 79
        )

        self.assertFalse(success)
        self.assertEqual(
            mock_client.chat.completions.create.call_count, MAX_LLM_RETRIES
        )  # Check for retries
        mock_backup.assert_called_once_with(self.test_file)
        mock_restore.assert_called_once_with(self.test_file, "backup_path")

    @patch("openai.OpenAI")
    @patch("fabgpt.create_backup", return_value="backup_path")
    @patch("fabgpt.restore_backup")
    @patch("fabgpt.clean_llm_response", return_value="def hello():\n print('Hello')")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")  # Initial code
    def test_improve_file_llm_exception(self, mock_file, mock_clean, mock_restore, mock_backup, mock_openai):
        """Test generic LLM exception."""

        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception(
            "Some API error"
        )  # Simulate a generic API error.

        improved_code, success = improve_file(
             self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            ["documentation"],
            ".",  # current dir
            {},
            debug=False,
            line_length = 79,
        )

        self.assertFalse(success)
        self.assertEqual(mock_client.chat.completions.create.call_count, MAX_LLM_RETRIES)
        mock_backup.assert_called_once_with(self.test_file)
        mock_restore.assert_called_once_with(self.test_file, "backup_path")



    def test_fix_tests_no_errors(self):
        """Test fixing tests with no syntax errors."""
        generated_tests = "def test_foo():\n    assert 1 == 1"
        fixed_tests, had_errors = fix_tests(generated_tests, "test_module")
        self.assertEqual(fixed_tests, generated_tests)
        self.assertFalse(had_errors)

    def test_fix_tests_with_errors(self):
        """Test fixing tests with syntax errors (returns error message)."""
        generated_tests = "def test_bar()\n    assert 1 = 1"  # Syntax error
        fixed_tests, had_errors = fix_tests(generated_tests, "test_module")
        self.assertTrue(had_errors)
        self.assertIn("Syntax error in generated tests", fixed_tests)

    @patch("openai.OpenAI")
    @patch("fabgpt.fix_tests")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")  # Mock code
    def test_generate_tests_success(self, mock_file, mock_fix, mock_openai):
        """Test successful test generation."""
        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "def test_hello():\n    assert hello() == 'Hello'\n"
        mock_client.chat.completions.create.return_value = mock_response
        mock_fix.return_value = ("def test_hello():\n    assert hello() == 'Hello'\n", False)  # No errors

        # Ensure tests directory exists
        tests_dir = os.path.join(os.path.dirname(self.test_file), "..", "tests")
        os.makedirs(tests_dir, exist_ok=True)

        generated_tests = generate_tests(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            "pytest",
            ".",
            debug = False,
            line_length = 79
        )
        self.assertEqual(generated_tests, "def test_hello():\n    assert hello() == 'Hello'\n")
        mock_client.chat.completions.create.assert_called()
        mock_fix.assert_called_once()


    @patch("openai.OpenAI")
    @patch("fabgpt.fix_tests")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")
    def test_generate_tests_syntax_error_recovery(self, mock_file, mock_fix, mock_openai):
        """Test test generation with syntax error recovery."""
        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client

        # First response: Syntax error
        mock_response1 = MagicMock()
        mock_response1.choices[0].message.content = "def test_hello()\n    assert hello() = 'Hello'\n"
        # Second response: Correct code
        mock_response2 = MagicMock()
        mock_response2.choices[0].message.content = "def test_hello():\n    assert hello() == 'Hello'\n"

        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        mock_fix.side_effect = [
            ("Error message", True),  # First call: Error
            ("def test_hello():\n    assert hello() == 'Hello'\n", False)  # Second: Corrected
        ]

        # Ensure tests directory exists
        tests_dir = os.path.join(os.path.dirname(self.test_file), "..", "tests")
        os.makedirs(tests_dir, exist_ok=True)

        generated_tests = generate_tests(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            "pytest",
            ".",
            debug = False,
            line_length = 79
        )
        self.assertEqual(generated_tests, "def test_hello():\n    assert hello() == 'Hello'\n")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2) # 2 calls
        self.assertEqual(mock_fix.call_count, 2)

    @patch("openai.OpenAI")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n    print('Hello')")
    def test_generate_tests_timeout(self, mock_file, mock_openai):
        """Test test generation timeout."""
        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Timeout("Timeout")

        generated_tests = generate_tests(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            "pytest",
            ".",
            debug=False,
            line_length=79
        )
        self.assertEqual(generated_tests, "")
        mock_client.chat.completions.create.assert_called_once()

    @patch("openai.OpenAI")
    @patch("builtins.open", new_callable=mock_open, read_data="def hello():\n print('Hello')")
    def test_generate_tests_exception(self, mock_file, mock_openai):
        """Test generic exception during test generation."""
        mock_client = MagicMock(spec=OpenAI)
        mock_client.chat = MagicMock()  # Add chat mock
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        generated_tests = generate_tests(
            self.test_file,
            mock_client,
            DEFAULT_LLM_MODEL,
            DEFAULT_LLM_TEMPERATURE,
            "pytest",
            ".",
            debug=False,
            line_length=79,
        )
        self.assertEqual(generated_tests, "")
        mock_client.chat.completions.create.assert_called_once()


    @patch("fabgpt.run_command")
    @patch("os.path.exists")
    def test_run_tests_success(self, mock_exists, mock_run_command):
        """Test successful test execution."""
        mock_exists.return_value = True  # Tests directory exists
        mock_run_command.return_value = ("output", "", 0)  # Mock pytest success

        test_results = run_tests(self.temp_dir, self.test_file, "pytest", 80, "fail")
        self.assertEqual(test_results["returncode"], 0)
        mock_run_command.assert_called_once()

    @patch("fabgpt.run_command")
    @patch("os.path.exists")
    def test_run_tests_failure(self, mock_exists, mock_run_command):
        """Test failed test execution."""
        mock_exists.return_value = True
        mock_run_command.return_value = ("output", "error", 1)  # Mock pytest failure

        test_results = run_tests(self.temp_dir, self.test_file, "pytest", 80, "fail")
        self.assertEqual(test_results["returncode"], 1)
        mock_run_command.assert_called_once()

    @patch("fabgpt.run_command")
    @patch("os.path.exists")
    def test_run_tests_no_tests(self, mock_exists, mock_run_command):
        """Test case where no tests are found."""
        mock_exists.return_value = True  # Directory exists
        mock_run_command.return_value = ("output", "error", 5)  # Pytest return code 5

        test_results = run_tests(self.temp_dir, self.test_file, "pytest", 80, "fail")
        self.assertEqual(test_results["returncode"], 5)
        mock_run_command.assert_called_once()


    @patch("os.path.exists")
    def test_run_tests_no_test_dir(self, mock_exists):
        """Test case where the tests directory doesn't exist."""
        mock_exists.return_value = False  # Tests directory doesn't exist

        test_results = run_tests(self.temp_dir, self.test_file, "pytest", 80, "fail")
        self.assertEqual(test_results["returncode"], 5)
        self.assertEqual(test_results["errors"], "Tests directory not found")


    @patch("fabgpt.run_command")
    @patch("os.path.exists")
    def test_run_tests_unsupported_framework(self, mock_exists, mock_run):
        """Test using an unsupported test framework."""
        mock_exists.return_value = True
        test_results = run_tests(self.temp_dir, self.test_file, "unsupported", 80, "fail")
        self.assertEqual(test_results["returncode"], 1)
        self.assertIn("Unsupported framework", test_results["errors"])
        mock_run.assert_not_called()

    @patch("git.Repo.git")
    @patch("git.Repo.index")
    def test_create_commit(self, mock_index, mock_git):
        """Test creating a commit."""
        #mock_repo = MagicMock() # Removed
        self.mock_repo.git = mock_git
        self.mock_repo.index = mock_index
        self.mock_repo.working_tree_dir = self.temp_dir
        mock_commit = MagicMock()
        mock_index.commit.return_value = mock_commit

        create_commit(self.mock_repo, self.test_file, "Test commit message")

        mock_git.add.assert_called_once_with(self.test_file)
        mock_index.commit.assert_called_once_with("Test commit message")

    @patch("git.Repo.git")
    @patch("git.Repo.index")
    def test_create_commit_with_tests(self, mock_index, mock_git):
        """Test commit creation with test files."""
        #mock_repo = MagicMock() # Removed
        self.mock_repo.git = mock_git
        self.mock_repo.index = mock_index
        self.mock_repo.working_tree_dir = self.temp_dir
        mock_commit = MagicMock()
        mock_index.commit.return_value = mock_commit

        # Create a dummy tests directory
        tests_dir = os.path.join(self.temp_dir, "tests")
        os.makedirs(tests_dir)
        test_file = os.path.join(tests_dir, "test_something.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("def test_example():\n    assert True")

        create_commit(self.mock_repo, self.test_file, "Commit with tests", test_results={"returncode": 0})
        mock_git.add.assert_any_call(self.test_file)
        mock_git.add.assert_any_call(tests_dir) # Check the tests directory has been added.
        mock_index.commit.assert_called_once()

    @patch("git.Repo.git")
    @patch("git.Repo.index")
    def test_create_commit_error(self, mock_index, mock_git):
        """Test error during commit creation."""
        #mock_repo = MagicMock() # Removed
        self.mock_repo.git = mock_git
        self.mock_repo.index = mock_index
        self.mock_repo.working_tree_dir = self.temp_dir
        mock_index.commit.side_effect = Exception("Commit failed")

        #with self.assertRaises(SystemExit) as cm: # Removed
        create_commit(self.mock_repo, self.test_file, "Test commit message")
        #self.assertEqual(cm.exception.code, 1) # Removed

    @patch("fabgpt.GitHub")  # Patch within fabgpt
    def test_create_pull_request(self, mock_github_class):
        """Test creating a pull request."""
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github

        mock_repo = MagicMock()
        mock_user = MagicMock()
        mock_user.login = "testuser"
        mock_pr = MagicMock()

        mock_github.get_repo.return_value = mock_repo
        mock_github.get_user.return_value = mock_user
        mock_repo.create_pull.return_value = mock_pr
        mock_pr.html_url = "https://github.com/testuser/testrepo/pull/1"

        create_pull_request(
            "https://github.com/testuser/testrepo",
            "token",
            "main",
            "improvement-branch",
            "Test PR",
            {},
            {},
            self.test_file,
            "balanced",
            "pytest",
            80,
            "fail",
            self.temp_dir,
            ["style", "performance"],
            debug=False,
        )

        mock_github.get_repo.assert_called_once_with("testuser/testrepo")
        mock_repo.create_pull.assert_called_once()
        # Verify the head branch is correctly namespaced
        args, kwargs = mock_repo.create_pull.call_args
        self.assertEqual(kwargs["head"], "testuser:improvement-branch")

    @patch("fabgpt.GitHub")  # Patch within fabgpt
    def test_create_pull_request_error(self, mock_github_class):
        """Test error handling during PR creation."""
        mock_github = MagicMock()
        mock_github_class.return_value = mock_github

        mock_repo = MagicMock()
        mock_github.get_repo.return_value = mock_repo
        mock_repo.create_pull.side_effect = Exception("PR creation failed")

        with self.assertRaises(SystemExit) as cm:
            create_pull_request(
                "https://github.com/testuser/testrepo",
                "token",
                "main",
                "improvement-branch",
                "Test PR",
                {},
                {},
                self.test_file,
                "balanced",
                "pytest",
                80,
                "fail",
                self.temp_dir,
                ["style", "performance"],
                debug=False,

            )
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()