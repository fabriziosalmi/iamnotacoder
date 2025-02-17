import pytest
import os
import tempfile
import shutil
import toml
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open
import git
from rich.console import Console
from rich.progress import Progress
from openai import OpenAI, types  # Import types
import subprocess



# --- Fixtures ---
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)

@pytest.fixture
def mock_console():
    """Mock rich console for testing."""
    return MagicMock(spec=Console)

@pytest.fixture
def mock_progress():
    """Mock rich progress for testing."""
    return MagicMock(spec=Progress)

@pytest.fixture
def mock_repo():
    """Mock git repo for testing."""
    return MagicMock(spec=git.Repo)

@pytest.fixture
def mock_openai():
    """Mock OpenAI client for testing."""
    return MagicMock(spec=OpenAI)



# --- Imports from your module ---
from iamnotacoder import (
    run_command,
    load_config,
    create_backup,
    restore_backup,
    clone_repository,
    checkout_branch,
    create_branch,
    infer_file_purpose,
    analyze_project,
    extract_code_from_response,
    format_llm_summary,
    validate_python_syntax,
    push_branch_with_retry,
    fix_tests_syntax_error,
    format_commit_and_pr_content,
    get_cli_config_priority
)

# --- Test Cases ---

@pytest.mark.parametrize("command, expected_code, expected_stdout, expected_stderr_contains", [
    (["echo", "test"], 0, "test", ""),
    (["nonexistentcommand"], 1, "", "no such file"),  # Corrected expected code and stderr
])
def test_run_command(command, expected_code, expected_stdout, expected_stderr_contains):
    """Test command execution with various scenarios."""
    stdout, stderr, code = run_command(command)
    assert code == expected_code
    assert expected_stdout in stdout
    if expected_stderr_contains:
        assert expected_stderr_contains in stderr.lower() or "not found" in stderr.lower()

@pytest.mark.parametrize("config_data, expected_key, expected_value", [
    ('[test]\nkey = "value"', "test.key", "value"), # Corrected key
    ('[section1]\nkey1 = "val1"\n[section2]\nkey2 = "val2"', "section1.key1", "val1") # Corrected key
])
def test_load_config_success(config_data, expected_key, expected_value):
    """Test successful config loading with different configurations."""
    with patch("builtins.open", mock_open(read_data=config_data)):
        config = load_config("test.toml")
        if '.' in expected_key:
            section, key = expected_key.split('.')
            assert config[section][key] == expected_value
        else:
            assert config[expected_key] == expected_value


def test_load_config_file_not_found():
    """Test config loading with missing file."""
    with pytest.raises((FileNotFoundError, SystemExit)):  # Expect either
        load_config("nonexistent.toml")


def test_create_backup_success(temp_dir):
    """Test successful file backup creation."""
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")

    backup_path = create_backup(test_file)
    assert backup_path is not None
    assert os.path.exists(backup_path)
    with open(backup_path, "r") as f:
        assert f.read() == "test content"

def test_restore_backup_success(temp_dir):
    """Test successful backup restoration."""
    test_file = os.path.join(temp_dir, "test.txt")
    backup_file = os.path.join(temp_dir, "test.txt.bak")

    with open(test_file, "w") as f:
        f.write("original")
    with open(backup_file, "w") as f:
        f.write("backup")

    restore_backup(test_file, backup_file)

    with open(test_file, "r") as f:
        content = f.read()
    assert content == "backup"

def test_clone_repository_success(mock_repo):
    """Test successful repository cloning."""
    with patch('git.Repo.clone_from', return_value=mock_repo), \
         patch('tempfile.mkdtemp', return_value='/tmp/test_repo'):
        repo, temp_dir = clone_repository("https://github.com/test/repo", "token")
        assert repo == mock_repo
        assert temp_dir == '/tmp/test_repo'
        git.Repo.clone_from.assert_called_once() # Check that it has been called at least once

def test_checkout_branch_success(mock_repo):
    """Test successful branch checkout."""
    checkout_branch(mock_repo, "main")
    mock_repo.git.fetch.assert_called_once_with("--all", "--prune")
    mock_repo.git.checkout.assert_called_once_with("main")

def test_create_branch_success(mock_repo):
    """Test successful branch creation."""
    branch_name = create_branch(mock_repo, ["test.py"], "test")
    assert "improvement" in branch_name
    assert "test_py" in branch_name
    mock_repo.git.checkout.assert_called_once()

@pytest.mark.parametrize("file_content, expected_purpose", [
    ("def test(): pass", "function"),
    ("class Test: pass", "class"),
    ("", "script"),  # Corrected expectation
    ("# This is a comment", "script"),  # Corrected expectation
    ("def func1(): pass\ndef func2(): pass", "function"),
    ("class Class1: pass\nclass Class2: pass", "class")
])
def test_infer_file_purpose(file_content, expected_purpose):
    """Test file purpose inference with various file contents."""
    with patch("builtins.open", mock_open(read_data=file_content)):
        assert infer_file_purpose("test.py") == expected_purpose

def test_extract_code_from_response():
    """Test code extraction from LLM response."""
    response = "```python\ndef test():\n    pass\n```"
    code = extract_code_from_response(response)
    assert code == "def test():\n    pass"

@pytest.mark.parametrize("code, expected_result", [
    ("def test():\n    pass", True),
    ("def test() pass", False),
    ("x = 1 +", False),
    ("print('hello)", False)
])
def test_validate_python_syntax(code, expected_result):
    """Test Python syntax validation with correct and incorrect code."""
    assert validate_python_syntax(code) is expected_result


@pytest.mark.parametrize("force_push", [False, True])
def test_push_branch_with_retry(mock_repo, force_push):
    """Test branch pushing with and without force."""
    push_branch_with_retry(mock_repo, "test-branch", force_push=force_push)
    if force_push:
        mock_repo.git.push.assert_called_once_with("--force", "origin", "test-branch")
    else:
        mock_repo.git.push.assert_called_once_with("origin", "test-branch")



def test_fix_tests_syntax_error_no_error(mock_openai):
    correct_code = "def test_func():\n    pass"
    fixed, flag = fix_tests_syntax_error(correct_code, "dummy", mock_openai, "model", 0.2)
    assert fixed == correct_code
    assert flag is False

from unittest.mock import call  # Import 'call'

def test_format_commit_and_pr_content():
    file_improvements = {"foo.py": "improvement details", "bar.py": "other details"}
    title, body = format_commit_and_pr_content(file_improvements)
    assert title == "Improved: foo.py, bar.py"  # Corrected expectation
    assert "## Improvements for foo.py:" in body
    assert "improvement details" in body
    assert "## Improvements for bar.py:" in body
    assert "other details" in body

def test_get_cli_config_priority():
    # Create a dummy click context with parameters
    class DummyContext:
        def __init__(self):
            self.default_map = {}
            self.params = {"key": "value"}
    ctx = DummyContext()
    dummy_param = None
    dummy_value = None
    # Calling function should update context default_map with params
    result = get_cli_config_priority(ctx, dummy_param, dummy_value)
    assert result.get("key") == "value"

if __name__ == '__main__':
    pytest.main(['-v'])