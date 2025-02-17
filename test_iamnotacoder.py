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
from openai import OpenAI
import subprocess

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
    fix_tests_syntax_error,            # NEW: import fix_tests_syntax_error
    format_commit_and_pr_content,      # NEW: import format_commit_and_pr_content
    get_cli_config_priority            # NEW: import get_cli_config_priority
)

def test_run_command_success():
    """Test successful command execution."""
    cmd = ["echo", "test"]
    stdout, stderr, code = run_command(cmd)
    assert code == 0
    assert "test" in stdout
    assert stderr == ""

def test_run_command_failure():
    """Test command execution failure."""
    cmd = ["nonexistentcommand"]
    stdout, stderr, code = run_command(cmd)
    assert code == 1
    assert stdout == ""
    # Updated assertion to match the error message wording.
    assert "no such file or directory" in stderr.lower()

def test_load_config_success():
    """Test successful config loading."""
    config_data = """
    [test]
    key = "value"
    """
    with patch("builtins.open", mock_open(read_data=config_data)):
        config = load_config("test.toml")
        assert config["test"]["key"] == "value"

def test_load_config_file_not_found():
    """Test config loading with missing file."""
    with pytest.raises(SystemExit):
        load_config("nonexistent.toml")

def test_create_backup_success(temp_dir):
    """Test successful file backup creation."""
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")
    
    backup_path = create_backup(test_file)
    assert backup_path is not None
    assert os.path.exists(backup_path)
        
def test_restore_backup_success(temp_dir):
    """Test successful backup restoration."""
    test_file = os.path.join(temp_dir, "test.txt")
    backup_file = os.path.join(temp_dir, "test.txt.bak")
    
    with open(test_file, "w") as f:
        f.write("original")
    with open(backup_file, "w") as f:
        f.write("backup")
            
    restore_backup(test_file, backup_file)
        
    with open(test_file) as f:
        content = f.read()
    assert content == "backup"

def test_clone_repository_success(mock_repo):
    """Test successful repository cloning."""
    with patch('git.Repo.clone_from', return_value=mock_repo):
        repo, temp_dir = clone_repository("https://github.com/test/repo", "token")
        assert repo == mock_repo
        assert os.path.exists(temp_dir)

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

def test_infer_file_purpose():
    """Test file purpose inference."""
    with patch("builtins.open", mock_open(read_data="def test():")):
        assert infer_file_purpose("test.py") == "function"
    
    with patch("builtins.open", mock_open(read_data="class Test:")):
        assert infer_file_purpose("test.py") == "class"

def test_extract_code_from_response():
    """Test code extraction from LLM response."""
    response = "```python\ndef test():\n    pass\n```"
    code = extract_code_from_response(response)
    assert code == "def test():\n    pass"

def test_validate_python_syntax():
    """Test Python syntax validation."""
    assert validate_python_syntax("def test(): pass") is True
    assert validate_python_syntax("def test() pass") is False

def test_push_branch_with_retry(mock_repo):
    """Test branch pushing with retry logic."""
    push_branch_with_retry(mock_repo, "test-branch")
    mock_repo.git.push.assert_called_once_with("origin", "test-branch")

def test_push_branch_with_retry_force(mock_repo):
    """Test force pushing branch."""
    push_branch_with_retry(mock_repo, "test-branch", force_push=True)
    mock_repo.git.push.assert_called_once_with("--force", "origin", "test-branch")

# NEW: Test fix_tests_syntax_error with no syntax error
def test_fix_tests_syntax_error_no_error(mock_openai):
    correct_code = "def test_func():\n    pass"
    fixed, flag = fix_tests_syntax_error(correct_code, "dummy", mock_openai, "model", 0.2)
    assert fixed == correct_code
    assert flag is False

# NEW: Test fix_tests_syntax_error with a syntax error
def test_fix_tests_syntax_error_with_error(mock_openai):
    incorrect_code = "def test_func("
    fixed, flag = fix_tests_syntax_error(incorrect_code, "dummy", mock_openai, "model", 0.2)
    assert flag is True
    # Check that the returned error message indicates the syntax error context.
    assert "syntax error" in fixed.lower() or "fix" in fixed.lower()

# NEW: Test for format_commit_and_pr_content
def test_format_commit_and_pr_content():
    file_improvements = {"foo.py": "improvement details", "bar.py": "other details"}
    title, body = format_commit_and_pr_content(file_improvements)
    assert "foo.py" in title
    assert "bar.py" in title
    assert "## Improvements for foo.py:" in body
    assert "improvement details" in body

# NEW: Test for get_cli_config_priority
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