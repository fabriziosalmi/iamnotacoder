# iamnotacoder.py (with extensive improvements)

import git
import os
import tempfile
import subprocess
import click
import time
import difflib
import uuid
import logging
import asyncio
import aiohttp
import shutil
import re
import ast
import json
import datetime
import hashlib
import sys
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.logging import RichHandler
from collections import Counter
from io import StringIO
from rich import box
import toml
import aiofiles  # type: ignore
from openai import OpenAI  # Corrected import: Use the official class, not the module


from github import Github
from typing import List, Dict, Any, Optional, Tuple  # <-- ADDED
# Optionally import rate limit error if available:
# from openai.error import RateLimitError

# Import helper functions from helpers.py
from helpers import (
    load_config,
    get_prompt,
    create_backup,
    restore_backup,
    get_cli_config_priority,
    validate_python_syntax,
    extract_code_from_response,
    format_llm_summary,
)

console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Constants
DEFAULT_LLM_MODEL = "gpt-3.5-turbo-1106"
DEFAULT_LLM_TEMPERATURE = 0.2
MAX_SYNTAX_RETRIES = 5
MAX_LLM_RETRIES = 3
OPENAI_TIMEOUT = 120.0
MAX_PUSH_RETRIES = 3
DEFAULT_LINE_LENGTH = 79
CONFIG_ENCODING = "utf-8"
CACHE_ENCODING = "utf-8"
REPORT_ENCODING = "utf-8"
FORCED_API_BASE = "http://localhost:1234/v1"  # <-- New: Forced endpoint URL constant
FOOTER_MARKDOWN = (
    " \nYou are welcome to improve the project any time by sending back a PR ❤️"
)


class CommandExecutionError(Exception):
    """Custom exception for command execution failures."""

    def __init__(self, command: str, returncode: int, stdout: str, stderr: str):
        super().__init__(
            f"Command `{command}` failed with return code {returncode}.\n"
            f"Stderr: {stderr}\nStdout: {stdout}"
        )
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# Add custom exception for API timeouts
class APITimeoutError(Exception):
    pass


async def run_command_async(
    command: List[str], cwd: Optional[str] = None
) -> Tuple[str, str, int]:
    """Executes a shell command asynchronously and returns stdout, stderr,
    and return code."""
    cmd_str = " ".join(command)
    try:
        start_time = time.time()
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        end_time = time.time()
        stdout_str = stdout.decode()
        stderr_str = stderr.decode()

        if proc.returncode != 0:
            raise CommandExecutionError(
                cmd_str, proc.returncode, stdout_str, stderr_str
            )

        logging.info(
            f"Command `{cmd_str}` executed in {end_time - start_time:.2f} seconds."
        )
        return stdout_str, stderr_str, proc.returncode

    except FileNotFoundError as e:
        logging.error(f"Command not found: {e}")
        return "", str(e), 127

    except CommandExecutionError as e:
        logging.error(str(e))
        return e.stdout, e.stderr, e.returncode

    except Exception as e:
        logging.exception(f"Unhandled error executing command `{cmd_str}`: {e}")
        return "", str(e), 1


# Alias run_command to the async version
run_command = run_command_async


async def clone_repository(repo_url: str, token: str) -> Tuple[git.Repo, str]:
    """Clones a repository (shallow clone) to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    auth_repo_url = repo_url.replace("https://", f"https://{token}@")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Cloning repository (shallow)..."),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Cloning repository...", start=True)
            start_time = time.time()
            repo = await asyncio.to_thread(
                git.Repo.clone_from, auth_repo_url, temp_dir, depth=1
            )
            end_time = time.time()
            progress.update(
                task,
                description=f"Repository cloned in {end_time - start_time:.2f} seconds",
                completed=100,
            )
        return repo, temp_dir
    except git.exc.GitCommandError as e:
        logging.exception(f"Error cloning repository from {repo_url}")
        shutil.rmtree(temp_dir, ignore_errors=True)  # Clean up temp dir
        sys.exit(1)


async def checkout_branch(repo: git.Repo, branch_name: str) -> None:
    """Checks out a specific branch, fetching if necessary."""
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Checking out branch..."),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Checking out branch...", start=True)
            start_time = time.time()
            await asyncio.to_thread(repo.git.fetch, "--all", "--prune")
            await asyncio.to_thread(repo.git.checkout, branch_name)
            end_time = time.time()
            progress.update(
                task,
                description=f"Checked out branch in {end_time - start_time:.2f} seconds",
                completed=100,
            )
    except git.exc.GitCommandError:
        try:
            logging.warning(f"Attempting to fetch remote branch {branch_name}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Checking out remote branch..."),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "Checking out remote branch...", start=True
                )
                start_time = time.time()
                await asyncio.to_thread(repo.git.fetch, "origin", branch_name)
                await asyncio.to_thread(
                    repo.git.checkout, f"origin/{branch_name}"
                )
                end_time = time.time()
                progress.update(
                    task,
                    description=f"Checked out remote branch in {end_time - start_time:.2f} seconds",
                    completed=100,
                )
        except git.exc.GitCommandError as e:
            logging.exception(f"Error checking out branch {branch_name}")
            sys.exit(1)


def create_branch(repo: git.Repo, files: List[str], file_purpose: str = "") -> str:
    """Creates a new, uniquely-named branch for the given files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sanitized_file_names = "_".join(
        "".join(c if c.isalnum() else "_" for c in file) for file in files
    )
    unique_id = uuid.uuid4().hex[:8]  # Shorten UUID for branch name
    branch_name = (
        f"improvement-{sanitized_file_names}-{file_purpose}-{timestamp}-{unique_id}"
    )
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Creating branch..."),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("Creating branch...", start=True)
            start_time = time.time()
            repo.git.checkout("-b", branch_name)
            end_time = time.time()
            progress.update(
                task,
                description=f"Created branch in {end_time - start_time:.2f} seconds",
                completed=100,
            )
        return branch_name
    except git.exc.GitCommandError as e:
        logging.exception(f"Error creating branch {branch_name}")
        sys.exit(1)


def infer_file_purpose(file_path: str) -> str:
    """Infers file's purpose (function, class, or script)."""
    try:
        with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
            first_line = f.readline()
            if "def " in first_line:
                return "function"
            elif "class " in first_line:
                return "class"
            return "script"
    except Exception:
        logging.exception(f"Error inferring purpose for {file_path}")
        return ""  # Consistent return type


def _create_analysis_table(
    results: Dict[str, Dict[str, Any]], analysis_verbose: bool
) -> Table:
    """Creates a Rich Table for static analysis results.  Helper function."""
    table = Table(title="Static Analysis Summary", box=box.ROUNDED)
    table.add_column("Tool", justify="left", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Errors/Warnings", justify="left")

    for tool, result in results.items():
        returncode = result["returncode"]
        errors = result.get("errors", "").strip()
        output = result.get("output", "").strip()

        if returncode == 0:
            status = "[green]Passed[/green]"
            error_summary = "-"
        else:
            status = "[red]Issues[/red]"
            if tool == "pylint":
                issue_codes = re.findall(r"([A-Z]\d{4})", output)
                error_count = len(issue_codes)
                if analysis_verbose:
                    error_summary = errors
                else:
                    top_codes = ", ".join(
                        code for code, _ in Counter(issue_codes).most_common(3)
                    )
                    error_summary = (
                        f"{error_count} ({top_codes})" if error_count > 0 else "-"
                    )
            elif tool == "flake8":
                error_codes = re.findall(r"([A-Z]\d{3})", output)
                error_count = len(error_codes)
                if analysis_verbose:
                    error_summary = errors
                else:
                    top_codes = ", ".join(
                        code for code, _ in Counter(error_codes).most_common(3)
                    )
                    error_summary = (
                        f"{error_count} ({top_codes})" if error_count > 0 else "-"
                    )
            elif tool == "black":
                if "would reformat" in output:
                    status = "[yellow]Would reformat[/yellow]"
                    error_summary = "1 file"
                else:
                    error_summary = errors
            elif tool == "isort":
                if "ERROR:" in output:
                    status = "[yellow]Would reformat[/yellow]"
                    error_summary = str(output.count("ERROR:"))
                else:
                    error_summary = errors
            elif tool == "mypy":
                error_count = output.count("error:")
                if analysis_verbose:
                    error_summary = errors
                else:
                    error_summary = (
                        str(error_count) if error_count > 0 else "-"
                    )
            else:
                error_summary = errors if analysis_verbose else "-"

        table.add_row(tool, status, error_summary)
    return table


async def analyze_project(
    repo_path: str,
    file_path: str,
    tools: List[str],
    exclude_tools: List[str],
    cache_dir: Optional[str] = None,
    debug: bool = False,
    analysis_verbose: bool = False,
    line_length: int = 79,
) -> Dict[str, Dict[str, Any]]:
    """Runs static analysis tools, caching results. Returns results dict."""
    cache_key_data = f"{file_path}-{','.join(sorted(tools))}-{','.join(sorted(exclude_tools))}-{line_length}".encode(
        CACHE_ENCODING
    )
    cache_key = hashlib.sha256(cache_key_data).hexdigest()
    cache_file = (
        os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else None
    )

    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding=CACHE_ENCODING) as f:
                cached_results = json.load(f)
            logging.info("Using static analysis results from cache.")
            return cached_results
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Error loading cache, re-running analysis: {e}")

    results: Dict[str, Dict[str, Any]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        MofNCompleteColumn(),  # Show "M of N"
        console=console,
        transient=True,
    ) as progress:
        analysis_task = progress.add_task("Analyzing...", total=len(tools))

        for tool in tools:
            progress.update(analysis_task, description=f"Running {tool}...")

            if tool in exclude_tools:
                results[tool] = {
                    "output": "",
                    "errors": "Tool excluded.",
                    "returncode": 0,  # Treat exclusion as success
                }
                progress.update(analysis_task, advance=1)
                continue

            if not shutil.which(tool):
                results[tool] = {
                    "output": "",
                    "errors": "Tool not found.",
                    "returncode": 127,  # Standard code for command not found
                }
                progress.update(analysis_task, advance=1)
                continue

            commands = {
                "pylint": ["pylint", file_path],
                "flake8": ["flake8", file_path],
                "black": [
                    "black",
                    "--check",
                    "--diff",
                    f"--line-length={line_length}",
                    file_path,
                ],
                "isort": ["isort", "--check-only", "--diff", file_path],
                "mypy": ["mypy", file_path],
            }
            if tool in commands:
                command = commands[tool]
                try:
                    stdout, stderr, returncode = await run_command(
                        command, cwd=repo_path
                    )

                    results[tool] = {
                        "output": stdout,
                        "errors": stderr,
                        "returncode": returncode,  # Store the return code
                    }

                except CommandExecutionError as e:  # Catch custom exception
                    results[tool] = {
                        "output": e.stdout,
                        "errors": e.stderr,
                        "returncode": e.returncode,
                    }

            else:
                results[tool] = {
                    "output": "",
                    "errors": "Unknown analysis tool.",
                    "returncode": 1,
                }
            progress.update(analysis_task, advance=1)

    if cache_file:
        try:
            with open(cache_file, "w", encoding=CACHE_ENCODING) as f:
                json.dump(results, f, indent=4)
            logging.info("Static analysis results saved to cache.")
        except Exception as e:
            logging.warning(f"Error saving to cache: {e}")

    return results  # Return the results dictionary


async def get_llm_improvements_summary(
    original_code: str,
    improved_code: str,
    categories: List[str],
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    config: Dict,  # Pass the config
) -> Dict[str, List[str]]:
    """Generates a summary of LLM improvements by category using the LLM."""
    diff_lines = list(
        difflib.unified_diff(
            original_code.splitlines(), improved_code.splitlines(), lineterm=""
        )
    )
    diff_text = "\n".join(diff_lines)

    improvements_summary = {}
    for category in categories:
        prompt = (
            f"Analyze the following code diff and list specific improvements in '{category}' category.\n"
            f"Focus only on direct code improvements without introductory or concluding sentences.\n\n"
            f"```diff\n{diff_text}\n```\n\n"
            f"Improvements in '{category}':\n"
        )

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a coding assistant summarizing code improvements.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=min(llm_temperature, 0.2),
                max_tokens=512,
            )
            summary = response.choices[0].message.content.strip()
            improvements = [
                line.strip() for line in summary.splitlines() if line.strip()
            ]
            improvements = [
                re.sub(r"^[\-\*\+] |\d+\.\s*", "", line)
                for line in improvements
            ]  # Clean list markers
            improvements_summary[category] = improvements

        except Exception as e:
            logging.exception(
                f"Error getting LLM improvements summary for category {category}"
            )
            improvements_summary[category] = ["Error retrieving improvements."]
    return improvements_summary


async def format_code_with_tools(file_path: str, line_length: int) -> None:
    """Formats the code using black and isort, if available."""
    if shutil.which("black"):
        await run_command(
            ["black", f"--line-length={line_length}", file_path],
            cwd=os.path.dirname(file_path),
        )
    if shutil.which("isort"):
        await run_command(["isort", file_path], cwd=os.path.dirname(file_path))


async def _process_category_improvement(category, code_snippet, start_line, end_line, config, custom_prompt_dir, client, llm_model, llm_temperature, debug, progress, task_id):
    """Processes improvement for one category with granular progress updates."""
    prompt = get_prompt(config, category, custom_prompt_dir)
    if "{code}" not in prompt:
        logging.error(f"Prompt for {category} missing {{code}} placeholder")
        progress.update(task_id, advance=0, fields={"status": "Prompt Error"})
        return None, 0
    prompt = prompt.replace("{code}", code_snippet)
    prompt += f"\nMaintain a maximum line length of {DEFAULT_LINE_LENGTH} characters."
    if debug:
        logging.debug(f"LLM prompt for {category}:\n{prompt}")
    retry_count = 0
    for attempt in range(MAX_LLM_RETRIES):
        retry_count = attempt + 1
        progress.update(task_id, description=f"[blue]Improving {category} (attempt {attempt+1}/{MAX_LLM_RETRIES})[/blue]", fields={"status": "In progress..."})
        logging.debug(f"LLM call for category '{category}', attempt {attempt+1}")
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant that improves code quality."},
                    {"role": "user", "content": prompt},
                ],
                temperature=llm_temperature,
                max_tokens=1024,
                timeout=OPENAI_TIMEOUT,
            )
            improved_code_snippet = extract_code_from_response(response.choices[0].message.content)
            if validate_python_syntax(improved_code_snippet):
                progress.update(task_id, fields={"status": "Completed"})
                return improved_code_snippet, retry_count
            else:
                logging.warning(f"Syntax error in LLM response for {category} on attempt {attempt+1}")
        except Exception as e:
            if "Rate limit" in str(e):
                logging.error(f"Rate limit encountered for {category}: {e}")
            else:
                logging.exception(f"LLM error in category {category} on attempt {attempt+1}: {e}")
    progress.update(task_id, fields={"status": "Failed"})
    return None, retry_count


async def apply_llm_improvements(
    file_path: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    categories: List[str],
    config: Dict,
    custom_prompt_dir: str,
    line_length: int,
    progress: Progress,
    improve_task_id: int,
    debug: bool,
    analysis_results: Dict[str, Dict[str, Any]],
) -> Tuple[str, bool, Dict[str, int]]:
    """Applies LLM improvements, handling retries, and tracking per-category attempts."""
    total_success = True
    improvements_by_category = {}
    retry_counts: Dict[str, int] = {category: 0 for category in categories} # Track retries

    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    updated_code = source_code
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
             code_snippet = ast.get_source_segment(source_code, node)
             if code_snippet is None:
                continue  # Skip if source segment not found
             start_line = node.lineno
             end_line = node.end_lineno

             for category in categories:
                relevant_errors = False
                if analysis_results:
                    for tool_results in analysis_results.values():
                        if tool_results["returncode"] != 0:
                            for error_line in tool_results["errors"].splitlines():
                                match = re.search(r":(\d+):", error_line)
                                if match:
                                    error_line_num = int(match.group(1))
                                    if start_line <= error_line_num <= end_line:
                                        relevant_errors = True
                                        break
                        if relevant_errors:
                            break
                if not relevant_errors and category not in ["general", "tests"]:
                    logging.info(f"Skipping LLM call for {category} due to no relevant errors.")
                    progress.update(improve_task_id, advance=1/len(categories), fields={"status": "Skipped"})
                    continue

                improved_snippet, attempts = await _process_category_improvement(
                    category, code_snippet, start_line, end_line, config, custom_prompt_dir, client, llm_model, llm_temperature, debug, progress, improve_task_id
                )
                retry_counts[category] = attempts
                if improved_snippet:
                    original_lines = updated_code.splitlines()
                    updated_lines = original_lines[:start_line - 1] + improved_snippet.splitlines() + original_lines[end_line:]
                    updated_code = "\n".join(updated_lines)
                    improvements_by_category[category] = improved_snippet
                else:
                    total_success = False
                progress.update(improve_task_id, advance=1/len(categories))
    return updated_code, total_success, retry_counts


async def improve_file(
    file_path: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    categories: List[str],
    config: Dict,
    custom_prompt_dir: str,
    analysis_results: Dict[str, Dict[str, Any]],
    debug: bool = False,
    line_length: int = DEFAULT_LINE_LENGTH,
) -> Tuple[str, bool]:
    """Improves file using LLM across categories, with retries."""
    backup_path = create_backup(file_path)
    if not backup_path:
        logging.error("Failed to create backup. Aborting file improvement.")
        return "", False

    await format_code_with_tools(file_path, line_length)

    # Track file-level status (added missing keys)
    file_status = {
        "changed": False,
        "restored": False,
        "llm_success": False,  # Overall LLM success
        "categories_attempted": [],
        "categories_skipped": [],
    }

    try:
        with Progress(
            SpinnerColumn("earth"),  # Keep the spinner
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),  # Keep basic bar for overall file progress
            TimeElapsedColumn(),
            TextColumn("[bold green]{task.fields[status]}"),
            console=console,
            transient=True,
            refresh_per_second=10,  # Adjust refresh rate as needed
        ) as progress:
            improve_task_id = progress.add_task(
                "Improving file...", total=len(categories), status="Starting..."
            )

            improved_code, llm_success, retry_counts = await apply_llm_improvements(
                file_path,
                client,
                llm_model,
                llm_temperature,
                categories,
                config,
                custom_prompt_dir,
                line_length,
                progress,
                improve_task_id,
                debug,
                analysis_results
            )

            file_status["llm_success"] = llm_success

            for category in categories:
                if retry_counts.get(category, 0) > 0:
                    file_status["categories_attempted"].append(category)
                else:
                    file_status["categories_skipped"].append(category)


        if not llm_success:
            restore_backup(file_path, backup_path)
            file_status["restored"] = True
            return "", False


        try:
            with open(file_path, "w", encoding=CONFIG_ENCODING) as f:
                f.write(improved_code)
            # Check if the file was actually changed
            with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
                new_code = f.read()
            with open(backup_path, "r", encoding=CONFIG_ENCODING) as f:
                old_code = f.read()

            if new_code.strip() != old_code.strip():
                file_status["changed"] = True
            else:
                console.print(f"[yellow]No changes detected after LLM processing for {file_path}.[/yellow]")

        except Exception as e:
            logging.exception(f"Error writing improved code to {file_path}")
            restore_backup(file_path, backup_path)
            file_status["restored"] = True
            return "", False  # Empty string on failure

        return improved_code, True

    except Exception as e:
        logging.exception(
            f"Unexpected error during file improvement for {file_path}"
        )
        restore_backup(file_path, backup_path)
        file_status["restored"] = True
        return "", False  # Empty string on failure



async def fix_tests_syntax_error(
    generated_tests: str,
    file_base_name: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
) -> Tuple[str, bool]:
    """Attempts to fix syntax errors in generated tests using LLM."""
    try:
        ast.parse(generated_tests)
        return generated_tests, False  # No errors
    except SyntaxError as e:
        logging.warning(f"Syntax error in test generation: {e}")
        line_number = e.lineno
        error_message = str(e)

        code_lines = generated_tests.splitlines()
        context_start = max(0, line_number - 3)
        context_end = min(len(code_lines), line_number + 2)
        context = "\n".join(code_lines[context_start:context_end])
        highlighted_context = context.replace(
            code_lines[line_number - 1], f"#>>> {code_lines[line_number - 1]}"
        )

        error_message_for_llm = (
            f"Syntax error in generated tests for {file_base_name}.py, line {line_number}: {error_message}.\n"
            f"Fix the following code:\n```python\n{highlighted_context}\n```\n"
            f"Return only the corrected code, no intro/outro text, no markdown fences."
        )
        return error_message_for_llm, True  # Error message and flag


async def generate_tests(
    file_path: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    test_framework: str,
    config: Dict,  # Pass config
    custom_prompt_dir: str,
    debug: bool = False,
    line_length: int = DEFAULT_LINE_LENGTH,
) -> str:
    """Generates tests using LLM with syntax error handling and retry."""
    try:
        with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
            code = f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}, cannot generate tests.")
        return ""  # Return empty string for consistency

    file_base_name = os.path.basename(file_path).split(".")[0]
    prompt = get_prompt(config, "tests", custom_prompt_dir)  # Use get_prompt

    if "{code}" not in prompt or "{file_base_name}" not in prompt:
        logging.error(
            "Test prompt missing {code} or {file_base_name} placeholder."
        )
        return ""

    prompt = (
        prompt.replace("{code}", code).replace("{file_base_name}", file_base_name)
    )
    prompt += f"\nMaintain {line_length} chars max line length.\n"
    prompt += (
        "Return only test code, no intro/outro text, no markdown fences."
    )  # Outside

    if debug:
        logging.debug(f"LLM prompt for test generation:\n{prompt}")

    generated_tests = ""
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant that generates tests.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=llm_temperature,
            max_tokens=2048, # Reduced max_tokens
            timeout=OPENAI_TIMEOUT,
        )
        end_time = time.time()
        logging.info(
            f"LLM test generation request took {end_time - start_time:.2f} seconds."
        )
        generated_tests = extract_code_from_response(
            response.choices[0].message.content
        )

        fixed_tests, has_syntax_errors = await fix_tests_syntax_error(
            generated_tests, file_base_name, client, llm_model, llm_temperature
        )

        syntax_error_attempts = 0
        while has_syntax_errors and syntax_error_attempts < MAX_SYNTAX_RETRIES:
            logging.warning("Attempting to fix syntax errors in generated tests...")
            start_time = time.time()
            error_message = fixed_tests  # Error message contains code and error
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a coding assistant fixing syntax errors in tests.",
                        },
                        {"role": "user", "content": error_message},
                    ],
                    temperature=min(llm_temperature, 0.2),  # Lower temp for fixes
                    max_tokens=2048,  # Reduced max_tokens
                    timeout=OPENAI_TIMEOUT,
                )
                end_time = time.time()
                logging.info(
                    f"LLM test syntax fix attempt {syntax_error_attempts + 1} took {end_time - start_time:.2f} seconds."
                )
                generated_tests = extract_code_from_response(
                    response.choices[0].message.content
                )
                fixed_tests, has_syntax_errors = await fix_tests_syntax_error(
                    generated_tests, file_base_name, client, llm_model, llm_temperature
                )
                syntax_error_attempts += 1
            except APITimeoutError:  # Corrected exception type
                logging.warning(
                    f"Timeout during test syntax correction (attempt {syntax_error_attempts + 1})"
                )
                if syntax_error_attempts == MAX_SYNTAX_RETRIES:
                    logging.error(
                        "Max syntax retries for tests reached. Skipping test generation."
                    )
                    return ""  # Give up on generating tests
                continue

        if has_syntax_errors:
            logging.error(
                "Max syntax retries for tests reached. Skipping test generation."
            )
            return ""  # Give up on generating tests if still errors

    except APITimeoutError: # Corrected exception type
        logging.warning("Timeout during initial LLM test generation call.")
        return ""
    except Exception as e:
        logging.exception(f"Error during LLM test generation call for {file_path}")
        return ""  # Return empty string

    tests_dir = os.path.join(os.path.dirname(file_path), "..", "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_file_name = f"test_{os.path.basename(file_path)}"
    test_file_path = os.path.join(tests_dir, test_file_name)

    # Always overwrite (or create) the test file.
    try:
        with open(test_file_path, "w", encoding=CONFIG_ENCODING) as f:
            f.write(generated_tests)
        logging.info(f"Test file written to: {test_file_path}")
        return generated_tests  # Return the generated tests
    except Exception as e:
        logging.exception(f"Error writing test file: {test_file_path}")
        return ""  # Consistent error handling


async def run_tests(
    repo_path: str,
    original_file_path: str,
    test_framework: str,
    min_coverage: Optional[float],
    coverage_fail_action: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """Runs tests (pytest only currently) and checks/enforces code coverage."""
    test_results: Dict[str, Any] = {}
    tests_dir = os.path.join(repo_path, "tests")

    if not os.path.exists(tests_dir):
        logging.warning(f"Tests directory not found: {tests_dir}. Skipping test run.")
        return {
            "output": "No tests were run.",
            "errors": "",
            "returncode": 0,
            "coverage": None,
        }  # Indicate no tests

    if test_framework == "pytest":
        command = ["pytest", "-v", tests_dir]
        if min_coverage is not None:
            rel_file_dir = os.path.relpath(
                os.path.dirname(original_file_path), repo_path
            )
            command.extend(
                [f"--cov={rel_file_dir}", "--cov-report", "term-missing"]
            )

        if debug:
            logging.debug(f"Test command: {' '.join(command)}")

        stdout, stderr, returncode = await run_command(command, cwd=repo_path)
        test_results = {"output": stdout, "errors": stderr, "returncode": returncode}

        if debug:
            logging.debug(f"Test return code: {test_results['returncode']}")

        if returncode == 0:
            logging.info("All tests passed.")
        elif returncode == 1:
            logging.info("Some tests failed.")
        elif returncode == 5:
            logging.warning("No tests found.")  # pytest return code 5: no tests collected
        else:
            logging.error(f"Error during test execution (code {returncode}).")

        # Coverage check and enforcement
        coverage_percentage = None
        if "TOTAL" in test_results.get("output", ""):
            for line in test_results["output"].splitlines():
                if line.lstrip().startswith("TOTAL"):
                    try:
                        coverage_percentage = float(line.split()[-1].rstrip("%"))
                        test_results["coverage"] = coverage_percentage  # Store coverage
                        if (
                            min_coverage is not None
                            and coverage_percentage < min_coverage
                        ):
                            if coverage_fail_action == "fail":
                                logging.error(
                                    f"Coverage ({coverage_percentage:.2f}%) below minimum ({min_coverage:.2f}%). Failing."
                                )
                                test_results[
                                    "returncode"
                                ] = 1  # Set returncode to 1 to fail
                            else:  # coverage_fail_action == "warn"
                                logging.warning(
                                    f"Coverage ({coverage_percentage:.2f}%) below minimum ({min_coverage:.2f}%)."
                                )
                    except (ValueError, IndexError):
                        logging.warning("Could not parse coverage percentage.")
                        test_results["coverage"] = None
        return test_results

    else:
        logging.warning(f"Unsupported test framework: {test_framework}")
        return {
            "output": "",
            "errors": f"Unsupported framework: {test_framework}",
            "returncode": 1,
            "coverage": None,
        }


def create_info_file(
    file_path: str,
    analysis_results: Dict[str, Dict[str, Any]],
    test_results: Optional[Dict[str, Any]],
    llm_success: bool,
    categories: List[str],
    optimization_level: str,
    output_info: str,
    min_coverage: Optional[float] = None,
    llm_improvements_summary: Dict[str, List[str]] = None,
    analysis_verbose: bool = False,
    file_status: Dict[str, Any] = None,  # Add file_status
) -> None:
    """Generates and saves an info file (plain text) summarizing changes."""

    report_table = _create_analysis_table(analysis_results, analysis_verbose)

    with open(output_info, "w", encoding=REPORT_ENCODING) as f:
        f.write(f"FabGPT Improvement Report for: {file_path}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"LLM Improvement Success: {llm_success}\n")
        f.write(f"LLM Optimization Level: {optimization_level}\n")
        f.write(f"Categories Attempted: {', '.join(categories)}\n\n")

        f.write("Changes Made:\n")

        # Use file_status for more accurate change reporting
        if file_status:
            if file_status["changed"]:
                f.write("* File was modified.\n")
            else:
                f.write("* No changes were made to the file.\n")
            if file_status["restored"]:
                f.write("* File was restored from backup.\n")
            f.write(f"* Categories Attempted: {', '.join(file_status['categories_attempted'])}\n")
            f.write(f"* Categories Skipped: {', '.join(file_status['categories_skipped'])}\n")


        f.write("\nStatic Analysis Results:\n")
        table_buffer = StringIO()
        temp_console = Console(file=table_buffer, width=80, record=True) # Increase width if needed
        temp_console.print(report_table)
        table_text = table_buffer.getvalue()
        f.write(table_text)
        f.write("\n")


        f.write("\nTest Results:\n")
        if test_results is not None:
            test_outcome = (
                "Passed" if test_results["returncode"] == 0 else "Failed"
            )
            f.write(f"  Tests: {test_outcome}\n")
            if "TOTAL" in test_results.get("output", ""):
                for line in test_results["output"].splitlines():
                    if line.lstrip().startswith("TOTAL"):
                        try:
                            coverage_percentage = float(
                                line.split()[-1].rstrip("%")
                            )
                            f.write(
                                f"  Code Coverage: {coverage_percentage:.2f}%\n"
                            )
                            if (
                                min_coverage is not None
                                and coverage_percentage < min_coverage
                            ):
                                f.write(
                                    "  WARNING: Coverage below minimum threshold!\n"
                                )
                        except (ValueError, IndexError):
                            pass
            if test_results["returncode"] != 0:
                f.write(
                    f"  WARNING: Some tests failed!\n  Output:\n{test_results.get('output', '')}\n"
                )
        else:
            f.write("  No tests performed.\n")

        if llm_success:
            f.write("\nLLM Improvements Summary:\n")
            if llm_improvements_summary is None:
                llm_improvements_summary = {}
            for category, improvements in llm_improvements_summary.items():
                f.write(f"\nCategory: {category}\n")
                if improvements and improvements != [
                    "Error retrieving improvements."
                ]:
                    for improvement in improvements:
                        f.write(f"- {improvement}\n")
                else:
                    f.write("- No improvements made.\n")


def create_commit(
    repo: git.Repo,
    file_paths: List[str],
    commit_message: str,
    test_results: Optional[Dict[str, Any]] = None,  # Test results optional
) -> None:
    """Creates a Git commit with the provided message, including test changes."""
    try:
        logging.info("Creating commit...")
        for fp in file_paths:
            full_fp = os.path.join(repo.working_tree_dir, fp)
            if os.path.exists(full_fp):
                repo.git.add(fp)
            else:
                logging.warning(f"Warning: '{fp}' not found. Skipping.")
        if test_results is not None:  # Only add tests dir if tests were run/generated
            tests_dir = os.path.join(repo.working_tree_dir, "tests")
            if os.path.exists(tests_dir):
                repo.git.add(tests_dir)

        commit_custom_file = os.path.join(
            repo.working_tree_dir, "commit_custom.txt"
        )
        if os.path.exists(commit_custom_file):
            with open(commit_custom_file, "r", encoding=CONFIG_ENCODING) as cc:
                custom_content = cc.read().strip()
            if custom_content:
                commit_message = (
                    f"{custom_content}\n\n{commit_message}"  # Prepend custom commit
                )

        repo.index.commit(commit_message)
        logging.info("Commit created successfully.")

    except Exception as e:
        logging.exception(f"Error creating commit with message: {commit_message}")
        sys.exit(1)


def format_commit_and_pr_content(
    file_improvements: Dict[str, str]
) -> Tuple[str, str]:
    """Formats improvements for commit message title & body, and PR body."""
    title = f"Improved: {', '.join(file_improvements.keys())}"  # Concise title

    body = ""
    for filename, formatted_summary in file_improvements.items():
        body += (
            f"## Improvements for {filename}:\n\n{formatted_summary}\n"
        )  # Per-file details

    return title, body


def push_branch_with_retry(
    repo: git.Repo, branch_name: str, force_push: bool = False
) -> None:
    """Pushes the branch to remote with retry logic."""
    for attempt in range(MAX_PUSH_RETRIES):
        try:
            logging.info(
                f"Pushing branch to remote (attempt {attempt + 1}/{MAX_PUSH_RETRIES})..."
            )
            if force_push:
                repo.git.push("--force", "origin", branch_name)
            else:
                repo.git.push("origin", branch_name)
            logging.info(
                f"Branch pushed successfully after {attempt + 1} attempt(s)."
            )
            return  # Success on push
        except git.exc.GitCommandError as e:
            logging.error(
                f"Error pushing branch (attempt {attempt + 1}/{MAX_PUSH_RETRIES}): {e}"
            )
            if attempt < MAX_PUSH_RETRIES - 1:
                time.sleep(2)  # Wait before retry
            else:
                logging.error("Max push retries reached. Push failed.")
                raise  # Re-raise exception after max retries


def create_pull_request_programmatically(
    repo_url: str,
    token: str,
    base_branch: str,
    head_branch: str,
    commit_title: str,
    commit_body: str,
) -> None:
    """Creates a GitHub Pull Request using PyGithub library."""
    try:
        logging.info("Creating Pull Request...")
        github_client = Github(token)
        repo_name = repo_url.replace("https://github.com/", "")
        github_repo = github_client.get_repo(repo_name)

        pull_request = github_repo.create_pull(
            title=commit_title,
            body=commit_body,
            head=head_branch,  # Format: "user:branch_name"
            base=base_branch,
        )
        logging.info(f"Pull Request created: {pull_request.html_url}")

    except Exception as e:
        logging.exception(
            f"Error creating pull request to {repo_url} from {head_branch} to {base_branch}"
        )
        sys.exit(1)


@click.command()
@click.option("--repo", "-r", required=False, help="GitHub repository URL.")
@click.option(
    "--files", "-f", required=True, help="Comma-separated file paths to improve."
)
@click.option("--branch", "-b", required=False, help="Target branch name.")
@click.option(
    "--token", "-t", required=False, help="GitHub Personal Access Token (PAT)."
)
@click.option(
    "--tools",
    "-T",
    default="black,isort,pylint,flake8,mypy",
    help="Static analysis tools (comma-separated).",
)
@click.option(
    "--exclude-tools", "-e", default="", help="Tools to exclude (comma-separated)."
)
@click.option("--llm-model", "-m", default=DEFAULT_LLM_MODEL, help="LLM model to use.")
@click.option(
    "--llm-temperature",
    "-temp",
    type=float,
    default=DEFAULT_LLM_TEMPERATURE,
    help="Temperature for the LLM.",
)
@click.option(
    "--llm-optimization-level", "-l", default="balanced", help="LLM optimization level."
)
@click.option("--llm-custom-prompt", "-p", default=".", help="Path to custom prompt directory.")
@click.option("--test-framework", "-F", default="pytest", help="Test framework.")
@click.option(
    "--min-coverage",
    "-c",
    default=None,
    type=float,
    help="Minimum code coverage threshold.",
)
@click.option(
    "--coverage-fail-action",
    default="fail",
    type=click.Choice(["fail", "warn"]),
    help="Action on insufficient coverage.",
)
@click.option(
    "--commit-message",
    "-cm",
    default=None,
    help="Custom commit message (prepended to default).",
)
@click.option(
    "--no-dynamic-analysis", is_flag=True, help="Disable dynamic analysis (testing)."
)
@click.option(
    "--cache-dir", default=None, help="Directory for caching analysis results."
)
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option(
    "--dry-run", is_flag=True, help="Run without making any changes (no commit/PR)."
)
@click.option(
    "--local-commit",
    is_flag=True,
    help="Only commit locally, skip creating a Pull Request.",
)
@click.option(
    "--fast", is_flag=True, help="Enable fast mode (reduces delays, currently no-op)."
)
@click.option(
    "--openai-api-base",
    default=None,
    help="Base URL for OpenAI API (e.g., for LMStudio).",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    callback=get_cli_config_priority,
    is_eager=True,
    expose_value=False,
    help="Path to TOML config file.",
)
@click.option("--no-output", is_flag=True, help="Disable console output.")
@click.option(
    "--categories",
    "-C",
    default="style,maintenance,security,performance",
    help="Comma-separated list of improvement categories.",
)
@click.option(
    "--force-push",
    is_flag=True,
    help="Force push the branch if it exists on remote.",
)
@click.option(
    "--output-file",
    "-o",
    default=None,
    help="Path to save the modified file (defaults to overwrite).",
)
@click.option(
    "--output-info",
    default="report.txt",
    help="Path to save the TEXT report (defaults to report.txt).",
)
@click.option(
    "--line-length",
    type=int,
    default=DEFAULT_LINE_LENGTH,
    help="Maximum line length for code formatting.",
)
@click.option("--fork-repo", is_flag=True, help="Automatically fork the repository.")
@click.option(
    "--fork-user", default=None, help="Your GitHub username for forking (if different)."
)
@click.option("--verbose", is_flag=True, help="Show detailed static analysis errors")
def main(
    repo: str,
    files: str,
    branch: str,
    token: str,
    tools: str,
    exclude_tools: str,
    llm_model: str,
    llm_temperature: float,
    llm_optimization_level: str,
    llm_custom_prompt: str,
    test_framework: str,
    min_coverage: Optional[float],
    coverage_fail_action: str,
    commit_message: Optional[str],
    no_dynamic_analysis: bool,
    cache_dir: Optional[str],
    debug: bool,
    dry_run: bool,
    local_commit: bool,
    fast: bool,
    openai_api_base: Optional[str],
    no_output: bool,
    categories: str,
    force_push: bool,
    output_file: Optional[str],
    output_info: str,
    line_length: int,
    fork_repo: bool,
    fork_user: Optional[str],
    verbose: bool,
) -> None:
    """Main function to improve code quality using static analysis and LLMs."""

    async def main_async():
        # Suppress verbose logging if not in debug mode.
        if not debug:
            logging.getLogger().setLevel(logging.ERROR)

        if no_output:
            console.print = lambda *args, **kwargs: None  # Suppress console output

        ctx = click.get_current_context()
        config_values = ctx.default_map if ctx.default_map else {}
        api_base = config_values.get(
            "openai_api_base", openai_api_base or os.getenv("OPENAI_API_BASE")
        )
        api_key = config_values.get("openai_api_key", os.getenv("OPENAI_API_KEY"))
        api_key_valid = api_key and api_key.strip() and api_key.lower() != "none"

        # Initialize the OpenAI client *outside* the conditional.
        #  We'll set api_key and api_base later.
        client = OpenAI()

        if api_base:
            client.base_url = api_base.rstrip('/')  # Use base_url, and remove trailing slash
            logging.info(f"Using OpenAI API base: {client.base_url}")
        # No else:  If no api_base, it defaults to the OpenAI default.

        if api_key and api_key.strip() and api_key.lower() != "none":
            client.api_key = api_key
            logging.debug("API Key provided; authentication enabled.")
        else:
            if api_base and ("localhost" in api_base.lower() or "127.0.0.1" in api_base):
                client.api_key = "dummy"  #  Use a dummy key for local endpoints
                logging.info("Localhost API endpoint detected; using dummy API key.")
            else:
                logging.error("No valid OpenAI API key provided for public endpoint. Exiting...")
                sys.exit(1)

        client.timeout = OPENAI_TIMEOUT # Set the timeout

        # Proactively test API endpoint
        endpoint_ok = await test_api_endpoint(api_base, api_key if api_key_valid else None)
        if not endpoint_ok:
            logging.error("API endpoint is not reachable. Exiting...")
            sys.exit(1)


        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        files_list = [f.strip() for f in files.split(",")]
        categories_list = [c.strip() for c in categories.split(",")]
        improved_files_info = {}
        pr_url = None  # Initialize pr_url
        # --- Handle Local Files or GitHub Repo ---
        if repo:  # GitHub repository workflow
            # --- Forking logic ---
            repo_url_to_clone = repo  # Default to original repo URL
            fork_owner = fork_user  # User for forking
            if fork_repo:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]Forking repository..."),
                    TimeElapsedColumn(),
                    transient=True,
                ) as progress:
                    task = progress.add_task("Forking repository...", start=True)
                    start_time = time.time()
                    try:
                        github_client = Github(token)
                        original_repo = github_client.get_repo(
                            repo.replace("https://github.com/", "")
                        )
                        forked_repo = await asyncio.to_thread(
                            original_repo.create_fork
                        )
                        repo_url_to_clone = forked_repo.clone_url
                        if not fork_owner:
                            fork_owner = await asyncio.to_thread(
                                github_client.get_user().login
                            )
                    except Exception as e:
                        logging.error(
                            f"Error forking repository: {e}. Please provide --fork-user."
                        )
                        sys.exit(1)
                    end_time = time.time()
                    progress.update(
                        task,
                        description=f"Forked repository in {end_time - start_time:.2f} seconds",
                        completed=100,
                    )
                    logging.info(f"Forked repository to: {repo_url_to_clone}")
            else:
                repo_url_to_clone = repo
                if not fork_owner:
                    try:
                        github_client = Github(token)
                        fork_owner = await asyncio.to_thread(
                            github_client.get_user().login
                        )

                    except Exception as e:
                        logging.error(
                            f"Error getting GitHub username: {e}. Please provide --fork-user."
                        )
                        sys.exit(1)

            repo_obj, temp_dir = await clone_repository(repo_url_to_clone, token)
            new_branch_name = create_branch(
                repo_obj, files_list, "code_improvements"
            )
            await checkout_branch(repo_obj, branch)  # Checkout target branch first
            await checkout_branch(repo_obj, new_branch_name)
            commit_func = create_commit
            push_func = push_branch_with_retry
            pr_func = create_pull_request_programmatically

        else:  # Local files workflow
            temp_dir = os.getcwd()  # Use current working directory
            repo_obj = None  # No repo object needed
            new_branch_name = None  # No branch needed.
            fork_owner = None  # No forking.
            commit_func = lambda *args: None  # No-op commit
            push_func = lambda *args: None  # No-op push
            pr_func = lambda *args: None  # No-op PR creation

        test_results = None
        llm_improvements_summary = {}  # Initialize llm_improvements_summary
        final_analysis_results = {}  # Initialize
        all_files_summary = [] # To store summary of each file


        for file in files_list:
            file_path = os.path.join(temp_dir, file) if repo else os.path.abspath(file)
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}. Skipping.")
                continue

            original_code = ""  # Capture original code per file
            try:
                with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
                    original_code = f.read()
            except Exception as e:
                logging.error(f"Error reading file {file}: {e}. Skipping.")
                continue

            analysis_results = {}
            test_results: Optional[Dict[str, Any]] = None  # Explicitly type
            tests_generated = False
            file_status = { # Initialize file status
                "changed": False,
                "restored": False,
                "categories_attempted": [],
                "categories_skipped": [],
                "llm_success": False,
            }


            if not no_dynamic_analysis:
                analysis_results = await analyze_project(
                    temp_dir,
                    file_path,
                    tools.split(","),
                    exclude_tools.split(","),
                    cache_dir,
                    debug,
                    analysis_verbose=verbose,
                    line_length=line_length,
                )
                # Display analysis results *after* running, *before* tests
                console.print(_create_analysis_table(analysis_results, verbose))

                logging.info("Test generation phase...")
                generated_tests_code = await generate_tests(  # Generate tests
                    file_path,
                    client,
                    llm_model,
                    llm_temperature,
                    test_framework,
                    config_values,  # Pass config
                    llm_custom_prompt,
                    debug,
                    line_length,
                )
                if generated_tests_code:
                    tests_generated = True
                    test_results = await run_tests(  # Run tests
                        temp_dir,
                        file_path,
                        test_framework,
                        min_coverage,
                        coverage_fail_action,
                        debug,
                    )
                    # Check and enforce coverage before proceeding.
                    if (
                        test_results
                        and test_results["returncode"] == 1
                        and test_results.get("coverage") is not None
                        and min_coverage is not None
                        and test_results["coverage"] < min_coverage
                        and coverage_fail_action == "fail"
                    ):
                        logging.error(
                            f"Coverage check failed for {file}. Skipping LLM improvement."
                        )
                        continue  # Skip to the next file if coverage fails

            logging.info("File improvement phase...")
            improved_code_final, llm_success = await improve_file(  # Improve files
                file_path,
                client,
                llm_model,
                llm_temperature,
                categories_list,
                config_values,  # Pass config
                llm_custom_prompt,
                analysis_results,
                debug,
                line_length,
            )


            if improved_code_final:
                # Re-analyze after LLM improvements
                final_analysis_results = await analyze_project(
                    temp_dir,
                    file_path,
                    tools.split(","),
                    exclude_tools.split(","),
                    cache_dir,
                    debug,
                    analysis_verbose=verbose,
                    line_length=line_length
                )
                console.print(_create_analysis_table(final_analysis_results, verbose))

                if improved_code_final.strip() != original_code.strip():
                    file_status["changed"] = True
                    file_status["llm_success"] = llm_success


            formatted_summary = (
                "No LLM-driven improvements were made."  # Default summary
            )
            if llm_success:
                llm_improvements_summary = await get_llm_improvements_summary(
                    original_code,
                    improved_code_final,
                    categories_list,
                    client,
                    llm_model,
                    llm_temperature,
                    config_values,  # Pass config
                )
                formatted_summary = format_llm_summary(llm_improvements_summary)

            improved_files_info[file] = formatted_summary


            if output_file:  # Save improved file to specified output
                output_file_current = (
                    os.path.abspath(output_file)
                    if not os.path.isdir(output_file)
                    else os.path.join(output_file, os.path.basename(file))
                )
                try:
                    with open(output_file_current, "w", encoding=CONFIG_ENCODING) as f:
                        f.write(improved_code_final)
                    logging.info(
                        f"Improved code for {file} saved to: {output_file_current}"
                    )
                except Exception as e:
                    logging.exception(
                        f"Error saving improved code to {output_file_current}"
                    )
                    sys.exit(1)

            create_info_file(  # Create info file
                file_path,
                final_analysis_results, #Pass final
                test_results,
                llm_success,
                categories_list,
                llm_optimization_level,
                output_info,
                min_coverage,
                llm_improvements_summary,
                analysis_verbose=verbose,
                file_status=file_status,  # Pass file_status
            )

             # --- Per-File Summary (Console) ---
            console.print(f"\n[bold underline]Summary for {file}:[/bold underline]")
            if file_status["changed"]:
                console.print("[green]File was modified.[/green]")
            else:
                console.print("[yellow]File was not modified.[/yellow]")
            if file_status["restored"]:
                console.print("[yellow]File was restored from backup.[/yellow]")

            console.print(f"[bold]Categories Attempted:[/bold] {', '.join(file_status['categories_attempted'])}")
            console.print(f"[bold]Categories Skipped:[/bold] {', '.join(file_status['categories_skipped'])}")
            console.print(f"[bold]LLM Improvement Success:[/bold] {file_status['llm_success']}")

            if test_results:
                test_status_str = "[green]Passed[/green]" if test_results["returncode"] == 0 else "[red]Failed[/red]"
                console.print(f"[bold]Test Status:[/bold] {test_status_str}")
                if test_results.get("coverage") is not None:
                    console.print(f"[bold]Coverage:[/bold] {test_results['coverage']:.2f}%")
            else:
                console.print("[bold]Test Status:[/bold] [yellow]No tests run[/yellow]")

            console.print(f"[bold]LLM Improvements:[/bold]\n{formatted_summary}")
            console.print(f"[bold]Full Report:[/bold] {os.path.abspath(output_info)}")
            console.print("-" * 50)

             # Determine overall file status for summary table
            if file_status["changed"]:
                overall_status = "[green]Improved[/green]"
            elif file_status["restored"]:
                overall_status = "[yellow]Restored[/yellow]"
            else:
                overall_status = "[yellow]No changes[/yellow]"


            test_status_table = "No tests"
            if test_results:
                test_status_table = "Passed" if test_results["returncode"] == 0 else "Failed"
            coverage_table = test_results.get("coverage", "N/A") if test_results else "N/A"

            all_files_summary.append({
                "file": file,
                "status": overall_status,
                "categories_improved": len(file_status["categories_attempted"]),  # Use attempted as a proxy for improved
                "test_status": test_status_table,
                "coverage": coverage_table,
            })


        # --- Final Summary Table (Console) ---
        summary_table = Table(title="Overall Summary", box=box.ROUNDED)
        summary_table.add_column("File", justify="left", style="cyan")
        summary_table.add_column("Status", justify="center")
        summary_table.add_column("Categories Improved", justify="center")
        summary_table.add_column("Test Status", justify="center")
        summary_table.add_column("Coverage (%)", justify="right")

        for file_summary in all_files_summary:
            summary_table.add_row(
                file_summary["file"],
                file_summary["status"],
                str(file_summary["categories_improved"]),
                file_summary["test_status"],
                str(file_summary["coverage"]),
            )
        console.print(summary_table)

        # If no improvements, don't commit/PR.
        if not improved_files_info:
            logging.warning(
                "No file improvements detected. Skipping commit and pull request creation."
            )
            if not repo:
                sys.exit(0)  # exit for local workflow

        if not dry_run:
            commit_title, commit_body = format_commit_and_pr_content(
                improved_files_info
            )
            full_commit_message = (
                f"{commit_title}\n\n{commit_body}"  # Combine title and body
            )

            if repo:  # Only perform Git operations if it's a repo
                commit_func(
                    repo_obj, files_list, full_commit_message, test_results
                )  # Create commit

                if not local_commit:
                    try:
                        push_func(
                            repo_obj, new_branch_name, force_push
                        )  # Push branch
                    except Exception:
                        logging.error("Push failed, skipping Pull Request creation.")
                        sys.exit(1)

                    pr_func(  # Create PR
                        repo,
                        token,
                        branch,
                        f"{fork_owner}:{new_branch_name}",
                        commit_title,
                        commit_body,
                    )
            else:  # for local files
                if commit_message:
                    logging.info(f"Commit message:\n{commit_message}")
                else:
                    logging.info(f"Commit message:\n{full_commit_message}")

        log_data = {  # Create log data
            "repository": repo,
            "branch": branch,
            "files_improved": improved_files_info,
            "commit_message": full_commit_message,
            "pr_url": pr_url,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        log_dir = os.path.join(
            "/Users/fab/GitHub/FabGPT", "logs"  # Hardcoded path
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, f"log_{int(time.time())}.json")
        with open(log_file_path, "w", encoding=CONFIG_ENCODING) as log_file:
            json.dump(log_data, log_file, indent=4)

        if repo and not debug:  # Only clean up temp dir for repo operations
            shutil.rmtree(temp_dir)  # Clean up temp dir

        logging.info("All operations completed successfully.")
        sys.exit(0)

    asyncio.run(main_async())


async def test_api_endpoint(api_base: str, api_key: Optional[str]) -> bool:
    import aiohttp
    url = f"{api_base.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as response:
                return response.status == 200
    except Exception as e:
        logging.error(f"API endpoint test failed: {e}")
        return False


if __name__ == "__main__":
    main()