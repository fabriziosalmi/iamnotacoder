import git
import os
import tempfile
import subprocess
import toml
import click
from openai import OpenAI, Timeout
import datetime
import shutil
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TimeElapsedColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
)  # Added MofNCompleteColumn
from rich.table import Table
import json
from github import Github
import hashlib
import time
import ast
from typing import List, Dict, Tuple, Any, Optional
import re
import uuid
import logging
import difflib
from rich.logging import RichHandler
import sys
from collections import Counter
from io import StringIO
from rich import box

console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Constants
DEFAULT_LLM_MODEL = "qwen2.5-coder-7b-instruct-mlx"
DEFAULT_LLM_TEMPERATURE = 0.2
MAX_SYNTAX_RETRIES = 5
MAX_LLM_RETRIES = 3
OPENAI_TIMEOUT = 120.0
MAX_PUSH_RETRIES = 3
DEFAULT_LINE_LENGTH = 79
CONFIG_ENCODING = "utf-8"  # Added constant for encoding
CACHE_ENCODING = "utf-8"   # Added constant for encoding
REPORT_ENCODING = "utf-8"  # Added constant for encoding


class CommandExecutionError(Exception):
    """Custom exception for command execution failures."""

    def __init__(self, command: str, returncode: int, stdout: str, stderr: str):
        super().__init__(
            f"Command `{command}` failed with return code {returncode}.\nStderr: {stderr}\nStdout: {stdout}"
        )
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_command(command: List[str], cwd: Optional[str] = None) -> Tuple[str, str, int]:
    """Executes a shell command and returns stdout, stderr, and return code."""
    cmd_str = " ".join(command)
    try:
        start_time = time.time()
        result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
        end_time = time.time()

        if result.returncode != 0:
            raise CommandExecutionError(
                cmd_str, result.returncode, result.stdout, result.stderr
            )

        logging.info(
            f"Command `{cmd_str}` executed in {end_time - start_time:.2f} seconds."
        )
        return result.stdout, result.stderr, result.returncode

    except FileNotFoundError as e:
        logging.error(f"Command not found: {e}")  # Log the error, More specific message
        return "", str(e), 127  # POSIX return code for command not found

    except CommandExecutionError as e:
        logging.error(str(e))  # Log the CommandExecutionError, Log the specific error
        return e.stdout, e.stderr, e.returncode

    except Exception as e:
        logging.exception(
            f"Unhandled error executing command `{cmd_str}`: {e}"
        )  # Use logging.exception,  More general error handling
        return "", str(e), 1


def load_config(config_file: str) -> Dict:
    """Loads configuration from a TOML file. Exits on failure."""
    try:
        with open(config_file, "r", encoding=CONFIG_ENCODING) as f:
            return toml.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        logging.error(f"Error decoding TOML configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Error loading configuration file: {e}")
        sys.exit(1)


def create_backup(file_path: str) -> Optional[str]:
    """Creates a timestamped backup of a file. Returns backup path or None."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        logging.info(f"Backup created: {backup_path}")
        return backup_path
    except Exception as e:
        logging.exception(f"Backup creation failure for {file_path}")
        return None


def restore_backup(file_path: str, backup_path: str) -> None:
    """Restores a file from its backup."""
    try:
        shutil.copy2(backup_path, file_path)
        logging.info(f"File restored from: {backup_path}")
    except FileNotFoundError:
        logging.error(f"Backup file not found: {backup_path}")
    except Exception as e:
        logging.exception(
            f"Restore backup failure for {file_path} from {backup_path}"
        )


def get_cli_config_priority(
    ctx: click.Context, param: click.Parameter, value: Any
) -> Dict:
    """Prioritizes CLI arguments over config file, updates context."""
    config = ctx.default_map or {}
    if value:
        config.update(load_config(value))
    config.update(
        {k: v for k, v in ctx.params.items() if v is not None}
    )  # CLI args override config
    ctx.default_map = config
    return config


def clone_repository(repo_url: str, token: str) -> Tuple[git.Repo, str]:
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
            repo = git.Repo.clone_from(auth_repo_url, temp_dir, depth=1)
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


def checkout_branch(repo: git.Repo, branch_name: str) -> None:
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
            repo.git.fetch("--all", "--prune")
            repo.git.checkout(branch_name)
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
                repo.git.fetch("origin", branch_name)
                repo.git.checkout(f"origin/{branch_name}")
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
    """Infers file's purpose (function, class, or script) from first line."""
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
        return ""  # Consistent return type: always string


def _create_analysis_table(results: Dict[str, Dict[str, Any]], analysis_verbose:bool) -> Table:
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
                    top_codes = ", ".join(code for code, _ in Counter(issue_codes).most_common(3))
                    error_summary = f"{error_count} ({top_codes})" if error_count > 0 else "-"
            elif tool == "flake8":
                error_codes = re.findall(r"([A-Z]\d{3})", output)
                error_count = len(error_codes)
                if analysis_verbose:
                    error_summary = errors
                else:
                    top_codes = ", ".join(code for code, _ in Counter(error_codes).most_common(3))
                    error_summary = f"{error_count} ({top_codes})" if error_count > 0 else "-"
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
                    error_summary = str(error_count) if error_count > 0 else "-"
            else:
                error_summary = errors if analysis_verbose else "-"

        table.add_row(tool, status, error_summary)
    return table


def analyze_project(
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
    cache_key_data = (
        f"{file_path}-{','.join(sorted(tools))}-{','.join(sorted(exclude_tools))}-{line_length}".encode(
            CACHE_ENCODING
        )
    )
    cache_key = hashlib.sha256(cache_key_data).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else None

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
                    "returncode": 0,
                }
                progress.update(analysis_task, advance=1)
                continue

            if not shutil.which(tool):
                results[tool] = {
                    "output": "",
                    "errors": "Tool not found.",
                    "returncode": 127,
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
                    stdout, stderr, returncode = run_command(
                        command, cwd=repo_path
                    )
                    if tool == "black" and returncode != 0:
                        output = stdout + stderr
                        if "would reformat" in output:
                            returncode = 0
                            stdout = (
                                "[black] Reformatting needed but not applied in analysis."
                            )
                    results[tool] = {
                        "output": stdout,
                        "errors": stderr,
                        "returncode": returncode,
                    }

                except CommandExecutionError as e:  # Catch the custom exception
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


def extract_code_from_response(response_text: str) -> str:
    """Extracts code from LLM responses. Uses improved regex."""
    code_blocks = re.findall(
        r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL
    )
    if code_blocks:
        return code_blocks[-1].strip()  # Use the *last* code block

    # Fallback (less reliable, but handles inline code)
    lines = response_text.strip().splitlines()
    cleaned_lines = []
    start_collecting = False
    for line in lines:
        line = line.strip()
        if not start_collecting:
            if line.startswith(("import ", "def ", "class ")) or re.match(
                r"^[a-zA-Z0-9_]+(\(.*\)| =.*):", line
            ):
                start_collecting = True  # Heuristic: start at code-like lines
        if start_collecting:
            if line.lower().startswith(
                "return only the"
            ):  # Stop at common LLM instructions
                break
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def format_llm_summary(improvements_summary: Dict[str, List[str]]) -> str:
    """Formats and deduplicates LLM improvement summaries."""
    unique_improvements = set()
    for improvements in improvements_summary.values():
        if improvements and improvements != ["Error retrieving improvements."]:
            unique_improvements.update(improvements)

    if unique_improvements:
        return "\n".join(f"- {improvement}" for improvement in unique_improvements) + "\n"
    return "No LLM-driven improvements were made.\n"


def get_llm_improvements_summary(
    original_code: str,
    improved_code: str,
    categories: List[str],
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
) -> Dict[str, List[str]]:
    """Generates a summary of LLM improvements by category by querying the LLM."""
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
                re.sub(r"^[\-\*\+] |\d+\.\s*", "", line) for line in improvements
            ]  # Clean list markers
            improvements_summary[category] = improvements

        except Exception as e:
            logging.exception(
                f"Error getting LLM improvements summary for category {category}"
            )
            improvements_summary[category] = ["Error retrieving improvements."]
    return improvements_summary


def format_code_with_tools(file_path: str, line_length: int) -> None:
    """Formats the code using black and isort, if available."""
    if shutil.which("black"):
        run_command(
            ["black", f"--line-length={line_length}", file_path],
            cwd=os.path.dirname(file_path),
        )
    if shutil.which("isort"):
        run_command(["isort", file_path], cwd=os.path.dirname(file_path))


def validate_python_syntax(code: str) -> bool:
    """Validates if the given code string is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def apply_llm_improvements(
    file_path: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    categories: List[str],
    custom_prompt_dir: str,
    current_code: str,  # Pass in current code
    line_length: int,
    progress: Progress,
    improve_task_id: int,
    debug: bool,
) -> Tuple[str, bool]:
    """Applies LLM improvements for each category, handling retries."""
    total_success = True
    improvements_by_category = {}

    for category in categories:
        progress.update(
            improve_task_id,
            description=f"[blue]Improving category: {category}[/blue]",
            fields={"status": "In progress..."},
        )

        prompt_file = os.path.join(custom_prompt_dir, f"prompt_{category}.txt")
        if not os.path.exists(prompt_file):
            logging.error(
                f"Prompt file not found: {prompt_file}. Skipping category {category}."
            )
            progress.update(
                improve_task_id, advance=1, fields={"status": "Prompt not found"}
            )
            continue

        try:
            with open(prompt_file, "r", encoding=CONFIG_ENCODING) as f:
                prompt_template = f.read()
                prompt = prompt_template.replace("{code}", current_code)
                prompt += (
                    f"\nMaintain a maximum line length of {line_length} characters."
                )

            success = False
            for attempt in range(MAX_LLM_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful coding assistant that improves code quality.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=llm_temperature,
                        max_tokens=4096,
                        timeout=OPENAI_TIMEOUT,
                    )
                    improved_code = extract_code_from_response(
                        response.choices[0].message.content
                    )

                    if validate_python_syntax(improved_code):
                        improvements_by_category[category] = improved_code
                        current_code = improved_code  # Update for next category
                        success = True
                        break  # Exit retry loop on success
                    else:
                        logging.warning(
                            f"Syntax error in LLM response (attempt {attempt+1}/{MAX_LLM_RETRIES}). Retrying..."
                        )

                except Timeout:
                    logging.warning(
                        f"Timeout during LLM call for category {category}, attempt {attempt+1}/{MAX_LLM_RETRIES}"
                    )
                except Exception as e:
                    logging.error(
                        f"LLM improvement attempt failed for category {category}, attempt {attempt+1}/{MAX_LLM_RETRIES}: {e}"
                    )

            if not success:
                total_success = False
                logging.error(
                    f"Failed to improve category: {category} after {MAX_LLM_RETRIES} retries."
                )

            if debug and success:
                logging.debug(f"Category {category} improvements:")
                diff = difflib.unified_diff(
                    current_code.splitlines(),
                    improved_code.splitlines(),
                    fromfile=f"before_{category}",
                    tofile=f"after_{category}",
                )
                logging.debug("".join(diff))

        except Exception as e:
            logging.exception(
                f"Error during LLM improvement for category {category}"
            )
            total_success = False

        progress.update(
            improve_task_id, advance=1, fields={"status": "Completed"}
        )

    return current_code, total_success


def improve_file(
    file_path: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    categories: List[str],
    custom_prompt_dir: str,
    analysis_results: Dict[str, Dict[str, Any]],
    debug: bool = False,
    line_length: int = DEFAULT_LINE_LENGTH,
) -> Tuple[str, bool]:
    """Improves file using LLM across categories, with retries and checks."""
    backup_path = create_backup(file_path)
    if not backup_path:
        logging.error("Failed to create backup. Aborting file improvement.")
        return "", False

    format_code_with_tools(file_path, line_length)

    try:
        with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
            current_code = f.read()

        with Progress(
            SpinnerColumn("earth"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            TextColumn("[bold green]{task.fields[status]}"),
            console=console,
            transient=True,  # Consistent transient
            refresh_per_second=10,
        ) as progress:
            improve_task_id = progress.add_task(
                "Improving file...", total=len(categories), status="Starting..."
            )
            improved_code, llm_success = apply_llm_improvements(
                file_path,
                client,
                llm_model,
                llm_temperature,
                categories,
                custom_prompt_dir,
                current_code,
                line_length,
                progress,
                improve_task_id,
                debug,
            )

        if not llm_success:
            restore_backup(file_path, backup_path)
            return current_code, False

        try:
            with open(file_path, "w", encoding=CONFIG_ENCODING) as f:
                f.write(improved_code)
        except Exception as e:
            logging.exception(f"Error writing improved code to {file_path}")
            restore_backup(file_path, backup_path)
            return current_code, False

        return improved_code, True

    except Exception as e:
        logging.exception(
            f"Unexpected error during file improvement for {file_path}"
        )
        restore_backup(file_path, backup_path)
        return "", False


def fix_tests_syntax_error(
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

def generate_tests(
    file_path: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    test_framework: str,
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
    prompt_file = os.path.join(custom_prompt_dir, "prompt_tests.txt")
    if not os.path.exists(prompt_file):
        logging.error(f"Test prompt file not found: {prompt_file}.")
        return ""  # Return empty string for consistency

    try:
        with open(prompt_file, "r", encoding=CONFIG_ENCODING) as f:
            prompt_template = f.read()
            prompt = prompt_template.replace("{code}", code).replace(
                "{file_base_name}", file_base_name
            )
            prompt += f"\nMaintain {line_length} chars max line length.\n"
            prompt += (
                "Return only test code, no intro/outro text, no markdown fences."
            )  # Outside
    except Exception as e:
        logging.exception(f"Error reading test prompt file: {prompt_file}")
        return ""  # Return empty string for consistency

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
            max_tokens=4096,
            timeout=OPENAI_TIMEOUT,
        )
        end_time = time.time()
        logging.info(
            f"LLM test generation request took {end_time - start_time:.2f} seconds."
        )
        generated_tests = extract_code_from_response(
            response.choices[0].message.content
        )

        fixed_tests, has_syntax_errors = fix_tests_syntax_error(
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
                    max_tokens=4096,
                    timeout=OPENAI_TIMEOUT,
                )
                end_time = time.time()
                logging.info(
                    f"LLM test syntax fix attempt {syntax_error_attempts + 1} took {end_time - start_time:.2f} seconds."
                )
                generated_tests = extract_code_from_response(
                    response.choices[0].message.content
                )
                fixed_tests, has_syntax_errors = fix_tests_syntax_error(
                    generated_tests, file_base_name, client, llm_model, llm_temperature
                )
                syntax_error_attempts += 1
            except Timeout:
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

    except Timeout:
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


def run_tests(
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

        stdout, stderr, returncode = run_command(command, cwd=repo_path)
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
) -> None:
    """Generates and saves an info file (plain text) summarizing changes."""

    report_table = _create_analysis_table(analysis_results, analysis_verbose) # Use helper

    with open(output_info, "w", encoding=REPORT_ENCODING) as f:
        f.write(f"FabGPT Improvement Report for: {file_path}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"LLM Improvement Success: {llm_success}\n")
        f.write(f"LLM Optimization Level: {optimization_level}\n")
        f.write(f"Categories Attempted: {', '.join(categories)}\n\n")

        f.write("Changes Made:\n")
        changes_made = []
        if shutil.which("black"):
            changes_made.append("Formatted with Black")
        if shutil.which("isort"):
            changes_made.append("Formatted with isort")
        if llm_success:
            changes_made.append(f"Applied LLM improvements ({optimization_level})")
        if test_results is not None:
            changes_made.append("Generated/updated tests")

        if changes_made:
            for change in changes_made:
                f.write(f"* {change}\n")
        else:
            f.write("No changes made\n")

        f.write("\nStatic Analysis Results:\n")
        # Correctly capture Rich table output:
        table_buffer = StringIO()
        temp_console = Console(file=table_buffer, width=80, record=True)
        temp_console.print(report_table)  # Use the created table
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

        # Add LLM Improvement Summary (if applicable)
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
@click.option("--repo", "-r", required=True, help="GitHub repository URL.")
@click.option(
    "--files", "-f", required=True, help="Comma-separated file paths to improve."
)
@click.option("--branch", "-b", required=True, help="Target branch name.")
@click.option(
    "--token", "-t", required=True, help="GitHub Personal Access Token (PAT)."
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
@click.option(
    "--llm-custom-prompt", "-p", default=".", help="Path to custom prompt directory."
)
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
    verbose: bool
) -> None:
    """Main function to improve code quality using static analysis and LLMs."""

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

    if debug:
        logging.debug("Debug mode enabled.")
        logging.debug(f"API Base: {api_base}")
        logging.debug(
            f"API Key from env/config: {api_key is not None}"
        )  # Mask API Key
        logging.debug(f"Effective Configuration: {config_values}")

    api_key_provided = api_key and api_key.lower() != "none"
    if not api_base and not api_key_provided:  # Require API Key or Base URL
        logging.error(
            "Error: OpenAI API key or base URL not found.\n"
            "Set OPENAI_API_KEY/OPENAI_API_BASE environment variables, or use --config or --openai-api-base/--openai-api-key."
        )
        sys.exit(1)

    client = (
        OpenAI(api_key=api_key, base_url=api_base, timeout=OPENAI_TIMEOUT)
        if api_base or api_key_provided
        else None
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

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
                forked_repo = original_repo.create_fork()
                repo_url_to_clone = forked_repo.clone_url
                if not fork_owner:
                    fork_owner = github_client.get_user().login
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
                fork_owner = github_client.get_user().login
            except Exception as e:
                logging.error(
                    f"Error getting GitHub username: {e}. Please provide --fork-user."
                )
                sys.exit(1)

    repo_obj, temp_dir = clone_repository(repo_url_to_clone, token)
    files_list = [f.strip() for f in files.split(",")]
    categories_list = [c.strip() for c in categories.split(",")]
    improved_files_info = {}
    pr_url = None

    new_branch_name = create_branch(repo_obj, files_list, "code_improvements")
    checkout_branch(repo_obj, branch)  # Checkout target branch first
    checkout_branch(repo_obj, new_branch_name)  # Then new improvement branch

    test_results = None
    llm_improvements_summary = {}  # Initialize llm_improvements_summary here
    final_analysis_results = {} # Initialize

    for file in files_list:
        file_path = os.path.join(temp_dir, file)
        original_code = ""  # Capture original code per file
        try:
            with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
                original_code = f.read()
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}. Skipping.")
            continue  # Skip to next file

        analysis_results = {}
        test_results: Optional[Dict[str, Any]] = None  # Explicitly type
        tests_generated = False

        if not no_dynamic_analysis:
            analysis_results = analyze_project(
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
            generated_tests_code = generate_tests(  # Generate tests
                file_path,
                client,
                llm_model,
                llm_temperature,
                test_framework,
                llm_custom_prompt,
                debug,
                line_length,
            )
            if generated_tests_code:
                tests_generated = True
                test_results = run_tests(  # Run tests
                    temp_dir,
                    file_path,
                    test_framework,
                    min_coverage,
                    coverage_fail_action,
                    debug,
                )
                # Check and enforce coverage *here* before proceeding.
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
        improved_code_final, llm_success = improve_file(  # Improve files
            file_path,
            client,
            llm_model,
            llm_temperature,
            categories_list,
            llm_custom_prompt,
            analysis_results,
            debug,
            line_length,
        )
        if improved_code_final.strip() == original_code.strip():
            logging.info(
                f"No changes detected for {file}. Skipping further processing."
            )
            continue

        final_analysis_results = analyze_project(  # Re-analyze
            temp_dir,
            file_path,
            tools.split(","),
            exclude_tools.split(","),
            cache_dir,
            debug,
            analysis_verbose=verbose,
            line_length=line_length,
        )
        # Display re-analysis results
        console.print(_create_analysis_table(final_analysis_results, verbose))


        formatted_summary = (
            "No LLM-driven improvements were made."  # Default summary
        )
        if llm_success:
            llm_improvements_summary = get_llm_improvements_summary(
                original_code,
                improved_code_final,
                categories_list,
                client,
                llm_model,
                llm_temperature,
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
            final_analysis_results,
            test_results,
            llm_success,
            categories_list,
            llm_optimization_level,
            output_info,
            min_coverage,
            llm_improvements_summary,
            analysis_verbose=verbose
        )

    # If no improvements, don't commit/PR.
    if not improved_files_info:
        logging.warning(
            "No file improvements detected. Skipping commit and pull request creation."
        )
        sys.exit(0)

    if not dry_run:
        commit_title, commit_body = format_commit_and_pr_content(
            improved_files_info
        )
        full_commit_message = (
            f"{commit_title}\n\n{commit_body}"  # Combine title and body
        )
        create_commit(
            repo_obj, files_list, full_commit_message, test_results
        )  # Create commit

        if not local_commit:
            try:
                push_branch_with_retry(
                    repo_obj, new_branch_name, force_push
                )  # Push branch
            except Exception:
                logging.error("Push failed, skipping Pull Request creation.")
                sys.exit(1)

            create_pull_request_programmatically(  # Create PR
                repo,
                token,
                branch,
                f"{fork_owner}:{new_branch_name}",
                commit_title,
                commit_body,
            )

    log_data = {  # Create log data
        "repository": repo,
        "branch": branch,
        "files_improved": improved_files_info,
        "commit_message": full_commit_message,
        "pr_url": pr_url,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    log_dir = os.path.join(
        "/Users/fab/GitHub/FabGPT", "logs"
    )  # Adjust as needed.  Consider making this configurable.
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"log_{int(time.time())}.json")
    with open(log_file_path, "w", encoding=CONFIG_ENCODING) as log_file:
        json.dump(log_data, log_file, indent=4)

    if not debug:
        shutil.rmtree(temp_dir)  # Clean up temp dir

    logging.info("All operations completed successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()