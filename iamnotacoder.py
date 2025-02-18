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
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
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
CONFIG_ENCODING = "utf-8"
CACHE_ENCODING = "utf-8"
REPORT_ENCODING = "utf-8"

class CommandExecutionError(Exception):
    """Custom exception for command execution failures."""


def run_command(command: List[str], cwd: Optional[str] = None) -> Tuple[str, str, int]:
    """Executes a shell command and returns stdout, stderr, and return code."""
    cmd_str = " ".join(command)
    try:
        start_time = time.time()
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=cwd, check=True
        )
        end_time = time.time()
        console.print(
            f"[cyan]Command `{cmd_str}` executed in"
            f" {end_time - start_time:.2f} seconds.[/cyan]"
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        logging.error(f"CalledProcessError for command `{cmd_str}`: %s", e)
        return e.stdout, e.stderr, e.returncode
    except FileNotFoundError as e:
        console.print(f"[red]Command not found: {e}[/red]")
        logging.error(f"FileNotFoundError for command `{cmd_str}`: %s", e)
        return "", str(e), 1
    except Exception as e:
        console.print(f"[red]Unhandled error executing command `{cmd_str}`: {e}[/red]")
        logging.exception(f"Unhandled exception in run_command for `{cmd_str}`")
        return "", str(e), 1


def load_config(config_file: str) -> Dict:
    """Loads configuration from a TOML file. Exits on failure."""
    try:
        with open(config_file, "r", encoding=CONFIG_ENCODING) as f:
            return toml.load(f)
    except FileNotFoundError:
        console.print(f"[red]Configuration file not found: {config_file}[/red]")
        logging.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        console.print(f"[red]Error decoding TOML configuration file: {e}[/red]")
        logging.error(f"TOML decode error in {config_file}: %s", e)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error loading configuration file: {e}[/red]")
        logging.exception(f"Failed to load config file: {config_file}")
        sys.exit(1)


def create_backup(file_path: str) -> Optional[str]:
    """Creates a timestamped backup of a file. Returns backup path or None on failure."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        console.print(f"[green]Backup created: {backup_path}[/green]")
        return backup_path
    except Exception as e:
        console.print(f"[red]Error creating backup for {file_path}: {e}[/red]")
        logging.exception(f"Backup creation failure for {file_path}")
        return None


def restore_backup(file_path: str, backup_path: str) -> None:
    """Restores a file from its backup."""
    try:
        shutil.copy2(backup_path, file_path)
        console.print(f"[green]File restored from: {backup_path}[/green]")
    except FileNotFoundError:
        console.print(f"[red]Backup file not found: {backup_path}[/red]")
        logging.error(f"Backup file not found: {backup_path}")
    except Exception as e:
        console.print(f"[red]Error restoring backup for {file_path} from {backup_path}: {e}[/red]")
        logging.exception(f"Restore backup failure for {file_path} from {backup_path}")


def get_cli_config_priority(ctx: click.Context, param: click.Parameter, value: Any) -> Dict:
    """
    Gets input values, prioritizing command-line arguments over config file.
    Updates the context's default map for subsequent calls.
    """
    config = ctx.default_map or {}
    if value:
        config.update(load_config(value))
    config.update({k: v for k, v in ctx.params.items() if v is not None}) # CLI args override config
    ctx.default_map = config
    return config


def clone_repository(repo_url: str, token: str) -> Tuple[git.Repo, str]:
    """Clones a repository (shallow clone) to a temporary directory. Exits on failure."""
    temp_dir = tempfile.mkdtemp()
    auth_repo_url = repo_url.replace("https://", f"https://{token}@")
    try:
        console.print(f"[blue]Cloning repository (shallow): {repo_url}[/blue]")
        start_time = time.time()
        repo = git.Repo.clone_from(auth_repo_url, temp_dir, depth=1)
        end_time = time.time()
        console.print(
            f"[green]Repository cloned to: {temp_dir} in"
            f" {end_time - start_time:.2f} seconds[/green]"
        )
        return repo, temp_dir
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error cloning repository: {e}[/red]")
        logging.exception(f"Error cloning repository from {repo_url}")
        shutil.rmtree(temp_dir, ignore_errors=True) # Clean up temp dir
        sys.exit(1)


def checkout_branch(repo: git.Repo, branch_name: str) -> None:
    """Checks out a specific branch, fetching if necessary. Exits on failure."""
    try:
        console.print(f"[blue]Checking out branch: {branch_name}[/blue]")
        start_time = time.time()
        repo.git.fetch("--all", "--prune")
        repo.git.checkout(branch_name)
        end_time = time.time()
        console.print(
            f"[green]Checked out branch: {branch_name} in"
            f" {end_time - start_time:.2f} seconds[/green]"
        )
    except git.exc.GitCommandError:
        try:
            console.print(f"[yellow]Attempting to fetch remote branch {branch_name}[/yellow]")
            start_time = time.time()
            repo.git.fetch("origin", branch_name)
            repo.git.checkout(f"origin/{branch_name}")
            end_time = time.time()
            console.print(
                f"[green]Checked out remote branch: {branch_name} in"
                f" {end_time - start_time:.2f} seconds[/green]"
            )
        except git.exc.GitCommandError as e:
            console.print(f"[red]Error checking out branch: {e}[/red]")
            logging.exception(f"Error checking out branch {branch_name}")
            sys.exit(1)


def create_branch(repo: git.Repo, files: List[str], file_purpose: str = "") -> str:
    """Creates a new, uniquely-named branch for the given files. Exits on failure."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sanitized_file_names = "_".join("".join(c if c.isalnum() else "_" for c in file) for file in files)
    unique_id = uuid.uuid4().hex[:8] # Shorten UUID for branch name
    branch_name = f"improvement-{sanitized_file_names}-{file_purpose}-{timestamp}-{unique_id}"
    try:
        console.print(f"[blue]Creating branch: {branch_name}[/blue]")
        start_time = time.time()
        repo.git.checkout("-b", branch_name)
        end_time = time.time()
        console.print(
            f"[green]Created branch: {branch_name} in"
            f" {end_time - start_time:.2f} seconds[/green]"
        )
        return branch_name
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error creating branch: {e}[/red]")
        logging.exception(f"Error creating branch {branch_name}")
        sys.exit(1)


def infer_file_purpose(file_path: str) -> str:
    """Infers the file's purpose (function, class, or script) based on the first line."""
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
        return ""


def analyze_project(
    repo_path: str,
    file_path: str,
    tools: List[str],
    exclude_tools: List[str],
    cache_dir: Optional[str] = None,
    debug: bool = False,
    line_length: int = DEFAULT_LINE_LENGTH,
) -> Dict[str, Dict[str, Any]]:
    """Runs static analysis tools, caching results to improve performance."""
    cache_key_data = (
        f"{file_path}-{','.join(sorted(tools))}-{','.join(sorted(exclude_tools))}-{line_length}".encode(CACHE_ENCODING)
    )
    cache_key = hashlib.sha256(cache_key_data).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else None

    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding=CACHE_ENCODING) as f:
                cached_results = json.load(f)
            console.print("[blue]Using static analysis results from cache.[/blue]")
            return cached_results
        except json.JSONDecodeError:
            console.print("[yellow]Error loading cache, re-running analysis.[/yellow]")
            logging.warning(f"JSON decode error for cache file: {cache_file}")
        except Exception as e:
            console.print("[yellow]Error loading cache, re-running analysis.[/yellow]")
            logging.exception(f"Error loading cache file: {cache_file}")

    results = {}
    console.print("[blue]Running static analysis...[/blue]")

    with Progress(
        SpinnerColumn("dots2"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[bold blue]{task.fields[tool]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        analysis_task = progress.add_task("Running analysis...", total=len(tools), tool="")
        for tool in tools:
            progress.update(analysis_task, tool=f"Running {tool}")
            if tool in exclude_tools:
                progress.update(analysis_task, advance=1, tool=f"Excluded {tool}")
                continue

            if not shutil.which(tool):
                console.print(f"[yellow]Tool not found: {tool}. Skipping.[/yellow]")
                progress.update(analysis_task, advance=1, tool=f"Not found {tool}")
                continue

            commands = {
                "pylint": ["pylint", file_path],
                "flake8": ["flake8", file_path],
                "black": ["black", "--check", "--diff", f"--line-length={line_length}", file_path],
                "isort": ["isort", "--check-only", "--diff", file_path],
                "mypy": ["mypy", file_path],
            }

            if tool in commands:
                command = commands[tool]
                stdout, stderr, returncode = run_command(command, cwd=repo_path)
                results[tool] = {"output": stdout, "errors": stderr, "returncode": returncode}
                status = "Completed" if returncode == 0 else "Errors"
            else:
                console.print(f"[yellow]Unknown analysis tool: {tool}[/yellow]")
                status = "Unknown"

            progress.update(analysis_task, advance=1, tool=status)

    if cache_file:
        try:
            with open(cache_file, "w", encoding=CACHE_ENCODING) as f:
                json.dump(results, f)
            console.print("[blue]Static analysis results saved to cache.[/blue]")
        except Exception as e:
            console.print(f"[yellow]Error saving to cache.[/yellow]")
            logging.exception(f"Error saving analysis results to cache file: {cache_file}")
    return results


def extract_code_from_response(response_text: str) -> str:
    """Extracts code from LLM responses, handling Markdown code blocks and inline code."""
    code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    lines = response_text.strip().splitlines()
    cleaned_lines = []
    start_collecting = False
    for line in lines:
        line = line.strip()
        if not start_collecting:
            if line.startswith(("import ", "def ", "class ")) or re.match(r"^[a-zA-Z0-9_]+(\(.*\)| =.*):", line):
                start_collecting = True # Heuristic to start when code-like lines appear
        if start_collecting:
            if line.lower().startswith("return only the"): # Stop at common LLM instructions
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
    diff_lines = list(difflib.unified_diff(original_code.splitlines(), improved_code.splitlines(), lineterm=""))
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
                    {"role": "system", "content": "You are a coding assistant summarizing code improvements."},
                    {"role": "user", "content": prompt},
                ],
                temperature=min(llm_temperature, 0.2),
                max_tokens=512,
            )
            summary = response.choices[0].message.content.strip()
            improvements = [line.strip() for line in summary.splitlines() if line.strip()]
            improvements = [re.sub(r"^[\-\*\+] |\d+\.\s*", "", line) for line in improvements] # Clean list markers
            improvements_summary[category] = improvements

        except Exception as e:
            console.print(f"[red]Error summarizing {category} improvements: {e}[/red]")
            logging.exception(f"Error getting LLM improvements summary for category {category}")
            improvements_summary[category] = ["Error retrieving improvements."]
    return improvements_summary


def format_code_with_tools(file_path: str, line_length: int) -> None:
    """Formats the code using black and isort, if available."""
    if shutil.which("black"):
        run_command(["black", f"--line-length={line_length}", file_path])
    if shutil.which("isort"):
        run_command(["isort", file_path])


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
    current_code: str, # Pass in current code instead of reading from file repeatedly
    line_length: int,
    progress: Progress,
    improve_task_id: int,
    debug: bool
) -> Tuple[str, bool]:
    """Applies LLM improvements for each category, handling retries and syntax validation."""
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
            console.print(f"[red]Prompt file not found: {prompt_file}. Skipping category {category}.[/red]")
            progress.update(improve_task_id, advance=1, fields={"status": "Prompt not found"})
            continue

        try:
            with open(prompt_file, "r", encoding=CONFIG_ENCODING) as f:
                prompt_template = f.read()
                prompt = prompt_template.replace("{code}", current_code)
                prompt += f"\nMaintain a maximum line length of {line_length} characters."

            success = False
            for attempt in range(MAX_LLM_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful coding assistant that improves code quality."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=llm_temperature,
                        max_tokens=4096,
                        timeout=OPENAI_TIMEOUT
                    )
                    improved_code = extract_code_from_response(response.choices[0].message.content)

                    if validate_python_syntax(improved_code):
                        improvements_by_category[category] = improved_code
                        current_code = improved_code # Update for next category
                        success = True
                        break # Exit retry loop on success
                    else:
                        console.print(f"[yellow]Syntax error in LLM response (attempt {attempt+1}/{MAX_LLM_RETRIES}). Retrying...[/yellow]")

                except Timeout:
                    console.print(f"[yellow]Timeout during LLM call (attempt {attempt+1}/{MAX_LLM_RETRIES}). Retrying...[/yellow]")
                    logging.warning(f"Timeout during LLM call for category {category}, attempt {attempt+1}/{MAX_LLM_RETRIES}")
                except Exception as e:
                    console.print(f"[red]LLM improvement attempt failed (attempt {attempt+1}/{MAX_LLM_RETRIES}): {e}. Retrying...[/red]")
                    logging.error(f"LLM improvement attempt failed for category {category}, attempt {attempt+1}/{MAX_LLM_RETRIES}: {e}")

            if not success:
                total_success = False
                console.print(f"[red]Failed to improve category: {category} after {MAX_LLM_RETRIES} retries.[/red]")

            if debug and success:
                console.print(f"[debug]Category {category} improvements:")
                diff = difflib.unified_diff(
                    current_code.splitlines(),
                    improved_code.splitlines(),
                    fromfile=f"before_{category}",
                    tofile=f"after_{category}",
                )
                console.print("".join(diff)) # diff is already lines, join to print as string

        except Exception as e:
            console.print(f"[red]Error improving category {category}: {e}[/red]")
            total_success = False
            logging.exception(f"Error during LLM improvement for category {category}")

        progress.update(improve_task_id, advance=1, fields={"status": "Completed"})

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
    """Improves the file using LLM across specified categories, with retries and syntax checks."""
    backup_path = create_backup(file_path)
    if not backup_path:
        console.print("[red]Failed to create backup. Aborting file improvement.[/red]")
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
            transient=True,
            refresh_per_second=10,
        ) as progress:
            improve_task_id = progress.add_task(
                "Improving file...", total=len(categories), status="Starting..."
            )
            improved_code, llm_success = apply_llm_improvements(
                file_path, client, llm_model, llm_temperature, categories,
                custom_prompt_dir, current_code, line_length, progress,
                improve_task_id, debug
            )

        if not llm_success:
            restore_backup(file_path, backup_path)
            return current_code, False

        try:
            with open(file_path, "w", encoding=CONFIG_ENCODING) as f:
                f.write(improved_code)
        except Exception as e:
            console.print(f"[red]Error writing improved code to {file_path}: {e}[/red]")
            restore_backup(file_path, backup_path)
            logging.exception(f"Error writing improved code to {file_path}")
            return current_code, False

        return improved_code, True

    except Exception as e:
        console.print(f"[red]Unexpected error during file improvement: {e}[/red]")
        restore_backup(file_path, backup_path)
        logging.exception(f"Unexpected error during file improvement for {file_path}")
        return "", False


def fix_tests_syntax_error(
    generated_tests: str, file_base_name: str, client: OpenAI, llm_model: str, llm_temperature: float
) -> Tuple[str, bool]:
    """Attempts to fix syntax errors in generated tests using LLM."""
    try:
        ast.parse(generated_tests)
        return generated_tests, False # No errors
    except SyntaxError as e:
        console.print(f"[yellow]Syntax error in test generation: {e}[/yellow]")
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
        return error_message_for_llm, True # Error message and flag indicating errors


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
    """Generates tests using LLM with syntax error handling and retry mechanism."""
    try:
        with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
            code = f.read()
    except FileNotFoundError:
        console.print(f"[red]File not found: {file_path}. Cannot generate tests.[/red]")
        logging.error(f"File not found: {file_path}, cannot generate tests.")
        return ""

    file_base_name = os.path.basename(file_path).split(".")[0]
    prompt_file = os.path.join(custom_prompt_dir, "prompt_tests.txt")
    if not os.path.exists(prompt_file):
        console.print(f"[red]Test prompt file not found: {prompt_file}.[/red]")
        return ""

    try:
        with open(prompt_file, "r", encoding=CONFIG_ENCODING) as f:
            prompt_template = f.read()
            prompt = prompt_template.replace("{code}", code).replace("{file_base_name}", file_base_name)
            prompt += f"\nMaintain {line_length} chars max line length.\n"
            prompt += "Return only test code, no intro/outro text, no markdown fences."
    except Exception as e:
        console.print(f"[red]Error reading test prompt file: {e}[/red]")
        logging.exception(f"Error reading test prompt file: {prompt_file}")
        return ""

    if debug:
        console.print(f"[debug]LLM prompt for test generation:\n{prompt}")

    generated_tests = ""
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant that generates tests."},
                {"role": "user", "content": prompt},
            ],
            temperature=llm_temperature,
            max_tokens=4096,
            timeout=OPENAI_TIMEOUT
        )
        end_time = time.time()
        console.print(f"[cyan]LLM test generation request took {end_time - start_time:.2f} seconds.[/cyan]")
        generated_tests = extract_code_from_response(response.choices[0].message.content)

        fixed_tests, has_syntax_errors = fix_tests_syntax_error(
            generated_tests, file_base_name, client, llm_model, llm_temperature
        )

        syntax_error_attempts = 0
        while has_syntax_errors and syntax_error_attempts < MAX_SYNTAX_RETRIES:
            console.print("[yellow]Attempting to fix syntax errors in generated tests...[/yellow]")
            start_time = time.time()
            error_message = fixed_tests # Error message contains the code context and error details.
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a coding assistant fixing syntax errors in tests."},
                        {"role": "user", "content": error_message},
                    ],
                    temperature=min(llm_temperature, 0.2),
                    max_tokens=4096,
                    timeout=OPENAI_TIMEOUT
                )
                end_time = time.time()
                console.print(f"[cyan]LLM test syntax fix attempt {syntax_error_attempts + 1} took {end_time - start_time:.2f} seconds.[/cyan]")
                generated_tests = extract_code_from_response(response.choices[0].message.content)
                fixed_tests, has_syntax_errors = fix_tests_syntax_error(
                    generated_tests, file_base_name, client, llm_model, llm_temperature
                )
                syntax_error_attempts += 1
            except Timeout:
                console.print(f"[yellow]Timeout during test syntax correction (attempt {syntax_error_attempts+1}).[/yellow]")
                logging.warning(f"Timeout during test syntax correction (attempt {syntax_error_attempts + 1})")
                if syntax_error_attempts == MAX_SYNTAX_RETRIES:
                    console.print("[red]Max syntax retries for tests reached. Skipping test generation.[/red]")
                    return "" # Give up on generating tests
                continue # Retry

        if has_syntax_errors:
            console.print("[red]Max syntax retries for tests reached. Skipping test generation.[/red]")
            return "" # Give up if still errors after retries

    except Timeout:
        console.print("[yellow]Timeout during initial LLM test generation call.[/yellow]")
        logging.warning("Timeout during initial LLM test generation call.")
        return ""
    except Exception as e:
        console.print(f"[red]Error during LLM test generation call: {e}[/red]")
        logging.exception(f"Error during LLM test generation call for {file_path}")
        return ""

    tests_dir = os.path.join(os.path.dirname(file_path), "..", "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_file_name = f"test_{os.path.basename(file_path)}"
    test_file_path = os.path.join(tests_dir, test_file_name)

    if os.path.exists(test_file_path):
        console.print(f"[yellow]Test file already exists: {test_file_path}. Skipping write.[/yellow]")
        return "" # Do not overwrite existing tests

    try:
        with open(test_file_path, "w", encoding=CONFIG_ENCODING) as f:
            f.write(generated_tests)
        console.print(f"[green]Test file written to: {test_file_path}[/green]")
        if debug:
            print(f"[DEBUG] Test file exists after write: {os.path.exists(test_file_path)}")
        return generated_tests # Return generated tests
    except Exception as e:
        console.print(f"[red]Error writing test file: {e}[/red]")
        logging.exception(f"Error writing test file: {test_file_path}")
        console.print(f"[debug] Generated test code:\n{generated_tests}") # For debug
        return ""


def run_tests(
    repo_path: str,
    original_file_path: str,
    test_framework: str,
    min_coverage: Optional[float],
    coverage_fail_action: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """Runs tests (pytest only currently) and checks code coverage if requested."""
    test_results = {}
    tests_dir = os.path.join(repo_path, "tests")

    if not os.path.exists(tests_dir):
        console.print(f"[yellow]Tests directory not found: {tests_dir}[/yellow]")
        return {"output": "", "errors": "Tests directory not found", "returncode": 5} # Special return code for no tests

    if test_framework == "pytest":
        command = ["pytest", "-v", tests_dir]
        if min_coverage is not None:
            rel_file_dir = os.path.relpath(os.path.dirname(original_file_path), repo_path)
            command.extend([f"--cov={rel_file_dir}", "--cov-report", "term-missing"])

        if debug:
            print(f"[DEBUG] Current working dir in run_tests: {repo_path}")
            print(f"[DEBUG] Test command: {' '.join(command)}")

        stdout, stderr, returncode = run_command(command, cwd=repo_path)
        test_results = {"output": stdout, "errors": stderr, "returncode": returncode}

        if debug:
            console.print(f"[debug]Test return code: {test_results['returncode']}")
            console.print(f"[debug]Test output:\n{test_results['output']}")
            console.print(f"[debug]Test errors:\n{test_results['errors']}")

        if returncode == 0:
            console.print("[green]All tests passed.[/green]")
        elif returncode == 1:
            console.print("[red]Some tests failed.[/red]")
        elif returncode == 5:
            console.print("[yellow]No tests found.[/yellow]") # No tests collected - pytest return code
        else:
            console.print(f"[red]Error during test execution (code {returncode}).[/red]")
            console.print(f"[debug] Pytest output:\n{stdout}") # Robustness
            console.print(f"[debug] Pytest errors:\n{stderr}")
        return test_results
    else:
        console.print(f"[yellow]Unsupported test framework: {test_framework}[/yellow]")
        return {"output": "", "errors": f"Unsupported framework: {test_framework}", "returncode": 1}


def create_info_file(
    file_path: str,
    analysis_results: Dict[str, Dict[str, Any]],
    test_results: Optional[Dict[str, Any]], # Test results can be None if no tests run
    llm_success: bool,
    categories: List[str],
    optimization_level: str,
    output_info: str,
    min_coverage: Optional[float] = None,
) -> None:
    """Generates and saves an info file (plain text) summarizing the changes."""
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
        if test_results is not None: # Only if tests were attempted
            changes_made.append("Generated/updated tests")

        if changes_made:
            for change in changes_made:
                f.write(f"* {change}\n")
        else:
            f.write("No changes made\n")

        f.write("\nStatic Analysis Results:\n")
        if analysis_results:
            for tool, result in analysis_results.items():
                outcome = "OK" if result["returncode"] == 0 else f"Errors/Warnings ({len(result.get('output', '').splitlines())})"
                f.write(f"* {tool}: {outcome}\n")
                if result["errors"]: # Detail errors if any
                    f.write(f"  Errors/Warnings:\n{result['errors']}\n")
        else:
            f.write("  No static analysis performed.\n")

        f.write("\nTest Results:\n")
        if test_results is not None: # Only report if tests were run
            test_outcome = "Passed" if test_results["returncode"] == 0 else "Failed"
            f.write(f"  Tests: {test_outcome}\n")
            if "TOTAL" in test_results.get("output", ""): # Coverage info if pytest with coverage
                for line in test_results["output"].splitlines():
                    if line.lstrip().startswith("TOTAL"):
                        try:
                            coverage_percentage = float(line.split()[-1].rstrip("%"))
                            f.write(f"  Code Coverage: {coverage_percentage:.2f}%\n")
                            if min_coverage is not None and coverage_percentage < min_coverage:
                                f.write("  WARNING: Coverage below minimum threshold!\n")
                        except (ValueError, IndexError):
                            pass # Ignore lines without coverage data
            if test_results["returncode"] != 0:
                f.write(f"  WARNING: Some tests failed!\n  Output:\n{test_results.get('output', '')}\n")
        else:
            f.write("  No tests performed.\n")


def create_commit(
    repo: git.Repo,
    file_paths: List[str],
    commit_message: str,
    test_results: Optional[Dict[str, Any]] = None, # Test results optional
) -> None:
    """Creates a Git commit with the provided message, including test changes if applicable."""
    try:
        console.print("[blue]Creating commit...[/blue]")
        for fp in file_paths:
            full_fp = os.path.join(repo.working_tree_dir, fp)
            if os.path.exists(full_fp):
                repo.git.add(fp)
            else:
                console.print(f"[yellow]Warning: '{fp}' not found. Skipping.[/yellow]")
        if test_results is not None: # Only add tests dir if tests were run/generated
            tests_dir = os.path.join(repo.working_tree_dir, "tests")
            if os.path.exists(tests_dir):
                repo.git.add(tests_dir)

        commit_custom_file = os.path.join(repo.working_tree_dir, "commit_custom.txt")
        if os.path.exists(commit_custom_file):
            with open(commit_custom_file, "r", encoding=CONFIG_ENCODING) as cc:
                custom_content = cc.read().strip()
            if custom_content:
                commit_message = f"{custom_content}\n\n{commit_message}" # Prepend custom commit message

        repo.index.commit(commit_message)
        console.print("[green]Commit created successfully.[/green]")

    except Exception as e:
        console.print(f"[red]Error creating commit: {e}[/red]")
        logging.exception(f"Error creating commit with message: {commit_message}")
        sys.exit(1)


def format_commit_and_pr_content(file_improvements: Dict[str, str]) -> Tuple[str, str]:
    """Formats improvements for commit message title & body, and PR body."""
    title = f"Improved: {', '.join(file_improvements.keys())}" # Concise title

    body = ""
    for filename, formatted_summary in file_improvements.items():
        body += f"## Improvements for {filename}:\n\n{formatted_summary}\n" # Per-file details

    return title, body


def push_branch_with_retry(repo: git.Repo, branch_name: str, force_push: bool = False) -> None:
    """Pushes the branch to remote with retry logic."""
    for attempt in range(MAX_PUSH_RETRIES):
        try:
            console.print(f"[blue]Pushing branch to remote (attempt {attempt + 1}/{MAX_PUSH_RETRIES})...[/blue]")
            if force_push:
                repo.git.push("--force", "origin", branch_name)
            else:
                repo.git.push("origin", branch_name)
            console.print(f"[green]Branch pushed successfully after {attempt + 1} attempt(s).[/green]")
            return # Success on push
        except git.exc.GitCommandError as e:
            console.print(f"[red]Error pushing branch (attempt {attempt + 1}/{MAX_PUSH_RETRIES}): {e}[/red]")
            logging.error(f"Error pushing branch (attempt {attempt + 1}/{MAX_PUSH_RETRIES}): {e}")
            if attempt < MAX_PUSH_RETRIES - 1:
                time.sleep(2) # Wait before retry
            else:
                console.print(f"[red]Max push retries reached. Push failed.[/red]")
                raise # Re-raise exception after max retries


def create_pull_request_programmatically(
    repo_url: str,
    token: str,
    base_branch: str,
    head_branch: str,
    commit_title: str,
    commit_body: str,
    analysis_results: Dict[str, Dict[str, Any]], # Unused parameter, remove if not needed
    test_results: Optional[Dict[str, Any]], # Unused parameter, remove if not needed
    file_paths: List[str], # Unused parameter, remove if not needed
    optimization_level: str, # Unused parameter, remove if not needed
    test_framework: str, # Unused parameter, remove if not needed
    min_coverage: Optional[float], # Unused parameter, remove if not needed
    coverage_fail_action: str, # Unused parameter, remove if not needed
    repo_path: str, # Unused parameter, remove if not needed
    categories: List[str], # Unused parameter, remove if not needed
    debug: bool = False, # Unused parameter, remove if not needed
    force_push: bool = False, # Unused parameter, remove if not needed
) -> None:
    """Creates a GitHub Pull Request using PyGithub library."""
    try:
        console.print("[blue]Creating Pull Request...[/blue]")
        github_client = Github(token)
        repo_name = repo_url.replace("https://github.com/", "")
        github_repo = github_client.get_repo(repo_name)

        pull_request = github_repo.create_pull(
            title=commit_title,
            body=commit_body,
            head=head_branch, # Format: "user:branch_name"
            base=base_branch,
        )
        console.print(f"[green]Pull Request created: {pull_request.html_url}[/green]")

    except Exception as e:
        console.print(f"[red]Error creating Pull Request: {e}[/red]")
        logging.exception(f"Error creating pull request to {repo_url} from {head_branch} to {base_branch}")
        sys.exit(1)


@click.command()
@click.option("--repo", "-r", required=True, help="GitHub repository URL.")
@click.option("--files", "-f", required=True, help="Comma-separated file paths to improve.")
@click.option("--branch", "-b", required=True, help="Target branch name.")
@click.option("--token", "-t", required=True, help="GitHub Personal Access Token (PAT).")
@click.option("--tools", "-T", default="black,isort,pylint,flake8,mypy", help="Static analysis tools (comma-separated).")
@click.option("--exclude-tools", "-e", default="", help="Tools to exclude (comma-separated).")
@click.option("--llm-model", "-m", default=DEFAULT_LLM_MODEL, help="LLM model to use.")
@click.option("--llm-temperature", "-temp", type=float, default=DEFAULT_LLM_TEMPERATURE, help="Temperature for the LLM.")
@click.option("--llm-optimization-level", "-l", default="balanced", help="LLM optimization level.")
@click.option("--llm-custom-prompt", "-p", default=".", help="Path to custom prompt directory.")
@click.option("--test-framework", "-F", default="pytest", help="Test framework.")
@click.option("--min-coverage", "-c", default=None, type=float, help="Minimum code coverage threshold.")
@click.option("--coverage-fail-action", default="fail", type=click.Choice(["fail", "warn"]), help="Action on insufficient coverage.")
@click.option("--commit-message", "-cm", default=None, help="Custom commit message (prepended to default).")
@click.option("--no-dynamic-analysis", is_flag=True, help="Disable dynamic analysis (testing).")
@click.option("--cache-dir", default=None, help="Directory for caching analysis results.")
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Run without making any changes (no commit/PR).")
@click.option("--local-commit", is_flag=True, help="Only commit locally, skip creating a Pull Request.")
@click.option("--fast", is_flag=True, help="Enable fast mode (reduces delays, currently no-op).")
@click.option("--openai-api-base", default=None, help="Base URL for OpenAI API (e.g., for LMStudio).")
@click.option("--config", default=None, type=click.Path(exists=True), callback=get_cli_config_priority, is_eager=True, expose_value=False, help="Path to TOML config file.")
@click.option("--no-output", is_flag=True, help="Disable console output.")
@click.option("--categories", "-C", default="style,maintenance,security,performance", help="Comma-separated list of improvement categories.")
@click.option("--force-push", is_flag=True, help="Force push the branch if it exists on remote.")
@click.option("--output-file", "-o", default=None, help="Path to save the modified file (defaults to overwrite).")
@click.option("--output-info", default="report.txt", help="Path to save the TEXT report (defaults to report.txt).")
@click.option("--line-length", type=int, default=DEFAULT_LINE_LENGTH, help="Maximum line length for code formatting.")
@click.option("--fork-repo", is_flag=True, help="Automatically fork the repository.")
@click.option("--fork-user", default=None, help="Your GitHub username for forking (if different).")
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
) -> None:
    """
    Improves Python files in a GitHub repository, generates tests, and creates a Pull Request using LLM.
    """
    if no_output:
        console.print = lambda *args, **kwargs: None # Suppress console output

    ctx = click.get_current_context()
    config_values = ctx.default_map if ctx.default_map else {}
    api_base = config_values.get("openai_api_base", openai_api_base or os.getenv("OPENAI_API_BASE"))
    api_key = config_values.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

    if debug:
        console.print("[yellow]Debug mode enabled.[/yellow]")
        console.print(f"[yellow]API Base: {api_base}[/yellow]")
        console.print(f"[yellow]API Key from env/config: {api_key is not None}[/yellow]") # Mask API Key in output
        console.print(f"[yellow]Effective Configuration: {config_values}[/yellow]")

    api_key_provided = api_key and api_key.lower() != "none"
    if not api_base and not api_key_provided: # Require API Key or Base URL
        console.print(
            "[red]Error: OpenAI API key or base URL not found.\n"
            "Set OPENAI_API_KEY/OPENAI_API_BASE environment variables, or use --config or --openai-api-base/--openai-api-key.[/red]"
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=OPENAI_TIMEOUT) if api_base or api_key_provided else None

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # --- Forking logic ---
    repo_url_to_clone = repo # Default to original repo URL
    fork_owner = fork_user # User for forking
    if fork_repo:
        if not fork_owner:
            try:
                github_client = Github(token)
                fork_owner = github_client.get_user().login
            except Exception as e:
                console.print(f"[red]Error getting GitHub username: {e}. Please provide --fork-user.[/red]")
                sys.exit(1)
        console.print(f"[blue]Forking repository to user: {fork_owner}[/blue]")

        try:
            github_client = Github(token)
            original_repo = github_client.get_repo(repo.replace("https://github.com/", ""))
            forked_repo = original_repo.create_fork()
            repo_url_to_clone = forked_repo.clone_url # Clone from forked repo URL
            console.print(f"[green]Forked repository to: {repo_url_to_clone}[/green]")
            time.sleep(5) # Wait for fork to be available
        except Exception as e:
            console.print(f"[red]Error forking repository: {e}[/red]")
            sys.exit(1)

    else: # No forking
        repo_url_to_clone = repo
        if not fork_owner:
            try:
                github_client = Github(token)
                fork_owner = github_client.get_user().login
            except Exception as e:
                console.print(f"[red]Error getting GitHub username: {e}. Please provide --fork-user.[/red]")
                sys.exit(1)


    repo_obj, temp_dir = clone_repository(repo_url_to_clone, token)
    files_list = [f.strip() for f in files.split(",")]
    categories_list = [c.strip() for c in categories.split(",")]
    improved_files_info = {}
    pr_url = None

    new_branch_name = create_branch(repo_obj, files_list, "code_improvements")
    checkout_branch(repo_obj, branch) # Checkout target branch first
    checkout_branch(repo_obj, new_branch_name) # Then new improvement branch

    test_results = None  # <-- Initialize test_results here to avoid UnboundLocalError

    # Initialize final_analysis_results to avoid UnboundLocalError if not set in the loop.
    final_analysis_results = {}

    for file in files_list:
        file_path = os.path.join(temp_dir, file)
        original_code = "" # Capture original code per file
        try:
            with open(file_path, "r", encoding=CONFIG_ENCODING) as f:
                original_code = f.read()
        except Exception as e:
            console.print(f"[red]Error reading file {file}: {e}. Skipping.[/red]")
            logging.error(f"Error reading file {file}: {e}. Skipping.")
            continue # Skip to next file if reading fails

        analysis_results = {}
        test_results: Optional[Dict[str, Any]] = None # Explicitly type test_results
        tests_generated = False

        if not no_dynamic_analysis:
            analysis_results = analyze_project(
                temp_dir, file_path, tools.split(","), exclude_tools.split(","), cache_dir, debug, line_length
            )
            console.print("[blue]Test generation phase...[/blue]")
            generated_tests_code = generate_tests(
                file_path, client, llm_model, llm_temperature, test_framework, llm_custom_prompt, debug, line_length
            )
            if generated_tests_code:
                tests_generated = True
                test_results = run_tests(
                    temp_dir, file_path, test_framework, min_coverage, coverage_fail_action, debug
                )

        console.print("[blue]File improvement phase...[/blue]")
        improved_code_final, llm_success = improve_file(
            file_path, client, llm_model, llm_temperature, categories_list,
            llm_custom_prompt, analysis_results, debug, line_length
        )

        if improved_code_final.strip() == original_code.strip(): # Check for actual code changes
            console.print(f"[yellow]No changes detected for {file}. Skipping further processing.[/yellow]")
            continue

        final_analysis_results = analyze_project( # Re-analyze after LLM
            temp_dir, file_path, tools.split(","), exclude_tools.split(","), cache_dir, debug, line_length
        )

        llm_improvements_summary = {}
        formatted_summary = "No LLM-driven improvements were made." # Default summary if no LLM improvements
        if llm_success:
            llm_improvements_summary = get_llm_improvements_summary(
                original_code, improved_code_final, categories_list, client, llm_model, llm_temperature
            )
            formatted_summary = format_llm_summary(llm_improvements_summary)

        improved_files_info[file] = formatted_summary

        if output_file: # Save improved file if output path is specified
            output_file_current = os.path.abspath(output_file) if not os.path.isdir(output_file) else os.path.join(output_file, os.path.basename(file))
            try:
                with open(output_file_current, "w", encoding=CONFIG_ENCODING) as f:
                    f.write(improved_code_final)
                console.print(f"[green]Improved code for {file} saved to: {output_file_current}[/green]")
            except Exception as e:
                console.print(f"[red]Error saving improved code to {output_file_current}: {e}[/red]")
                logging.exception(f"Error saving improved code to {output_file_current}")
                sys.exit(1)

        create_info_file( # Create info file per processed file
            file_path, final_analysis_results, test_results, llm_success,
            categories_list, llm_optimization_level, output_info, min_coverage
        )

    # New: If no file improvements were made, do not create commit or PR.
    if not improved_files_info:
        console.print("[yellow]No file improvements detected. Skipping commit and pull request creation.[/yellow]")
        sys.exit(0)

    if not dry_run:
        commit_title, commit_body = format_commit_and_pr_content(improved_files_info)
        full_commit_message = f"{commit_title}\n\n{commit_body}" # Combine title and body
        create_commit(repo_obj, files_list, full_commit_message, test_results)

        if not local_commit:
            try:
                push_branch_with_retry(repo_obj, new_branch_name, force_push)
            except Exception:
                console.print("[red]Push failed, skipping Pull Request creation.[/red]")
                sys.exit(1)

            create_pull_request_programmatically( # Create PR to original repo
                repo, token, branch, f"{fork_owner}:{new_branch_name}",
                commit_title, commit_body, final_analysis_results, test_results,
                files_list, llm_optimization_level, test_framework, min_coverage,
                coverage_fail_action, temp_dir, categories_list, debug, force_push
            )

    log_data = { # Log operation details
        "repository": repo,
        "branch": branch,
        "files_improved": improved_files_info,
        "commit_message": full_commit_message, # Use combined commit message for log
        "pr_url": pr_url,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    log_dir = os.path.join("/Users/fab/GitHub/FabGPT", "logs") # Log dir path
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"log_{int(time.time())}.json")
    with open(log_file_path, "w", encoding=CONFIG_ENCODING) as log_file:
        json.dump(log_data, log_file, indent=4)

    if not debug:
        shutil.rmtree(temp_dir)

    console.print("[green]All operations completed successfully.[/green]")
    sys.exit(0)


if __name__ == "__main__":
    main()
