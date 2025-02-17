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
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn
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
import sys  # Import sys

console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Constants
DEFAULT_LLM_MODEL: str = "qwen2.5-coder-14b-instruct-mlx"  # Default model
DEFAULT_LLM_TEMPERATURE: float = 0.2
MAX_SYNTAX_RETRIES: int = 5
MAX_LLM_RETRIES: int = 3
OPENAI_TIMEOUT: float = 120.0
MAX_PUSH_RETRIES: int = 3  # Maximum number of times to retry pushing
DEFAULT_LINE_LENGTH: int = 79  # PEP 8 default


def run_command(
    command: List[str], cwd: Optional[str] = None
) -> Tuple[str, str, int]:
    """
    Executes a shell command. Handles errors robustly.
    """
    try:
        start_time = time.time()
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=cwd, check=True
        )
        end_time = time.time()
        console.print(
            f"[cyan]Command `{' '.join(command)}` executed in"
            f" {end_time - start_time:.2f} seconds.[/cyan]"
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        logging.error("CalledProcessError: %s", e)
        return e.stdout, e.stderr, e.returncode
    except FileNotFoundError as e:
        console.print(f"[red]Command not found:[/red] {e}")
        logging.error("FileNotFoundError: %s", e)
        return "", str(e), 1
    except Exception as e:
        console.print(f"[red]Unhandled error during command execution:[/red] {e}")
        logging.exception("Unhandled exception in run_command")
        return "", str(e), 1


def load_config(config_file: str) -> dict:
    """Loads configuration from a TOML file."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return toml.load(f)
    except Exception as e:
        console.print(f"[red]Error loading configuration file:[/red] {e}")
        logging.exception("Failed to load config")
        sys.exit(1)  # Exit on config load failure


def create_backup(file_path: str) -> Optional[str]:
    """Creates a timestamped backup of a file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        console.print(f"[green]Backup created:[/green] {backup_path}")
        return backup_path
    except Exception as e:
        console.print(f"[red]Error creating backup:[/red] {e}")
        logging.exception("Backup creation failure")
        return None


def restore_backup(file_path: str, backup_path: str) -> None:
    """Restores a file from its backup."""
    try:
        shutil.copy2(backup_path, file_path)
        console.print(f"[green]File restored from:[/green] {backup_path}")
    except Exception as e:
        console.print(f"[red]Error restoring backup:[/red] {e}")
        logging.exception("Restore backup failure")


def get_inputs(ctx: click.Context, param: click.Parameter, value: Any) -> dict:
    """Gets input values, prioritizing command-line over config file."""
    config = {}
    if ctx.default_map:
        config.update(ctx.default_map)
    if value:  # Load config file if provided
        config.update(load_config(value))
    # Command-line options override config file
    for k, v in ctx.params.items():
        if v is not None:
            config[k] = v
    ctx.default_map = config  # Update context for subsequent calls
    return config


def clone_repository(repo_url: str, token: str) -> Tuple[git.Repo, str]:
    """Clones a repository (shallow clone) to a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    auth_repo_url = repo_url.replace("https://", f"https://{token}@")
    try:
        console.print(f"[blue]Cloning repository (shallow): {repo_url}[/blue]")
        start_time = time.time()
        repo = git.Repo.clone_from(auth_repo_url, temp_dir, depth=1)
        end_time = time.time()
        console.print(
            f"[green]Repository cloned to:[/green] {temp_dir} in"
            f" {end_time - start_time:.2f} seconds"
        )
        return repo, temp_dir
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error cloning repository:[/red] {e}")
        logging.exception("Error cloning repository")
        sys.exit(1)  # Exit on clone failure


def checkout_branch(repo: git.Repo, branch_name: str) -> None:
    """Checks out a specific branch, fetching if necessary."""
    try:
        console.print(f"[blue]Checking out branch: {branch_name}[/blue]")
        start_time = time.time()
        repo.git.fetch("--all", "--prune")  # Fetch and remove stale branches
        repo.git.checkout(branch_name)
        end_time = time.time()
        console.print(
            f"[green]Checked out branch:[/green] {branch_name} in"
            f" {end_time - start_time:.2f} seconds"
        )
    except git.exc.GitCommandError:
        try:  # Attempt to fetch and checkout remote branch
            console.print(f"[yellow]Attempting to fetch branch {branch_name}[/yellow]")
            start_time = time.time()
            repo.git.fetch("origin", branch_name)
            repo.git.checkout(f"origin/{branch_name}")  # Checkout remote branch
            end_time = time.time()
            console.print(
                f"[green]Checked out branch:[/green] {branch_name} in"
                f" {end_time - start_time:.2f} seconds"
            )
        except git.exc.GitCommandError as e:
            console.print(f"[red]Error checking out branch:[/red] {e}")
            logging.exception("Error checking out branch")
            sys.exit(1)  # Exit on checkout failure


def create_branch(repo: git.Repo, files: List[str], file_purpose: str = "") -> str:
    """Creates a new, uniquely-named branch for multiple files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Create a combined, sanitized file name from all files
    sanitized_file_names = "_".join(
        "".join(c if c.isalnum() else "_" for c in file) for file in files
    )
    unique_id = uuid.uuid4().hex
    branch_name = (
        f"improvement-{sanitized_file_names}-{file_purpose}-{timestamp}-{unique_id}"
    )
    try:
        console.print(f"[blue]Creating branch: {branch_name}[/blue]")
        start_time = time.time()
        repo.git.checkout("-b", branch_name)
        end_time = time.time()
        console.print(
            f"[green]Created branch:[/green] {branch_name} in"
            f" {end_time - start_time:.2f} seconds"
        )
        return branch_name
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error creating branch:[/red] {e}")
        logging.exception("Error creating branch")
        sys.exit(1)  # Exit on branch creation failure


def get_file_purpose(file_path: str) -> str:
    """Infers the file's purpose (function, class, or script)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if "def " in first_line:
                return "function"
            elif "class " in first_line:
                return "class"
            else:
                return "script"
    except Exception:
        return ""  # Return empty string on any error


def analyze_project(
    repo_path: str,
    file_path: str,
    tools: List[str],
    exclude_tools: List[str],
    cache_dir: Optional[str] = None,
    debug: bool = False,
    line_length: int = DEFAULT_LINE_LENGTH,
) -> Dict[str, Dict[str, Any]]:
    """Runs static analysis tools, caching results."""
    cache_key_data = (
        f"{file_path}-{','.join(sorted(tools))}-{','.join(sorted(exclude_tools))}-{line_length}".encode(
            "utf-8"
        )
    )  # Include line_length
    cache_key = hashlib.sha256(cache_key_data).hexdigest()
    cache_file = (
        os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else None
    )

    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_results = json.load(f)
            console.print("[blue]Using static analysis results from cache.[/blue]")
            return cached_results
        except Exception as e:
            console.print(
                "[yellow]Error loading cache, re-running analysis.[/yellow]"
            )
            logging.exception("Error loading cache")

    results = {}
    console.print("[blue]Running static analysis...[/blue]")

    # Replace the existing analysis progress with:
    with Progress(
        SpinnerColumn("dots2"),  # Use a different spinner
        TextColumn("[bold cyan]{task.description}"),
        "•",
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        "•",
        TextColumn("[blue]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("[green]{task.fields[tool]}"),  # Show current tool
        console=console,
        transient=True,
    ) as progress:
        analysis_task = progress.add_task(
            "Running analysis...", total=len(tools), tool=""
        )
        for tool in tools:
            progress.update(analysis_task, tool=f"Running {tool}")
            if tool in exclude_tools:
                progress.update(
                    analysis_task, advance=1, tool=f"Excluded {tool}"
                )
                continue

            if shutil.which(tool) is None:
                console.print(
                    f"[yellow]Tool not found: {tool}. Skipping.[/yellow]"
                )
                progress.update(
                    analysis_task, advance=1, tool=f"Not found {tool}"
                )
                continue

            # Use a dictionary for commands - easier to manage
            commands = {
                "pylint": ["pylint", file_path],
                "flake8": ["flake8", file_path],
                "black": [
                    "black",
                    "--check",
                    "--diff",
                    f"--line-length={line_length}",
                    file_path,
                ],  # Pass line_length
                "isort": ["isort", "--check-only", "--diff", file_path],
                "mypy": ["mypy", file_path],
            }

            if tool in commands:
                command = commands[tool]
                stdout, stderr, returncode = run_command(
                    command, cwd=repo_path
                )
                results[tool] = {
                    "output": stdout,
                    "errors": stderr,
                    "returncode": returncode,
                }
                status = "Completed" if returncode == 0 else "Errors"
            else:
                console.print(f"[yellow]Unknown analysis tool: {tool}[/yellow]")
                status = "Unknown"

            progress.update(analysis_task, advance=1, tool=status)

    if cache_file:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(results, f)
            console.print(f"[blue]Static analysis results saved to cache.[/blue]")
        except Exception as e:
            console.print(f"[yellow]Error saving to cache.[/yellow]")
            logging.exception("Error saving to cache")

    return results


def clean_llm_response(response_text: str) -> str:
    """Extracts code from LLM responses, handling various formats."""
    code_blocks = re.findall(
        r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL
    )
    if code_blocks:
        return code_blocks[-1].strip()  # Use the last code block

    # Fallback: Extract lines starting with common code constructs
    lines = response_text.splitlines()
    cleaned_lines = []
    started = False
    for line in lines:
        line = line.strip()
        if not started:
            # Start capturing lines that look like code
            if (
                line.startswith("import ")
                or line.startswith("def ")
                or line.startswith("class ")
                or (line and not line.startswith("#") and not line[0].isspace())
            ):
                started = True
        if started:
            if line.lower().startswith(
                "return only the"
            ):  # Stop at common LLM phrases
                break
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def format_llm_summary(improvements_summary: Dict[str, List[str]]) -> str:
    """Formats the LLM improvement summary, deduplicating improvements."""
    formatted_summary = ""
    unique_improvements = set()  # Use a set for deduplication

    for category, improvements in improvements_summary.items():
        if improvements and improvements != ["Error retrieving improvements."]:
            for improvement in improvements:
                # Add to the set (automatically handles duplicates)
                unique_improvements.add(improvement)

    # Build the formatted string from the unique improvements
    if unique_improvements:
        for improvement in unique_improvements:
            formatted_summary += f"- {improvement}\n"
    else:
        formatted_summary = "No LLM-driven improvements were made.\n"

    return formatted_summary


def get_llm_improvements_summary(
    original_code: str,
    improved_code: str,
    categories: List[str],
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
) -> Dict[str, List[str]]:
    """Gets a summary of LLM improvements, by category, and formats it."""
    diff = list(
        difflib.unified_diff(
            original_code.splitlines(), improved_code.splitlines(), lineterm=""
        )
    )
    diff_text = "\n".join(diff)

    improvements_summary = {}
    for category in categories:
        prompt = (
            f"Review the following diff of a Python file and identify specific improvements"
            f" made in the '{category}' category. Only list the specific improvements;"
            f" do not include any introductory or concluding text.\n\n"
            f"Diff:\n```diff\n{diff_text}\n```\n"
            f"Improvements in the '{category}' category:"
        )
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful coding assistant that summarizes code"
                            " improvements."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=min(
                    llm_temperature, 0.2
                ),  # Lower temp for concise summaries
                max_tokens=512,  # Limit response length
            )
            summary = response.choices[0].message.content.strip()

            # Split and clean the improvements
            improvements = [
                line.strip() for line in summary.splitlines() if line.strip()
            ]
            improvements = [
                re.sub(r"^[\-\*\+] |\d+\.\s*", "", line) for line in improvements
            ]
            improvements_summary[category] = improvements

        except Exception as e:
            console.print(
                f"[red]Error getting LLM improvements summary for {category}:"
                f" {e}[/red]"
            )
            logging.exception("Error getting LLM improvements summary")
            improvements_summary[category] = ["Error retrieving improvements."]

    return improvements_summary


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
    """Improves the file using LLM, with retries and syntax checking."""
    backup_path = create_backup(file_path)
    if not backup_path:
        console.print("[red]Failed to create backup. Aborting.[/red]")
        sys.exit(1)

    # Format with Black and isort before LLM processing
    if shutil.which("black"):
        run_command(["black", f"--line-length={line_length}", file_path])
    if shutil.which("isort"):
        run_command(["isort", file_path])

    # Read initial formatted code
    with open(file_path, "r", encoding="utf-8") as f:
        current_code = f.read()

    total_success = True
    improvements_by_category = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),  # Make description bold and blue
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        "•",
        TextColumn("[progress.completed]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn(
            "[bold green]{task.fields[status]}"
        ),  # Add status field with color
        console=console,  # Explicitly pass console
        transient=True,
        refresh_per_second=10,  # Increase refresh rate
    ) as progress:
        improve_task = progress.add_task(
            "Improving file...", total=len(categories), status="Starting..."
        )

        for category in categories:
            progress.update(
                improve_task,
                description=f"[blue]Improving category: {category}[/blue]",
                status="In progress...",
            )

            # Load category-specific prompt
            prompt_file = os.path.join(custom_prompt_dir, f"prompt_{category}.txt")
            if not os.path.exists(prompt_file):
                console.print(
                    f"[red]Prompt file not found: {prompt_file}. Skipping category.[/red]"
                )
                progress.update(improve_task, advance=1, status="Prompt not found")
                continue

            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_template = f.read()
                    # Use the current_code that includes previous improvements
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
                                    "content": (
                                        "You are a helpful coding assistant that improves code"
                                        " quality."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            temperature=llm_temperature,
                            max_tokens=4096,  # Adjust as needed
                        )
                        improved_code = clean_llm_response(
                            response.choices[0].message.content
                        )

                        # Validate syntax
                        try:
                            ast.parse(improved_code)
                            # Store improvement for this category
                            improvements_by_category[category] = improved_code
                            # Update current code for next category
                            current_code = improved_code
                            success = True
                            break
                        except SyntaxError:
                            # Handle syntax errors...
                            pass
                    except Exception as e:
                        logging.error("LLM improvement attempt failed: %s", e)

                if not success:
                    total_success = False

                if debug:
                    console.print(f"[debug]Category {category} improvements:")
                    diff = list(
                        difflib.unified_diff(
                            current_code.splitlines(),
                            improved_code.splitlines(),
                            fromfile=f"before_{category}",
                            tofile=f"after_{category}",
                        )
                    )
                    console.print("\n".join(diff))

            except Exception as e:
                console.print(f"[red]Error improving {category}: {e}[/red]")
                total_success = False

            progress.update(improve_task, advance=1, status="Completed")

    # If any improvements failed, restore from backup
    if not total_success:
        restore_backup(file_path, backup_path)
        return current_code, False

    # Write final improved code
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(current_code)
    except Exception as e:
        console.print(f"[red]Error writing improved code: {e}[/red]")
        restore_backup(file_path, backup_path)
        return current_code, False

    return current_code, True


def fix_tests(generated_tests: str, file_base_name: str) -> Tuple[str, bool]:
    """Fixes syntax errors in generated tests, using LLM if needed."""
    try:
        ast.parse(generated_tests)  # Initial syntax check
        return generated_tests, False  # No errors
    except SyntaxError as e:
        console.print(f"[yellow]Syntax error in test generation: {e}[/yellow]")
        line_number = e.lineno
        error_message = str(e)

        # Prepare context for the LLM (as in improve_file)
        code_lines = generated_tests.splitlines()
        start_line = max(0, line_number - 3)
        end_line = min(len(code_lines), line_number + 2)
        context = "\n".join(code_lines[start_line:end_line])
        highlighted_context = context.replace(
            code_lines[line_number - 1], f"#>>> {code_lines[line_number - 1]}"
        )

        # Construct the error message for the LLM
        error_message_with_line = (
            "Syntax error in generated tests for"
            f" {file_base_name}.py on line {line_number}: {error_message}.  Please"
            f" fix the following code:\n```python\n{highlighted_context}\n```\n\n"
            "Return only the corrected code, without any introductory or concluding"
            " text. Do not include markdown code fences."
        )
        return (
            error_message_with_line,
            True,
        )  # Return the error message and the "had_errors" flag


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
    """Generates tests using LLM, with syntax error handling."""
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()

    file_base_name = os.path.basename(file_path).split(".")[0]

    prompt_file = os.path.join(custom_prompt_dir, "prompt_tests.txt")
    if not os.path.exists(prompt_file):
        console.print(
            f"[red]Prompt file for tests not found: {prompt_file}.[/red]"
        )
        return ""  # Return empty string if prompt file is missing

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            prompt = (
                prompt_template.replace("{code}", code).replace(
                    "{file_base_name}", file_base_name
                )
            )
            # Add line length instruction
            prompt += (
                f"\nMaintain a maximum line length of {line_length} characters."
                " Return only the test code, without any introductory or concluding"
                " text.  Do not include markdown code fences (```)."
            )
    except Exception as e:
        console.print(f"[red]Error reading test prompt file: {e}[/red]")
        logging.exception("Error reading test prompt file")
        return ""

    if debug:
        console.print(f"[debug]LLM prompt for test generation:\n{prompt}")

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful coding assistant that generates tests."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=llm_temperature,
            max_tokens=4096,  # Adjust as needed
        )
        end_time = time.time()
        console.print(
            "[cyan]LLM request for test generation took"
            f" {end_time - start_time:.2f} seconds.[/cyan]"
        )
        generated_tests = response.choices[0].message.content.strip()
        generated_tests = clean_llm_response(
            generated_tests
        )  # Clean the response

        fixed_tests, had_errors = fix_tests(
            generated_tests, file_base_name
        )  # Initial fix attempt

        # Retry loop for syntax errors in generated tests
        syntax_errors = 0
        while had_errors and syntax_errors < MAX_SYNTAX_RETRIES:
            console.print(
                "[yellow]Attempting to fix syntax errors in generated tests...[/yellow]"
            )
            start_time = time.time()
            error_message = (
                fixed_tests
            )  # Use the error message from fix_tests
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful coding assistant that generates tests."
                                " Fix the following code that has syntax errors."
                            ),
                        },
                        {"role": "user", "content": error_message},
                    ],
                    temperature=min(
                        llm_temperature, 0.2
                    ),  # Lower temperature
                    max_tokens=4096,
                )
                end_time = time.time()
                console.print(
                    "[cyan]LLM retry for test generation (attempt"
                    f" {syntax_errors + 1}) took {end_time - start_time:.2f}"
                    " seconds.[/cyan]"
                )
                generated_tests = response.choices[0].message.content.strip()
                generated_tests = clean_llm_response(
                    generated_tests
                )  # Clean again
                fixed_tests, had_errors = fix_tests(
                    generated_tests, file_base_name
                )  # Check again
                syntax_errors += 1
            except Timeout:
                console.print(
                    "[yellow]Timeout during test syntax correction (attempt"
                    f" {syntax_errors+1}).[/yellow]"
                )
                logging.warning(
                    "Timeout during test syntax correction (attempt %d)",
                    syntax_errors + 1,
                )
                if syntax_errors == MAX_SYNTAX_RETRIES:
                    console.print(
                        "[red]Max syntax retries reached for test generation."
                        " Skipping.[/red]"
                    )
                    return ""  # Give up on generating tests
                continue  # Try again

        if had_errors:
            console.print(
                "[red]Max syntax retries reached for test generation. Skipping.[/red]"
            )
            return ""

    except Timeout:
        console.print(
            "[yellow]Timeout during initial LLM call for test generation.[/yellow]"
        )
        logging.warning("Timeout during initial LLM call for test generation")
        return ""
    except Exception as e:
        console.print(
            f"[red]Error during LLM call for test generation: {e}[/red]"
        )
        logging.exception("Error during LLM call for test generation")
        return ""

    # Create the tests directory if it doesn't exist
    tests_dir = os.path.join(os.path.dirname(file_path), "..", "tests")
    os.makedirs(tests_dir, exist_ok=True)

    # Construct the test file path
    test_file_name = "test_" + os.path.basename(file_path)
    test_file_path = os.path.join(tests_dir, test_file_name)
    if debug:
        print(f"[DEBUG] Test file path: {test_file_path}")

    # Check if the test file already exists
    if os.path.exists(test_file_path):
        console.print(
            f"[yellow]Test file already exists: {test_file_path}. Skipping"
            " writing.[/yellow]"
        )
        return ""  # Don't overwrite existing tests

    # Write the generated tests to the test file
    try:
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(generated_tests)
        console.print(f"[green]Test file written to: {test_file_path}[/green]")
        if debug:
            print(
                "[DEBUG] Test file exists after write:"
                f" {os.path.exists(test_file_path)}"
            )

        return generated_tests  # Return the generated tests
    except Exception as e:
        console.print(f"[red]Error writing test file: {e}[/red]")
        logging.exception("Error writing test file")
        console.print(
            f"[debug] Generated test code:\n{generated_tests}"
        )  # Print the code for debugging
        return ""


def run_tests(
    repo_path: str,
    original_file_path: str,
    test_framework: str,
    min_coverage: float,
    coverage_fail_action: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """Runs tests (currently only pytest) and checks coverage."""
    test_results = {}
    tests_dir = os.path.join(repo_path, "tests")

    if not os.path.exists(tests_dir):
        console.print(f"[yellow]Tests directory not found: {tests_dir}[/yellow]")
        return {
            "output": "",
            "errors": "Tests directory not found",
            "returncode": 5,
        }

    if test_framework == "pytest":
        # Construct the pytest command, including coverage if requested
        command = ["pytest", "-v", tests_dir]
        if min_coverage is not None:
            rel_path = os.path.relpath(
                os.path.dirname(original_file_path), repo_path
            )
            command.extend(
                [f"--cov={rel_path}", "--cov-report", "term-missing"]
            )

        if debug:
            print(
                f"[DEBUG] Current working directory in run_tests: {repo_path}"
            )
            print(f"[DEBUG] Test command: {' '.join(command)}")

        stdout, stderr, returncode = run_command(
            command, cwd=repo_path
        )  # Run tests in the repo directory
        test_results = {
            "output": stdout,  # Store standard output
            "errors": stderr,  # Store error output
            "returncode": returncode,  # Store the return code
        }

        if test_results:
            console.print(f"[debug]Test return code: {test_results['returncode']}")
            console.print(f"[debug]Test output: {test_results['output']}")
            console.print(f"[debug]Test errors: {test_results['errors']}")

        # Provide feedback based on the test results
        if returncode == 0:
            console.print("[green]All tests passed.[/green]")
        elif returncode == 1:
            console.print("[red]Some tests failed.[/red]")  # Tests failed
        elif returncode == 5:
            console.print("[yellow]No tests found.[/yellow]")  # No tests collected
        else:
            console.print(
                f"[red]Error during test execution (code {returncode}).[/red]"
            )
            console.print(
                f"[debug] Pytest output:\n{stdout}"
            )  # Added for robusteness
            console.print(f"[debug] Pytest errors:\n{stderr}")

        return test_results

    else:
        console.print(
            f"[yellow]Unsupported test framework: {test_framework}[/yellow]"
        )
        return {
            "output": "",
            "errors": f"Unsupported framework: {test_framework}",
            "returncode": 1,
        }


def create_info_file(
    file_path: str,
    analysis_results: Dict,
    test_results: Dict,
    llm_success: bool,
    categories: List[str],
    optimization_level: str,
    output_info: str,
    min_coverage: float = None,
) -> None:
    """Generates and saves an info file (plain text) of the changes."""
    with open(output_info, "w", encoding="utf-8") as f:
        f.write(f"FabGPT Improvement Report for: {file_path}\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"LLM Improvement Success: {llm_success}\n")
        f.write(f"LLM Optimization Level: {optimization_level}\n")
        f.write(f"Categories Attempted: {', '.join(categories)}\n\n")

        f.write("Changes Made:\n")
        changes_made = []
        if shutil.which("black"):
            changes_made.append("Formatted with [Black](https://github.com/psf/black)")
        if shutil.which("isort"):
            changes_made.append("Formatted with [isort](https://pycqa.github.io/isort/)")
        if llm_success:
            changes_made.append(f"Applied LLM improvements ({optimization_level})")
        if test_results:  # Check if test_results exist
            changes_made.append("Generated/updated tests")
        if changes_made:
            for change in changes_made:
                f.write(f"* {change}\n")
        else:
            f.write("No changes made\n")

        f.write("\nStatic Analysis Results:\n")
        if analysis_results:
            for tool, result in analysis_results.items():
                if "returncode" in result:  # Check if tool ran
                    outcome = (
                        "OK"
                        if result["returncode"] == 0
                        else f"Errors/Warnings ({len(result.get('output', '').splitlines())})"
                    )
                    f.write(f"* {tool}: {outcome}\n")
                    if result["errors"]:  # Include error details
                        f.write(f"  Errors/Warnings:\n{result['errors']}\n")
                else:
                    f.write(f"* {tool}: Skipped (tool not found)\n")
        else:
            f.write("  No static analysis performed.\n")

        f.write("\nTest Results:\n")
        if test_results:
            test_outcome = (
                "Passed" if test_results["returncode"] == 0 else "Failed"
            )
            f.write(f"  Tests: {test_outcome}\n")
            # Extract and report coverage information, if available
            if "TOTAL" in test_results.get("output", ""):
                for line in test_results["output"].splitlines():
                    if line.lstrip().startswith("TOTAL"):
                        try:
                            coverage_percentage = float(
                                line.split()[-1].rstrip("%")
                            )
                            f.write(f"  Code Coverage: {coverage_percentage:.2f}%\n")
                            if coverage_percentage < min_coverage:
                                f.write(
                                    "  WARNING: Coverage is below the minimum"
                                    " threshold!\n"
                                )
                        except (ValueError, IndexError):
                            pass  # Ignore lines that don't have coverage
            if test_results["returncode"] != 0:
                f.write(
                    "  WARNING: Some tests failed!\n  Output:\n"
                    f"{test_results.get('output', '')}\n"
                )
        else:
            f.write("  No tests performed.\n")


def create_commit(
    repo: git.Repo,
    file_paths: List[str],  # Now accepts a list of file paths
    commit_message: str,
    test_results: Dict[str, Any] = None,
    llm_improvements_summary: Dict[str, List[str]] = None,
) -> None:
    """Creates a Git commit."""
    try:
        console.print("[blue]Creating commit...[/blue]")
        for fp in file_paths:
            repo.git.add(fp)  # Add each file individually
        if test_results is not None:
            tests_dir = os.path.join(repo.working_tree_dir, "tests")
            if os.path.exists(tests_dir):
                repo.git.add(tests_dir)

        # Prepend custom commit message
        commit_custom_file = os.path.join(repo.working_tree_dir, "commit_custom.txt")
        if os.path.exists(commit_custom_file):
            with open(commit_custom_file, "r", encoding="utf-8") as cc:
                custom_content = cc.read().strip()
            if custom_content:
                # Prepend custom commit message
                # commit_message = f"{custom_content}\n\n{commit_message\n\nCheck [this](https://github.com/fabriziosalmi/FabGPT) out!}"
                commit_message = f"{custom_content}\n\n{commit_message}"

        repo.index.commit(commit_message)

    except Exception as e:
        console.print(f"[red]Error creating commit:[/red] {e}")
        logging.exception("Error creating commit")
        sys.exit(1)


def format_for_commit_and_pr(file_improvements: Dict[str, str]) -> Tuple[str, str]:
    """Formats improvements for commit message (title and body) and PR body."""

    # Title:  Concise, mentioning all files.
    title = f"Improved: {', '.join(file_improvements.keys())}"

    # Body:  Detailed, per-file breakdown.
    body = ""
    for filename, formatted_summary in file_improvements.items():
        body += f"## Improvements for {filename}:\n\n"
        body += formatted_summary + "\n"

    return title, body


def create_pull_request_programatically(
    repo_url: str,
    token: str,
    base_branch: str,
    head_branch: str,
    commit_title: str,
    commit_body: str,
    analysis_results: Dict[str, Dict[str, Any]],
    test_results: Dict[str, Any],
    file_paths: List[str],
    optimization_level: str,
    test_framework: str,
    min_coverage: float,
    coverage_fail_action: str,
    repo_path: str,
    categories: List[str],
    debug: bool = False,
    force_push: bool = False,
) -> None:
    """Creates a GitHub Pull Request."""
    try:
        console.print("[blue]Creating Pull Request...[/blue]")
        g = Github(token)
        repo_name = repo_url.replace("https://github.com/", "")
        gh_repo = g.get_repo(repo_name)

        # Create the PR using the PyGithub repo object
        pr = gh_repo.create_pull(
            title=commit_title,  # Use the separate title
            body=commit_body,  # Use the separate body
            head=head_branch,  # Correct head format: "user:branch"
            base=base_branch,
        )

        console.print(f"[green]Pull Request created:[/green] {pr.html_url}")

    except Exception as e:
        console.print(f"[red]Error creating Pull Request:[/red] {e}")
        logging.exception("Error creating Pull Request")
        sys.exit(1)


@click.command()
@click.option("--repo", "-r", required=True, help="GitHub repository URL.")
@click.option(
    "--files",
    "-f",
    required=True,
    help="Comma-separated relative paths to files to improve.",
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
    "--exclude-tools",
    "-e",
    default="",
    help="Tools to exclude (comma-separated).",
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
    "--llm-optimization-level",
    "-l",
    default="balanced",
    help="LLM optimization level.",
)
@click.option(
    "--llm-custom-prompt",
    "-p",
    default=".",
    help="Path to a custom prompt directory.",
)
@click.option(
    "--test-framework", "-F", default="pytest", help="Test framework."
)
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
@click.option("--commit-message", "-cm", default=None, help="Custom commit message.")
@click.option(
    "--no-dynamic-analysis", is_flag=True, help="Disable dynamic analysis (testing)."
)
@click.option("--cache-dir", default=None, help="Directory for caching.")
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Run without making changes.")
@click.option(
    "--local-commit",
    is_flag=True,
    help="Only commit locally, don't create a Pull Request.",
)
@click.option(
    "--fast", is_flag=True, help="Enable fast mode by reducing delays."
)
@click.option(
    "--openai-api-base",
    default=None,
    help="Base URL for OpenAI API (for LMStudio, e.g., http://localhost:1234/v1).",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    callback=get_inputs,
    is_eager=True,
    expose_value=False,
)
@click.option("--no-output", is_flag=True, help="Disable console output.")
@click.option(
    "--categories",
    "-C",
    default="style,maintenance,security,performance",
    help=(
        "Comma-separated list of improvement categories.  Defaults to"
        " 'style,maintenance,security,performance'."
    ),
)
@click.option(
    "--force-push",
    is_flag=True,
    help="Force push the branch if it already exists.",
)
@click.option(
    "--output-file",
    "-o",
    default=None,
    help="Path to save the modified file. Defaults to overwriting the original.",
)
@click.option(
    "--output-info",
    default="report.txt",
    help="Path to save the TEXT report. Defaults to report.txt",
)
@click.option(
    "--line-length",
    type=int,
    default=DEFAULT_LINE_LENGTH,
    help="Maximum line length for code formatting.",
)
@click.option("--fork-repo", is_flag=True, help="Automatically fork the repository.")
@click.option("--fork-user", default=None, help="Your GitHub username (if different).")
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
    min_coverage: float,
    coverage_fail_action: str,
    commit_message: str,
    no_dynamic_analysis: bool,
    cache_dir: str,
    debug: bool,
    dry_run: bool,
    local_commit: bool,
    fast: bool,
    openai_api_base: str,
    no_output: bool,
    categories: str,
    force_push: bool,
    output_file: str,
    output_info: str,
    line_length: int,
    fork_repo: bool,
    fork_user: str,
) -> None:
    """
    Improves a Python file in a GitHub repository, generates tests, and creates a Pull Request.
    """
    if no_output:
        console.print = lambda *args, **kwargs: None  # Disable console output

    # --- API Key and Client Initialization ---
    ctx = click.get_current_context()
    config_values = ctx.default_map if ctx.default_map else {}
    api_base = config_values.get(
        "openai_api_base", openai_api_base or os.getenv("OPENAI_API_BASE")
    )
    api_key = config_values.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

    if debug:
        console.print("[yellow]Debug mode enabled.[/yellow]")
        console.print(f"[yellow]API Base: {api_base}[/yellow]")
        console.print(f"[yellow]API Key from env/config: {api_key}[/yellow]")
        console.print(f"[yellow]Effective Configuration: {config_values}[/yellow]")

    if api_key and api_key.lower() == "none":
        api_key = None  # Treat "none" as not provided

    if api_base:
        client = OpenAI(
            api_key="dummy", base_url=api_base, timeout=OPENAI_TIMEOUT
        )  # Use dummy key with custom base URL
    elif api_key:
        client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)
    else:
        console.print(
            "[red]Error: OpenAI API key not found. Set OPENAI_API_KEY environment"
            " variable, use --config, or --openai-api-key.[/red]"
        )
        sys.exit(1)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)  # Create cache directory

    # --- FORK HANDLING ---
    if fork_repo:
        # 1. Get the user's GitHub username
        if not fork_user:
            try:
                g = Github(token)
                fork_user = g.get_user().login
            except Exception as e:
                console.print(
                    f"[red]Error getting GitHub username: {e}.  Please provide --fork-user.[/red]"
                )
                sys.exit(1)
        console.print(f"[blue]Using GitHub username for forking: {fork_user}[/blue]")

        # 2. Fork the repository using PyGithub
        try:
            g = Github(token)
            original_repo_name = repo.replace("https://github.com/", "")
            original_repo = g.get_repo(original_repo_name)
            forked_repo = original_repo.create_fork()
            repo_url = forked_repo.clone_url  # Use the URL of the *forked* repo
            console.print(f"[green]Forked repository to: {repo_url}[/green]")

            # --- ADD DELAY HERE ---
            time.sleep(5)  # Wait 5 seconds (adjust as needed)
            
        except Exception as e:
            console.print(f"[red]Error forking repository: {e}[/red]")
            sys.exit(1)

    else:  # No forking
        repo_url = repo
        # 1. Get the user's GitHub username
        if not fork_user:
            try:
                g = Github(token)
                fork_user = g.get_user().login
            except Exception as e:
                console.print(
                    f"[red]Error getting GitHub username: {e}.  Please provide --fork-user.[/red]"
                )
                sys.exit(1)

    # --- Repository Cloning and Branch Handling ---
    repo_obj, temp_dir = clone_repository(
        repo_url, token
    )  # Clone the fork or original
    files_list = [f.strip() for f in files.split(",")]
    final_commit_message = (
        ""  # Initialize an empty string to store the commit message
    )
    improved_files_info = {}  # Store per-file improvements
    pr_url = None  # Initialize pr_url here

    # --- Branch Creation (BEFORE processing files) ---
    combined_file_purpose = "general_improvements"
    new_branch_name = create_branch(repo_obj, files_list, combined_file_purpose)
    checkout_branch(repo_obj, branch)  # Checkout the target branch
    checkout_branch(repo_obj, new_branch_name)  # Checkout new branch

    for file in files_list:
        file_path = os.path.join(temp_dir, file)
        file_purpose = get_file_purpose(file_path)

        # Capture original file content
        with open(file_path, "r", encoding="utf-8") as f:
            original_code = f.read()

        categories_list = [c.strip() for c in categories.split(",")]

        # --- Analysis and Test Generation ---
        analysis_results = {}
        test_results = None
        tests_generated = False

        if not no_dynamic_analysis:
            analysis_results = analyze_project(
                temp_dir,
                file_path,
                tools.split(","),
                exclude_tools.split(","),
                cache_dir,
                debug,
                line_length,
            )
            console.print("[blue]Test generation phase...[/blue]")
            generated_tests = generate_tests(
                file_path,
                client,
                llm_model,
                llm_temperature,
                test_framework,
                llm_custom_prompt,
                debug,
                line_length,
            )
            if generated_tests:
                tests_generated = True
                test_results = run_tests(
                    temp_dir,
                    file_path,
                    test_framework,
                    min_coverage,
                    coverage_fail_action,
                    debug,
                )

        # --- File Improvement ---
        console.print("[blue]File improvement phase...[/blue]")
        improved_code, llm_success = improve_file(
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

        # --- Check for Changes (before generating summary) ---
        with open(file_path, "r", encoding="utf-8") as f:
            new_code = f.read()
        if new_code.strip() == original_code.strip():
            console.print(
                f"[yellow]No changes detected for {file}. Skipping.[/yellow]"
            )
            continue  # Skip to the next file

        # --- Final Static Analysis Report (after modifications) ---
        final_analysis_results = analyze_project(
            temp_dir,
            file_path,
            tools.split(","),
            exclude_tools.split(","),
            cache_dir,
            debug,
            line_length,
        )
        analysis_diff = {}
        for tool in tools.split(","):
            if tool in exclude_tools.split(","):
                continue
            if tool not in analysis_results or tool not in final_analysis_results:
                continue

            initial_count = (
                len(analysis_results[tool].get("output", "").splitlines())
                if analysis_results[tool]["returncode"] != 0
                else 0
            )
            final_count = (
                len(final_analysis_results[tool].get("output", "").splitlines())
                if final_analysis_results[tool]["returncode"] != 0
                else 0
            )
            analysis_diff[tool] = final_count - initial_count
        # --- LLM Improvement Summary ---
        llm_improvements_summary = {}
        if llm_success:
            llm_improvements_summary = get_llm_improvements_summary(
                original_code,
                improved_code,
                categories_list,
                client,
                llm_model,
                llm_temperature,
            )
            formatted_summary = format_llm_summary(
                llm_improvements_summary
            )  # Deduplicated summary
        else:
            formatted_summary = "No LLM-driven improvements were made."

        improved_files_info[file] = formatted_summary  # Store the formatted summary

        # --- Save Improved Code (if requested) ---
        if output_file:
            base, ext = os.path.splitext(output_file)
            output_file_for_current = f"{base}_{file}{ext}"
            try:
                output_file_for_current = os.path.abspath(output_file_for_current)
                with open(output_file_for_current, "w", encoding="utf-8") as f:
                    f.write(improved_code)
                console.print(
                    f"[green]Improved code for {file} saved to: {output_file_for_current}[/green]"
                )

            except Exception as e:
                console.print(
                    f"[red]Error saving improved code to {output_file_for_current}: {e}[/red]"
                )
                logging.exception(
                    "Error saving improved code to %s", output_file_for_current
                )
                sys.exit(1)

        # --- Create and Save Info File ---
        create_info_file(
            file_path,
            final_analysis_results,
            test_results,
            llm_success,
            categories_list,
            llm_optimization_level,
            output_info,
            min_coverage,
        )

    # --- Commit and Pull Request (outside the file loop) ---
    if not dry_run:
        # Format for commit and PR
        commit_title, commit_body = format_for_commit_and_pr(improved_files_info)
        create_commit(
            repo_obj,
            files_list,
            f"{commit_title}\n\n{commit_body}",
            test_results,
        )  # combine title and body
        if not local_commit:
            # --- PUSH (to your fork) ---
            try:
                if force_push:
                    repo_obj.git.push("--force", "origin", new_branch_name)
                else:
                    repo_obj.git.push("origin", new_branch_name)
                console.print(
                    f"[green]Branch pushed to your fork: {new_branch_name}[/green]"
                )  # Clearer message
            except git.exc.GitCommandError as e:
                console.print(
                    f"[red]Error pushing branch to your fork: {e}[/red]"
                )  # and here
                logging.exception("Error pushing branch")
                sys.exit(1)  # Use sys.exit

            # --- CREATE PULL REQUEST (from your fork to the original repo) ---
            # This is the key part that changes:
            create_pull_request_programatically(
                repo,  # Original Repo
                token,  # Your Token
                branch,  # Base Branch (e.g., main, develop)
                f"{fork_user}:{new_branch_name}",  # Head branch: YOURUSER:your-branch-name
                commit_title,
                commit_body,
                final_analysis_results,
                test_results,
                files_list,
                llm_optimization_level,
                test_framework,
                min_coverage,
                coverage_fail_action,
                temp_dir,  # Pass temp_dir instead of repo_path
                categories_list,
                debug,
                force_push,
            )

    # --- Save JSON Log ---
    log_data = {
        "repository": repo,
        "branch": branch,  # Original branch
        "files_improved": improved_files_info,
        "commit_message": final_commit_message,
        "pr_url": pr_url,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    log_dir = os.path.join(
        "/Users/fab/GitHub/FabGPT", "logs"
    )  # set up your correct log dir
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"log_{int(time.time())}.json")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        json.dump(log_data, log_file, indent=4)

    # --- Cleanup ---
    if not debug:
        shutil.rmtree(temp_dir)

    console.print("[green]All operations completed successfully.[/green]")
    sys.exit(0)


if __name__ == "__main__":
    main()