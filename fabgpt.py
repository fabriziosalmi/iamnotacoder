import uuid  # Add this import at the top of your file
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
from rich.progress import Progress
from rich.table import Table
import json
from github import Github
import hashlib
import time
import ast
from typing import List, Dict, Tuple, Any
import re  # Import the regular expression module

console = Console()

# Constants
DEFAULT_LLM_MODEL = "qwen2.5-coder-7b-instruct"  # Or your preferred model
DEFAULT_LLM_TEMPERATURE = 0.2  # Lower temperature for more deterministic output
MAX_SYNTAX_RETRIES = 5  # Increased retry count
MAX_LLM_RETRIES = 3
OPENAI_TIMEOUT = 120.0


def run_command(command: List[str], cwd: str = None) -> Tuple[str, str, int]:
    """Executes a shell command and returns stdout, stderr, and return code."""
    try:
        start_time = time.time()
        # Safer: Use shell=False and pass arguments as a list.
        result = subprocess.run(
            command, capture_output=True, text=True, cwd=cwd, check=True
        )
        end_time = time.time()
        console.print(f"[cyan]Command `{' '.join(command)}` executed in {end_time - start_time:.2f} seconds.[/cyan]")
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode
    except FileNotFoundError as e:  # Catch FileNotFoundError specifically
        console.print(f"[red]Command not found:[/red] {e}")
        return "", str(e), 1
    except Exception as e:
        console.print(f"[red]Unhandled error during command execution:[/red] {e}")
        return "", str(e), 1

def load_config(config_file: str) -> dict:
    """Loads configuration from a TOML file."""
    try:
        with open(config_file, "r") as f:
            return toml.load(f)
    except Exception as e:
        console.print(f"[red]Error loading configuration file:[/red] {e}")
        exit(1)

def create_backup(file_path: str) -> str:
    """Creates a backup copy of the file, with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        console.print(f"[green]Backup created:[/green] {backup_path}")
        return backup_path
    except Exception as e:
        console.print(f"[red]Error creating backup:[/red] {e}")
        return None

def restore_backup(file_path: str, backup_path: str) -> None:
    """Restores the file from the backup copy."""
    try:
        shutil.copy2(backup_path, file_path)
        console.print(f"[green]File restored from:[/green] {backup_path}")
    except Exception as e:
        console.print(f"[red]Error restoring backup:[/red] {e}")

def get_inputs(ctx: click.Context, param: click.Parameter, value: Any) -> dict:
    """Handles inputs: CLI > config file > defaults."""
    config = {}
    if ctx.default_map:
        config.update(ctx.default_map)

    if value:
        config.update(load_config(value))

    for k, v in ctx.params.items():
        if v is not None:
            config[k] = v

    ctx.default_map = config
    return config

def clone_repository(repo_url: str, token: str) -> Tuple[git.Repo, str]:
    """Clones the repository (shallow clone) into a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    auth_repo_url = repo_url.replace("https://", f"https://{token}@")

    try:
        console.print(f"[blue]Cloning repository (shallow): {repo_url}[/blue]")
        start_time = time.time()
        # Use depth=1 for shallow clone
        repo = git.Repo.clone_from(auth_repo_url, temp_dir, depth=1)
        end_time = time.time()
        console.print(f"[green]Repository cloned to:[/green] {temp_dir} in {end_time - start_time:.2f} seconds")
        return repo, temp_dir
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error cloning repository:[/red] {e}")
        exit(1)

def checkout_branch(repo: git.Repo, branch_name: str) -> None:
     """Checks out the specified branch, fetching and pruning updates first."""
     try:
         console.print(f"[blue]Checking out branch: {branch_name}[/blue]")
         start_time = time.time()
         repo.git.fetch("--all", "--prune")  # Fetch and prune all remotes.  THIS IS KEY.
         repo.git.checkout(branch_name)
         end_time = time.time()
         console.print(f"[green]Checked out branch:[/green] {branch_name} in {end_time - start_time:.2f} seconds")
     except git.exc.GitCommandError:
        try:
            console.print(f"[yellow]Attempting to fetch branch {branch_name}[/yellow]")
            start_time = time.time()
            repo.git.fetch("origin", branch_name)
            repo.git.checkout(f"origin/{branch_name}")  # Checkout remote branch
            end_time = time.time()
            console.print(f"[green]Checked out branch:[/green] {branch_name} in {end_time - start_time:.2f} seconds")
        except git.exc.GitCommandError as e:
            console.print(f"[red]Error checking out branch:[/red] {e}")
            exit(1)

def create_branch(repo: git.Repo, file_name: str, file_purpose: str = "") -> str:
    """Creates a new branch with a highly unique name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sanitized_file_name = "".join(c if c.isalnum() else "_" for c in file_name)
    # Use a UUID to make the branch name virtually guaranteed to be unique
    unique_id = uuid.uuid4().hex  # Generate a UUID and get its hex representation
    branch_name = f"improvement-{sanitized_file_name}-{file_purpose}-{timestamp}-{unique_id}" #Added UUID

    try:
        console.print(f"[blue]Creating branch: {branch_name}[/blue]")
        start_time = time.time()
        repo.git.checkout("-b", branch_name)
        end_time = time.time()
        console.print(f"[green]Created branch:[/green] {branch_name} in {end_time - start_time:.2f} seconds")
        return branch_name
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error creating branch:[/red] {e}")
        exit(1)

def get_file_purpose(file_path: str) -> str:
    """Attempts to determine the file's purpose (for branch naming)."""
    try:
        # This is a very basic heuristic.  More sophisticated methods could be used.
        with open(file_path, "r") as f:
            first_line = f.readline()
            if "def " in first_line:
                return first_line.split("def ")[1].split("(")[0].strip()
            elif "class " in first_line:
                return first_line.split("class ")[1].split(":")[0].strip()
            else:
                return ""
    except Exception:
        return ""

def analyze_project(repo_path: str, file_path: str, tools: List[str], exclude_tools: List[str], cache_dir: str = None, debug: bool = False) -> Dict[str, Dict[str, Any]]:
    """Performs static analysis, handling caching and tool availability."""
    cache_key_data = f"{file_path}-{','.join(sorted(tools))}-{','.join(sorted(exclude_tools))}".encode('utf-8')
    cache_key = hashlib.sha256(cache_key_data).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json") if cache_dir else None

    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_results = json.load(f)
            console.print("[blue]Using static analysis results from cache.[/blue]")
            return cached_results
        except Exception as e:
            console.print(f"[yellow]Error loading cache, re-running analysis.[/yellow]")

    results = {}
    console.print("[blue]Running static analysis...[/blue]")
    with Progress(
        "[progress.description]{task.description}",
        "•",
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        "[progress.completed]{task.completed}/{task.total}",
        transient=True,
    ) as progress:
        analysis_task = progress.add_task("Analyzing...", total=len(tools))
        for tool in tools:
            if tool in exclude_tools:
                progress.update(analysis_task, advance=1)
                continue

            # Check if the tool is installed.
            if shutil.which(tool) is None:
                console.print(f"[yellow]Tool not found: {tool}. Skipping.[/yellow]")
                progress.update(analysis_task, advance=1)
                continue

            if tool == "pylint":
                command = ["pylint", file_path]
            elif tool == "flake8":
                command = ["flake8", file_path]
            elif tool == "black":
                command = ["black", "--check", "--diff", file_path]
            elif tool == "isort":
                command = ["isort", "--check-only", "--diff", file_path]
            elif tool == "mypy":
                command = ["mypy", file_path]
            else:
                console.print(f"[yellow]Unknown analysis tool: {tool}[/yellow]")
                progress.update(analysis_task, advance=1)
                continue

            stdout, stderr, returncode = run_command(command, cwd=repo_path)
            results[tool] = {
                "output": stdout,
                "errors": stderr,
                "returncode": returncode,
            }
            progress.update(analysis_task, advance=1)

    if cache_file:
        try:
            with open(cache_file, "w") as f:
                json.dump(results, f)
            console.print(f"[blue]Static analysis results saved to cache.[/blue]")
        except Exception as e:
            console.print(f"[yellow]Error saving to cache.[/yellow]")

    return results

def clean_llm_response(response_text: str) -> str:
    """Removes extraneous text and code fences from LLM responses."""

    # First, try to extract content within code blocks (```python ... ```)
    code_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", response_text, re.DOTALL)
    if code_blocks:
        # Use the *last* full code block (heuristic: often better)
        return code_blocks[-1].strip()

    # If no code blocks found, fall back to line-based cleaning
    lines = response_text.splitlines()
    cleaned_lines = []
    started = False  # Flag to indicate we've found the start of code

    for line in lines:
        line = line.strip()
        # Start accumulating lines when a valid Python statement is found
        if not started:
            if (
                line.startswith("import ")
                or line.startswith("def ")
                or line.startswith("class ")
                or (line and not line.startswith("#") and not line[0].isspace())  # Heuristic for code
            ):
                started = True
        if started:
             # Stop if we hit something that's clearly not code.
            if line.lower().startswith("return only the"):  # Common LLM phrase.
                break
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def improve_file(file_path: str, client: OpenAI, llm_model: str, llm_temperature: float, categories: List[str], custom_prompt_dir: str, analysis_results: Dict[str, Dict[str, Any]], debug: bool = False) -> Tuple[str, bool]:
    """Improves the file using the LLM, iteratively by category."""
    backup_path = create_backup(file_path)
    if not backup_path:
        console.print("[red]Failed to create backup. Aborting.[/red]")
        exit(1)

    # Check and run black/isort only if they are installed
    if shutil.which("black"):
        console.print(f"[blue]Formatting with Black...[/blue]")
        run_command(["black", file_path], cwd=os.path.dirname(file_path))
    else:
        console.print("[yellow]Black not found, skipping formatting.[/yellow]")

    if shutil.which("isort"):
        console.print(f"[blue]Formatting with Isort...[/blue]")
        run_command(["isort", file_path], cwd=os.path.dirname(file_path))
    else:
        console.print("[yellow]Isort not found, skipping formatting.[/yellow]")


    with open(file_path, "r") as f:
        improved_code = f.read()

    total_success = True  # Track overall success
    with Progress(
        "[progress.description]{task.description}",
        "•",
        "[progress.percentage]{task.percentage:>3.0f}%",
        transient=True,
        ) as progress:

        improve_task = progress.add_task("Improving file...", total=len(categories))
        for category in categories:
            progress.update(improve_task, description=f"[blue]Improving category: {category}[/blue]")
            prompt_file = os.path.join(custom_prompt_dir, f"prompt_{category}.txt")
            if not os.path.exists(prompt_file):
                console.print(f"[red]Prompt file not found: {prompt_file}. Skipping category.[/red]")
                progress.update(improve_task, advance=1)
                continue
            try:
                with open(prompt_file, "r") as f:
                    prompt_template = f.read()
                    # More specific prompt instructions.
                    prompt = prompt_template.replace("{code}", improved_code)
                    prompt += "\nReturn only the corrected code, without any introductory or concluding text. Do not include markdown code fences (```)."
            except Exception as e:
                console.print(f"[red]Error loading custom prompt: {e}[/red]")
                progress.update(improve_task, advance=1)
                continue

            if debug:
                console.print(f"[debug]LLM prompt for {category}:\n{prompt}")

            success = False
            for attempt in range(MAX_LLM_RETRIES):
                try:
                    start_time = time.time()
                    response = client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful coding assistant that improves code quality."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=llm_temperature,
                        max_tokens=4096
                    )
                    end_time = time.time()
                    console.print(f"[cyan]LLM request for {category} (attempt {attempt + 1}) took {end_time - start_time:.2f} seconds.[/cyan]")

                    improved_code = response.choices[0].message.content.strip()
                    improved_code = clean_llm_response(improved_code) # Clean the llm response


                    # Inline syntax correction (similar to test generation)
                    syntax_errors = 0
                    while syntax_errors < MAX_SYNTAX_RETRIES:
                        try:
                            ast.parse(improved_code)
                            break  # No syntax errors
                        except SyntaxError as e:
                            console.print(f"[yellow]Syntax error in generated code (attempt {syntax_errors + 1}), retrying...[/yellow]")
                            line_number = e.lineno
                            error_message = str(e)

                            # Get code context
                            code_lines = improved_code.splitlines()
                            start_line = max(0, line_number - 3)  # 2 lines before
                            end_line = min(len(code_lines), line_number + 2)  # 2 lines after
                            context = "\n".join(code_lines[start_line:end_line])

                            # Highlight the error line
                            highlighted_context = context.replace(code_lines[line_number -1], f"#> {code_lines[line_number - 1]}")

                            syntax_prompt = (
                                f"Fix the following Python code that has a syntax error on line {line_number}:\n\n"
                                f"```python\n{highlighted_context}\n```\n\nError: {error_message}\n\n"
                                "Return only the corrected code, without any introductory or concluding text. Do not include markdown code fences."
                            )

                            syntax_errors += 1
                            start_time = time.time()

                            try:
                                response = client.chat.completions.create(
                                    model=llm_model,
                                    messages=[
                                        {"role": "system", "content": "You are a helpful coding assistant that improves code quality."},
                                        {"role": "user", "content": syntax_prompt}
                                    ],
                                    temperature=min(llm_temperature, 0.2),  # Lower temp for syntax correction
                                    max_tokens=4096
                                )
                                end_time = time.time()
                                console.print(f"[cyan]LLM retry for syntax correction (attempt {syntax_errors}) took {end_time - start_time:.2f} seconds.[/cyan]")
                                improved_code = response.choices[0].message.content.strip()
                                improved_code = clean_llm_response(improved_code) # Clean the llm response
                            except Timeout:
                                console.print(f"[yellow]Timeout during syntax correction attempt {syntax_errors}.[/yellow]")
                                if syntax_errors == MAX_SYNTAX_RETRIES:
                                     console.print(f"[red]Max syntax correction attempts reached for {category}. Skipping.[/red]")
                                     break #exit inner loop
                                continue # retry
                    if syntax_errors == MAX_SYNTAX_RETRIES:
                        console.print(f"[red]Max syntax correction attempts reached for {category}. Skipping.[/red]")
                        break # exit inner loop

                    with open(file_path, "w") as f:
                        f.write(improved_code)
                    success = True
                    break  # Exit the attempt loop if successful

                except Timeout:
                    console.print(f"[yellow]Timeout during LLM call for {category} (attempt {attempt + 1}).[/yellow]")
                    if attempt < MAX_LLM_RETRIES - 1:
                        time.sleep(2)  # Wait before retrying
                    else:
                        console.print(f"[red]Max LLM retries reached for {category}.[/red]")
                        success = False
                        break

                except Exception as e:
                    console.print(f"[red]Error during LLM call for {category} (attempt {attempt + 1}): {e}[/red]")
                    if attempt < MAX_LLM_RETRIES - 1:
                        time.sleep(2)
                    else:
                        console.print(f"[red]Max LLM retries reached for {category}.[/red]")
                        success = False
                        break

            if not success:
                restore_backup(file_path, backup_path)
                console.print(f"[yellow]Restoring backup due to failure in {category} improvements.[/yellow]")
                total_success = False # Update overall success
            progress.update(improve_task, advance=1)

    return improved_code, total_success

def fix_tests(generated_tests: str, file_base_name: str) -> Tuple[str, bool]:
    """Analyzes generated test code and creates a corrective prompt if needed."""
    try:
        ast.parse(generated_tests)
        return generated_tests, False  # No errors
    except SyntaxError as e:
        console.print(f"[yellow]Syntax error in test generation: {e}[/yellow]")
         # Get more specific error information
        line_number = e.lineno
        error_message = str(e)

        # Get code context
        code_lines = generated_tests.splitlines()
        start_line = max(0, line_number - 3)  # 2 lines before
        end_line = min(len(code_lines), line_number + 2)  # 2 lines after
        context = "\n".join(code_lines[start_line:end_line])
        # Highlight error line
        highlighted_context = context.replace(code_lines[line_number-1], f"#>>> {code_lines[line_number-1]}")


        # Include line number in the error message
        error_message_with_line = (
            f"Syntax error in generated tests for {file_base_name}.py on line {line_number}: {error_message}.  "
            f"Please fix the following code:\n```python\n{highlighted_context}\n```\n\n"
            "Return only the corrected code, without any introductory or concluding text. Do not include markdown code fences."
            )
        return error_message_with_line, True  # Return error message and True for had_errors

def generate_tests(file_path: str, client: OpenAI, llm_model: str, llm_temperature: float, test_framework: str, custom_prompt_dir: str, debug: bool = False) -> str:
    """Generates tests using the LLM and attempts to fix syntax errors."""
    with open(file_path, "r") as f:
        code = f.read()

    file_base_name = os.path.basename(file_path).split(".")[0]

    prompt_file = os.path.join(custom_prompt_dir, "prompt_tests.txt")
    if not os.path.exists(prompt_file):
        console.print(f"[red]Prompt file for tests not found: {prompt_file}.[/red]")
        return ""  # Return empty string if prompt file is missing
    try:
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
            prompt = prompt_template.replace("{code}", code).replace("{file_base_name}", file_base_name)
            prompt += "\nReturn only the test code, without any introductory or concluding text.  Do not include markdown code fences (```)."

    except Exception as e:
        console.print(f"[red]Error reading test prompt file: {e}[/red]")
        return ""
    if debug:
        console.print(f"[debug]LLM prompt for test generation:\n{prompt}")

    # LLM interaction
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful coding assistant that generates tests."},
                {"role": "user", "content": prompt}
            ],
            temperature=llm_temperature,
            max_tokens=4096  # Adjust as needed
        )
        end_time = time.time()
        console.print(f"[cyan]LLM request for test generation took {end_time - start_time:.2f} seconds.[/cyan]")
        generated_tests = response.choices[0].message.content.strip()
        generated_tests = clean_llm_response(generated_tests)  # Clean LLM response


        # Use fix_tests to handle syntax errors
        fixed_tests, had_errors = fix_tests(generated_tests, file_base_name)

        syntax_errors = 0
        while had_errors and syntax_errors < MAX_SYNTAX_RETRIES:
            console.print("[yellow]Attempting to fix syntax errors in generated tests...[/yellow]")
            # Retry with a corrective prompt
            start_time = time.time()
            # Use the detailed error message from fix_tests
            error_message = fixed_tests

            try:
                response = client.chat.completions.create(
                        model=llm_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful coding assistant that generates tests. Fix the following code that has syntax errors."},
                            {"role": "user", "content": error_message}  # Use the detailed error message
                        ],
                        temperature=min(llm_temperature, 0.2),  # Lower temp for corrections
                        max_tokens=4096
                )
                end_time = time.time()
                console.print(f"[cyan]LLM retry for test generation (attempt {syntax_errors + 1}) took {end_time - start_time:.2f} seconds.[/cyan]")
                generated_tests = response.choices[0].message.content.strip()
                generated_tests = clean_llm_response(generated_tests) # Clean the llm response
                fixed_tests, had_errors = fix_tests(generated_tests, file_base_name)
                syntax_errors += 1
            except Timeout:
                console.print(f"[yellow]Timeout during test syntax correction (attempt {syntax_errors+1}).[/yellow]")
                if syntax_errors == MAX_SYNTAX_RETRIES:
                    console.print(f"[red]Max syntax retries reached for test generation. Skipping.[/red]")
                    return ""
                continue # retry

        if had_errors:
            console.print(f"[red]Max syntax retries reached for test generation. Skipping.[/red]")
            return ""


    except Timeout:
        console.print(f"[yellow]Timeout during initial LLM call for test generation.[/yellow]")
        return ""

    except Exception as e:
        console.print(f"[red]Error during LLM call for test generation: {e}[/red]")
        return ""


    tests_dir = os.path.join(os.path.dirname(file_path), "..", "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_file_name = "test_" + os.path.basename(file_path)
    test_file_path = os.path.join(tests_dir, test_file_name)
    if debug: # only print if debug is enabled
        print(f"[DEBUG] Test file path: {test_file_path}")

    # Check if test file already exists
    if os.path.exists(test_file_path):
        console.print(f"[yellow]Test file already exists: {test_file_path}. Skipping writing.[/yellow]")
        return ""

    try:
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(generated_tests)
        console.print(f"[green]Test file written to: {test_file_path}[/green]")
        if debug:  # only print if debug mode is enabled
            print(f"[DEBUG] Test file exists after write: {os.path.exists(test_file_path)}")
        return generated_tests
    except Exception as e:
        console.print(f"[red]Error writing test file: {e}[/red]")
        console.print(f"[debug] Generated test code:\n{generated_tests}")
        return ""

def run_tests(repo_path: str, original_file_path: str, test_framework: str, min_coverage: float, coverage_fail_action: str, debug: bool = False) -> Dict[str, Any]:
    """Runs tests, measures coverage (optional), and handles results."""
    test_results = {}
    tests_dir = os.path.join(repo_path, "tests")  # Construct the tests directory path

    if not os.path.exists(tests_dir):
        console.print(f"[yellow]Tests directory not found: {tests_dir}[/yellow]")
        return {"output": "", "errors": "Tests directory not found", "returncode": 5}


    if test_framework == "pytest":
        # Construct the tests directory path

        command = ["pytest", "-v", tests_dir]  # Pass the tests directory to pytest
        if min_coverage is not None:
            # Get relative path from repo root to the *directory* containing the file
            rel_path = os.path.relpath(os.path.dirname(original_file_path), repo_path)
            command.extend([f"--cov={rel_path}", "--cov-report", "term-missing"])


        if debug:
            print(f"[DEBUG] Current working directory in run_tests: {repo_path}")
            print(f"[DEBUG] Test command: {' '.join(command)}") # Debug print
        stdout, stderr, returncode = run_command(command, cwd=repo_path)  # cwd is correct
        test_results = {
            "output": stdout,
            "errors": stderr,
            "returncode": returncode,
        }

        if returncode == 0:
            console.print("[green]All tests passed.[/green]")
        elif returncode == 1:
            console.print("[red]Some tests failed.[/red]")
        elif returncode == 5:
            console.print("[yellow]No tests found.[/yellow]")
        else:
            console.print(f"[red]Error during test execution (code {returncode}).[/red]")
            console.print(f"[debug] Pytest output:\n{stdout}")
            console.print(f"[debug] Pytest errors:\n{stderr}")

        return test_results
    else:
        console.print(f"[yellow]Unsupported test framework: {test_framework}[/yellow]")
        return {"output": "", "errors": f"Unsupported framework: {test_framework}", "returncode": 1}

def create_commit(repo: git.Repo, file_path: str, commit_message: str, test_results: Dict[str, Any] = None) -> None:
    """Creates a commit with the changes. (No changes here, just for context)"""
    try:
        console.print("[blue]Creating commit...[/blue]")
        repo.git.add(file_path)
        if test_results is not None:
            tests_dir = os.path.join(repo.working_tree_dir, "tests")
            if os.path.exists(tests_dir):
                repo.git.add(tests_dir)
        repo.index.commit(commit_message)
        console.print(f"[green]Commit created:[/green] {commit_message}")
    except Exception as e:
        console.print(f"[red]Error creating commit:[/red] {e}")
        exit(1)

def create_pull_request(repo_url: str, token: str, base_branch: str, head_branch: str, commit_message: str, analysis_results: Dict[str, Dict[str, Any]], test_results: Dict[str, Any], file_path: str, optimization_level: str, test_framework: str, min_coverage: float, coverage_fail_action:str, repo_path: str, debug: bool = False) -> None:
    """Creates a Pull Request on GitHub."""
    try:
        console.print(f"[blue]Creating Pull Request...[/blue]")
        g = Github(token)
        repo_name = repo_url.replace("https://github.com/", "")
        repo = g.get_repo(repo_name)

        # Get the user's login (username)
        user = g.get_user()
        username = user.login

        # Namespace the head branch
        namespaced_head = f"{username}:{head_branch}"

        body = f"## Pull Request: Improvements to {os.path.basename(file_path)}\n\n"
        body += "Changes made:\n\n"
        body += "* Code formatting with Black and isort (if installed).\n"
        body += f"* Improvements suggested by LLM (level: {optimization_level}).\n"
        if test_results and test_results.get('returncode', 1) == 0:
            body += "* Added/Updated unit tests.\n"

        if analysis_results:
            table = Table(title="Static Analysis Results")
            table.add_column("Tool", style="cyan")
            table.add_column("Result", style="magenta")
            for tool, result in analysis_results.items():
                outcome = "[green]OK[/green]" if result['returncode'] == 0 else f"[red]Errors/Warnings ({len(result.get('output', '').splitlines())})[/red]" if 'returncode' in result else "[yellow]Skipped[/yellow]"
                table.add_row(tool, outcome)
            body += "\n" + str(table) + "\n"


        if test_results:
            test_outcome = "[green]Passed[/green]" if test_results['returncode'] == 0 else f"[red]Failed[/red]"
            body += f"\n| Tests | {test_outcome} |\n"

            if "TOTAL" in test_results.get('output', ''):
                for line in test_results['output'].splitlines():
                    if line.lstrip().startswith("TOTAL"):
                        try:
                            coverage_percentage = float(line.split()[-1].rstrip("%"))
                            body += f"\n**Code Coverage:** {coverage_percentage:.2f}%\n"
                            if min_coverage and coverage_percentage < min_coverage:
                                if coverage_fail_action == "warn":
                                    body += f"\n**[WARNING] Coverage is below the minimum threshold! ({min_coverage}%)**\n"
                                else:
                                    body += f"\n**[ERROR] Coverage is below the minimum threshold! ({min_coverage}%)**\n"

                        except (ValueError, IndexError):
                            pass

            if test_results['returncode'] != 0:
                body += "\n**[WARNING] Some tests failed!  Check the CI results for details.**\n"

        body += "\n---\n\nTo run tests manually:\n\n```bash\n"
        if test_framework == 'pytest':
            rel_path = os.path.dirname(os.path.relpath(file_path, repo_path))
            body += f"pytest -v --cov={rel_path} --cov-report term-missing\n"
        elif test_framework == 'unittest':
            body += "python -m unittest discover\n"
        body += "```"

        pr = repo.create_pull(
            title=commit_message if commit_message else "Refactor: Automatic improvements",
            body=body,
            head=namespaced_head,  # Use the namespaced head
            base=base_branch
        )
        console.print(f"[green]Pull Request created:[/green] {pr.html_url}")

    except Exception as e:
        console.print(f"[red]Error creating Pull Request:[/red] {e}")
        exit(1)


@click.command()
@click.option("--repo", "-r", required=True, help="GitHub repository URL.")
@click.option("--file", "-f", required=True, help="Relative path to the file to improve.")
@click.option("--branch", "-b", required=True, help="Target branch name.")
@click.option("--token", "-t", required=True, help="GitHub Personal Access Token (PAT).")
@click.option("--tools", "-T", default="black,isort,pylint,flake8,mypy", help="Static analysis tools (comma-separated).")
@click.option("--exclude-tools", "-e", default="", help="Tools to exclude (comma-separated).")
@click.option("--llm-model", "-m", default=DEFAULT_LLM_MODEL, help="LLM model to use.")
@click.option("--llm-temperature", "-temp", type=float, default=DEFAULT_LLM_TEMPERATURE, help="Temperature for the LLM.")
@click.option("--llm-optimization-level", "-l", default="balanced", help="LLM optimization level.")
@click.option("--llm-custom-prompt", "-p", default=".", help="Path to a custom prompt directory.")
@click.option("--test-framework", "-F", default="pytest", help="Test framework.")
@click.option("--min-coverage", "-c", default=None, type=float, help="Minimum code coverage threshold.")
@click.option("--coverage-fail-action", default="fail", type=click.Choice(["fail", "warn"]), help="Action on insufficient coverage.")
@click.option("--commit-message", "-cm", default=None, help="Custom commit message.")
@click.option("--no-dynamic-analysis", is_flag=True, help="Disable dynamic analysis (testing).")
@click.option("--cache-dir", default=None, help="Directory for caching.")
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--dry-run", is_flag=True, help="Run without making changes.")
@click.option("--local-commit", is_flag=True, help="Only commit locally, don't create a Pull Request.")
@click.option("--fast", is_flag=True, help="Enable fast mode by reducing delays.")
@click.option("--openai-api-base", default=None, help="Base URL for OpenAI API (for LMStudio, e.g., http://localhost:1234/v1).")
@click.option("--config", default=None, type=click.Path(exists=True), callback=get_inputs, is_eager=True, expose_value=False)
@click.option("--no-output", is_flag=True, help="Disable console output.")
@click.option("--categories", "-C", default="style,maintenance", help="Comma-separated list of improvement categories.")
def main(repo: str, file: str, branch: str, token: str, tools: str, exclude_tools: str, llm_model: str, llm_temperature: float,
         llm_optimization_level: str, llm_custom_prompt: str, test_framework: str, min_coverage: float,
         coverage_fail_action: str, commit_message: str, no_dynamic_analysis: bool, cache_dir: str, debug: bool,
         dry_run: bool, local_commit: bool, fast: bool, openai_api_base: str, no_output: bool, categories: str) -> None:
    """Improves a Python file in a GitHub repository, generates tests, and creates a Pull Request."""
    if no_output:
        console.print = lambda *args, **kwargs: None

    ctx = click.get_current_context()
    config_values = ctx.default_map if ctx.default_map else {}
    api_base = config_values.get("openai_api_base", openai_api_base or os.getenv("OPENAI_API_BASE"))
    api_key = config_values.get("openai_api_key", os.getenv("OPENAI_API_KEY"))


    if debug:
        console.print("[yellow]Debug mode enabled.[/yellow]")
        console.print(f"[yellow]API Base: {api_base}[/yellow]")
        console.print(f"[yellow]API Key from env/config: {api_key}[/yellow]")
        console.print(f"[yellow]Effective Configuration: {config_values}[/yellow]")


    if api_base:
      client = OpenAI(api_key="dummy", base_url=api_base, timeout=OPENAI_TIMEOUT)
    else:
      client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    repo_obj, temp_dir = clone_repository(repo, token)
    file_path = os.path.join(temp_dir, file)
    checkout_branch(repo_obj, branch)
    file_purpose = get_file_purpose(file_path)
    categories_list = [c.strip() for c in categories.split(",")]

    # --- Analysis and Test Generation (Keep track of what happened) ---
    analysis_results = {}  # Initialize for later use
    test_results = None
    tests_generated = False

    if not no_dynamic_analysis:
        analysis_results = analyze_project(
            temp_dir, file_path, tools.split(","), exclude_tools.split(","), cache_dir, debug
        )
        console.print("[blue]Test generation phase...[/blue]")
        with Progress(
            "[progress.description]{task.description}",
            transient=True,
            ) as progress:
            progress.add_task("[cyan]Generating tests...", total=None)
            generated_tests = generate_tests(
                file_path, client, llm_model, llm_temperature, test_framework, llm_custom_prompt, debug
            )
            if generated_tests:
                tests_generated = True  # Set the flag!
                console.print("[blue]Test execution phase...[/blue]")
                with Progress(
                    "[progress.description]{task.description}",
                    transient=True,
                ) as progress:
                    progress.add_task("[cyan]Running tests...", total=None)
                    test_results = run_tests(temp_dir, file_path, test_framework, min_coverage, coverage_fail_action, debug)

    new_branch_name = create_branch(repo_obj, file, file_purpose)

    console.print("[blue]File improvement phase...[/blue]")
    improved_code, llm_success = improve_file(
        file_path, client, llm_model, llm_temperature, categories_list, llm_custom_prompt, analysis_results, debug
    )

    if not dry_run:
        console.print("[blue]Commit phase...[/blue]")

        # --- Build the Commit Message ---
        if commit_message:  # Use custom message if provided
             final_commit_message = commit_message
        else:
            # 1. Summary Line
            base_name = os.path.basename(file_path)
            summary = f"refactor({base_name}): Improve code"
            if not llm_success:
                summary += " (LLM improvements failed)"  # Be specific if LLM failed

            # 2. Detailed Body (Optional, but very helpful)
            body_lines = []
            # Add a line for each successfully improved category.
            changes_made = []
            if shutil.which("black"):
                changes_made.append("Formatted with Black")
            if shutil.which("isort"):
                changes_made.append("Formatted with isort")
            if llm_success:
                changes_made.append(f"Applied LLM improvements ({llm_optimization_level})")
            if tests_generated:
                changes_made.append("Generated/updated tests")


            if changes_made:
                body_lines.append("Changes made:")
                for change in changes_made:
                     body_lines.append(f"* {change}")
            else:
                body_lines.append("No changes made.")

            # 3. Combine Summary and Body
            final_commit_message = summary
            if body_lines:
                final_commit_message += "\n\n" + "\n".join(body_lines)


        create_commit(repo_obj, file_path, final_commit_message, test_results)

    if not dry_run and not local_commit:
        console.print("[blue]Pull Request creation phase...[/blue]")
        create_pull_request(
            repo_url=repo,
            token=token,
            base_branch=branch,
            head_branch=new_branch_name,  # Pass the branch name. We'll namespace it inside.
            commit_message=final_commit_message,
            analysis_results=analysis_results,
            test_results=test_results,
            file_path=file_path,
            optimization_level=llm_optimization_level,
            test_framework=test_framework,
            min_coverage=min_coverage,
            coverage_fail_action=coverage_fail_action,
            repo_path = temp_dir,
            debug=debug
        )
    elif local_commit:
        console.print("[yellow]Local commit performed: no Pull Request created.[/yellow]")
    else:
        console.print("[yellow]Dry run performed: no changes made to the remote files.[/yellow]")
        restore_backup(file_path, file_path + ".bak")

    if not debug:
        shutil.rmtree(temp_dir)

if __debug__:  # pragma: no cover
    main()  # pragma: no cover
