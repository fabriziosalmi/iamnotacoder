import git
import os
import tempfile
import subprocess
import toml
import click
from openai import OpenAI
import datetime
import shutil
from rich.console import Console
from rich.progress import track
import json
from github import Github
import hashlib
import time
import ast
from typing import List, Dict, Tuple, Any

console = Console()

def run_command(command: str, cwd: str = None, shell: bool = True) -> Tuple[str, str, int]:
    """Executes a shell command and returns stdout, stderr, and return code."""
    try:
        result = subprocess.run(
            command, shell=shell, capture_output=True, text=True, cwd=cwd, check=True
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode
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
    """Creates a backup copy of the file."""
    backup_path = file_path + ".bak"
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
    """Clones the repository into a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    auth_repo_url = repo_url.replace("https://", f"https://{token}@")

    try:
        repo = git.Repo.clone_from(auth_repo_url, temp_dir)
        console.print(f"[green]Repository cloned to:[/green] {temp_dir}")
        return repo, temp_dir
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error cloning repository:[/red] {e}")
        exit(1)

def checkout_branch(repo: git.Repo, branch_name: str) -> None:
    """Checks out the specified branch."""
    try:
        repo.git.checkout(branch_name)
        console.print(f"[green]Checked out branch:[/green] {branch_name}")
    except git.exc.GitCommandError:
        try:
            console.print(f"[yellow]Attempting to fetch branch {branch_name}[/yellow]")
            repo.git.fetch("origin", branch_name)
            repo.git.checkout(branch_name)
            console.print(f"[green]Checked out branch:[/green] {branch_name}")
        except git.exc.GitCommandError as e:
            console.print(f"[red]Error checking out branch:[/red] {e}")
            exit(1)

def create_branch(repo: git.Repo, file_name: str) -> str:
    """Creates a new branch for the changes."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sanitized_file_name = "".join(c if c.isalnum() else "_" for c in file_name)
    branch_name = f"improvement-{sanitized_file_name}-{timestamp}"

    try:
        repo.git.checkout("-b", branch_name)
        console.print(f"[green]Created branch:[/green] {branch_name}")
        return branch_name
    except git.exc.GitCommandError as e:
        console.print(f"[red]Error creating branch:[/red] {e}")
        exit(1)

def analyze_project(repo_path: str, file_path: str, tools: List[str], exclude_tools: List[str], cache_dir: str = None) -> Dict[str, Dict[str, Any]]:
    """Performs static analysis, handling caching."""
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
    for tool in tools:
        if tool in exclude_tools:
            continue

        if tool == "pylint":
            command = f"pylint {file_path}"
        elif tool == "flake8":
            command = f"flake8 {file_path}"
        elif tool == "black":
            command = f"black --check --diff {file_path}"
        elif tool == "isort":
            command = f"isort --check-only --diff {file_path}"
        elif tool == "mypy":
            command = f"mypy {file_path}"
        else:
            console.print(f"[yellow]Unknown analysis tool: {tool}[/yellow]")
            continue

        stdout, stderr, returncode = run_command(command, cwd=repo_path)
        results[tool] = {
            "output": stdout,
            "errors": stderr,
            "returncode": returncode,
        }

    if cache_file:
        try:
            with open(cache_file, "w") as f:
                json.dump(results, f)
            console.print(f"[blue]Static analysis results saved to cache.[/blue]")
        except Exception as e:
            console.print(f"[yellow]Error saving to cache.[/yellow]")

    return results

def improve_file(file_path: str, client: OpenAI, llm_model: str, llm_temperature: float, categories: List[str], custom_prompt_dir: str, analysis_results: Dict[str, Dict[str, Any]], debug: bool = False) -> Tuple[str, bool]:
    """Improves the file using the LLM, iteratively by category."""
    backup_path = create_backup(file_path)
    if not backup_path:
        console.print("[red]Failed to create backup. Aborting.[/red]")
        exit(1)

    run_command(f"black {file_path}", cwd=os.path.dirname(file_path))
    run_command(f"isort {file_path}", cwd=os.path.dirname(file_path))

    with open(file_path, "r") as f:
        improved_code = f.read()

    for category in categories:
        console.print(f"[blue]Improving category: {category}[/blue]")
        prompt_file = os.path.join(custom_prompt_dir, f"prompt_{category}.txt")
        if not os.path.exists(prompt_file):
            console.print(f"[red]Prompt file not found: {prompt_file}. Skipping category.[/red]")
            continue

        try:
            with open(prompt_file, "r") as f:
                prompt_template = f.read()
                prompt = prompt_template.replace("{code}", improved_code)
        except Exception as e:
            console.print(f"[red]Error loading custom prompt: {e}[/red]")
            continue

        if debug:
            console.print(f"[debug]LLM prompt for {category}:\n{prompt}")

        max_retries = 3
        max_syntax_retries = 3
        success = False
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=llm_temperature,
                    max_tokens=4096
                )
                improved_code = response.choices[0].message.content.strip()
                improved_code = improved_code.replace("```python", "").replace("```", "").strip()

                syntax_errors = 0
                while syntax_errors < max_syntax_retries:
                    try:
                        ast.parse(improved_code)
                        success = True
                        break
                    except SyntaxError as e:
                        console.print(f"[yellow]Syntax error in generated code (attempt {syntax_errors + 1}), retrying...[/yellow]")
                        syntax_errors += 1
                        response = client.chat.completions.create(
                            model=llm_model,
                            messages=[
                                {"role": "system", "content": "You are a helpful coding assistant that improves code quality."},
                                {"role": "user", "content": f"{prompt}\n\nFix the following code that has syntax errors:\n\n```python\n{improved_code}\n```"}
                            ],
                            temperature=llm_temperature,
                            max_tokens=4096
                        )
                        improved_code = response.choices[0].message.content.strip()
                        improved_code = improved_code.replace("```python", "").replace("```", "").strip()

                if syntax_errors == max_syntax_retries:
                    console.print(f"[red]Max syntax correction attempts reached for {category}. Skipping.[/red]")
                    break

                with open(file_path, "w") as f:
                    f.write(improved_code)
                success = True
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    console.print(f"[yellow]Attempt {attempt + 1} for {category} failed, retrying... Error: {e}[/yellow]")
                    time.sleep(2)
                else:
                    console.print(f"[red]Error during LLM call for {category} after {max_retries} attempts: {e}[/red]")
                    success = False
                    break

        if not success:
            restore_backup(backup_path, file_path)
            console.print(f"[yellow]Restoring backup due to failure in {category} improvements.[/yellow]")

    return improved_code, success

def fix_tests(generated_tests: str, file_base_name: str) -> Tuple[str, bool]:
    """Analyzes generated test code and creates a corrective prompt if needed."""
    try:
        ast.parse(generated_tests)
        return generated_tests, False
    except SyntaxError as e:
        console.print(f"[yellow]Syntax error in test generation: {e}[/yellow]")
        return f"Syntax error in generated tests: {e}. Please regenerate.", True

def generate_tests(file_path: str, client: OpenAI, llm_model: str, llm_temperature: float, test_framework: str, custom_prompt_dir: str) -> str:
    """Generates tests using the LLM, with iterative correction and fallback."""
    with open(file_path, "r") as f:
        code = f.read()

    file_base_name = os.path.basename(file_path).split(".")[0]

    prompt = f"""Generate pytest unit tests for the following Python code (file: `{file_base_name}.py`):

```python
{code}
```

Return ONLY the Python code for the tests. No introductory text.
"""

    max_retries = 3
    max_corrections = 3
    generated_tests = ""
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful coding assistant that generates unit tests."},
                    {"role": "user", "content": prompt}
                ],
                temperature=llm_temperature,
                max_tokens=4096
            )
            generated_tests = response.choices[0].message.content.strip()
            generated_tests = generated_tests.replace("```python", "").replace("```", "").strip()

            corrections = 0
            while corrections < max_corrections:
                new_prompt, errors_found = fix_tests(generated_tests, file_base_name)
                if not errors_found:
                    break
                console.print("[yellow]Found errors in generated tests, attempting correction...[/yellow]")
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful coding assistant that generates unit tests."},
                        {"role": "user", "content": new_prompt}
                    ],
                    temperature=llm_temperature,
                    max_tokens=4096
                )
                generated_tests = response.choices[0].message.content.strip()
                generated_tests = generated_tests.replace("```python", "").replace("```", "").strip()
                corrections += 1
            break

        except Exception as e:
            if _ < max_retries - 1:
                console.print(f"[yellow]Attempt {_ + 1} failed, retrying... Error: {e}[/yellow]")
                time.sleep(2)
            else:
                console.print(f"[red]Error during test generation after {max_retries} attempts:[/red] {e}")
                return ""

    tests_dir = os.path.join(os.path.dirname(file_path), "..", "tests")
    os.makedirs(tests_dir, exist_ok=True)
    test_file_name = "test_" + os.path.basename(file_path)
    test_file_path = os.path.join(tests_dir, test_file_name)

    try:
        ast.parse(generated_tests)
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(generated_tests)
        console.print(f"[green]Test file written to: {test_file_path}[/green]")
        return generated_tests
    except SyntaxError as e:
        console.print(f"[red]Syntax error in generated tests after corrections: {e}[/red]")
        console.print(f"[debug] Generated test code:\n{generated_tests}")
        return ""

def run_tests(repo_path: str, original_file_path: str, test_framework: str, min_coverage: float, coverage_fail_action: str) -> Dict[str, Any]:
    """Runs tests, measures coverage (optional), and handles results."""
    test_results = {}

    if test_framework == "pytest":
        command = ["pytest", "-v"]

        stdout, stderr, returncode = run_command(" ".join(command), cwd=repo_path, shell=True)
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

    elif test_framework == "unittest":
        tests_dir = os.path.join(repo_path, "tests")
        command = f"coverage run -m unittest discover -s {tests_dir} -p 'test_*.py' -v"
        stdout, stderr, returncode = run_command(command, cwd=repo_path)
        if returncode == 0:
            console.print("[green]All tests passed.[/green]")
        else:
            console.print("[red]Some tests failed.[/red]")
        run_command("coverage report -m", cwd=repo_path)
        run_command("coverage xml", cwd=repo_path)
        run_command("coverage html", cwd=repo_path)
        test_results = {
            "output": stdout,
            "errors": stderr,
            "returncode": returncode,
        }
        return test_results
    else:
        console.print(f"[yellow]Unsupported test framework: {test_framework}[/yellow]")
        return {"output": "", "errors": f"Unsupported framework: {test_framework}", "returncode": 1}

def create_commit(repo: git.Repo, file_path: str, commit_message: str, test_results: Dict[str, Any]) -> None:
    """Creates a commit with the changes."""
    if not commit_message:
        commit_message = "Refactor: Automatic improvements"
    try:
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

def create_pull_request(repo_url: str, token: str, base_branch: str, head_branch: str, commit_message: str, analysis_results: Dict[str, Dict[str, Any]], test_results: Dict[str, Any], file_path: str, optimization_level: str, test_framework: str, min_coverage: float) -> None:
    """Creates a Pull Request on GitHub."""
    try:
        g = Github(token)
        repo_name = repo_url.replace("https://github.com/", "")
        repo = g.get_repo(repo_name)

        body = f"## Pull Request: Improvements to {os.path.basename(file_path)}\n\n"
        body += "Changes made:\n\n"
        body += "* Code formatting with Black and isort.\n"
        body += f"* Improvements suggested by LLM (level: {optimization_level}).\n"
        if test_results and test_results.get('returncode', 1) == 0:
            body += "* Added/Updated unit tests.\n"

        if analysis_results:
            table_text = "| Tool | Result |\n| --- | --- |\n"
            for tool, result in analysis_results.items():
                outcome = "[green]OK[/green]" if result['returncode'] == 0 else f"[red]Errors/Warnings ({len(result.get('output', '').splitlines())})[/red]"
                table_text += f"| {tool} | {outcome} |\n"
            body += "\n" + table_text + "\n"

        if test_results:
            test_outcome = "[green]Passed[/green]" if test_results['returncode'] == 0 else f"[red]Failed[/red]"
            body += f"\n| Tests | {test_outcome} |\n"

        if test_results and "TOTAL" in test_results.get('output', ''):
            for line in test_results['output'].splitlines():
                if line.lstrip().startswith("TOTAL"):
                    try:
                        coverage_percentage = float(line.split()[-1].rstrip("%"))
                        body += f"\n**Code Coverage:** {coverage_percentage:.2f}%\n"
                        if min_coverage and coverage_percentage < min_coverage:
                            body += f"\n**[WARNING] Coverage is below the minimum threshold! ({min_coverage}%)**\n"
                    except (ValueError, IndexError):
                        pass

        if test_results and test_results['returncode'] != 0:
            body += "\n**[WARNING] Some tests failed!  Check the CI results for details.**\n"

        body += "\n---\n\nTo run tests manually:\n\n```bash\n"
        if test_framework == 'pytest':
            rel_path = os.path.dirname(os.path.relpath(file_path, os.path.dirname(repo.git_dir)))
            body += f"pytest -v --cov={rel_path} --cov-report term-missing\n"
        elif test_framework == 'unittest':
            body += "python -m unittest discover\n"
        body += "```"

        pr = repo.create_pull(
            title=commit_message if commit_message else "Refactor: Automatic improvements",
            body=body,
            head=head_branch,
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
@click.option("--llm-model", "-m", default="hermes-3-llama-3.1-8b", help="LLM model to use.")
@click.option("--llm-temperature", "-temp", type=float, default=0.5, help="Temperature for the LLM.")
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
@click.option("--config", default=None, type=click.Path(exists=True), help="Configuration file", callback=get_inputs, is_eager=True, expose_value=False)
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

    if api_base:
      client = OpenAI(api_key="dummy", base_url=api_base, timeout=60.0)
    else:
      client = OpenAI(api_key=api_key, timeout=60.0)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    repo_obj, temp_dir = clone_repository(repo, token)
    file_path = os.path.join(temp_dir, file)
    original_file_path = os.path.join(repo, file)
    checkout_branch(repo_obj, branch)

    categories = [c.strip() for c in categories.split(",")]

    if not no_dynamic_analysis:
        analysis_results = analyze_project(
            temp_dir, file_path, tools.split(","), exclude_tools.split(","), cache_dir
        )
        console.print("[blue]Test generation phase...[/blue]")
        for _ in track(range(100), description="[cyan]Generating tests...[/cyan]"):
            time.sleep(0.005 if fast else 0.015)
        generated_tests = generate_tests(
            file_path, client, llm_model, llm_temperature, test_framework, llm_custom_prompt
        )
        console.print("[blue]Test execution phase...[/blue]")
        for _ in track(range(100), description="[cyan]Running tests...[/cyan]"):
            time.sleep(0.005 if fast else 0.03)
        test_results = run_tests(temp_dir, original_file_path, test_framework, min_coverage, coverage_fail_action)
    else:
        test_results = None
        analysis_results = {}

    new_branch_name = create_branch(repo_obj, file)

    console.print("[blue]File improvement phase...[/blue]")
    for _ in track(range(100), description="[cyan]Improving...[/cyan]"):
        time.sleep(0.005 if fast else 0.02)
    improved_code, llm_success = improve_file(
        file_path, client, llm_model, llm_temperature, categories, llm_custom_prompt, analysis_results, debug
    )

    if not dry_run:
        console.print("[blue]Commit phase...[/blue]")
        commit_message_content = (
            "Improvements:\n- Code formatting with Black and isort\n- LLM-driven optimization\n- Syntax validation and test updates\n"
            if llm_success else
            "Improvements:\n- Code formatting with Black and isort\n- Syntax validation and test updates\n- LLM-driven optimization skipped due to LLM failure\n"
        )
        commit_message = commit_message or f"Refactor: Automatic improvements\n\n{commit_message_content}"

        create_commit(repo_obj, file_path, commit_message, test_results)

    if not dry_run and not local_commit:
        console.print("[blue]Pull Request creation phase...[/blue]")
        create_pull_request(
            repo, token, branch, new_branch_name, commit_message, analysis_results,
            test_results, file_path, llm_optimization_level, test_framework, min_coverage
        )
    elif local_commit:
        console.print("[yellow]Local commit performed: no Pull Request created.[/yellow]")
    else:
        console.print("[yellow]Dry run performed: no changes made to the remote files.[/yellow]")
        restore_backup(file_path, file_path + ".bak")

    if not debug:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
