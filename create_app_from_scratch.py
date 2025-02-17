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
import hashlib
import time
import logging
from rich.logging import RichHandler
import sys
from typing import List, Dict, Tuple, Any, Optional
import re

console = Console()

# Configure logging with Rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Constants
DEFAULT_LLM_MODEL = "qwen2.5-coder-7b-instruct-mlx"  # Or any other suitable default
DEFAULT_LLM_TEMPERATURE = 0.2
MAX_LLM_RETRIES = 3
OPENAI_TIMEOUT = 120.0
CONFIG_ENCODING = "utf-8"
CACHE_ENCODING = "utf-8"

# --- Helper Functions (from the original, with modifications) ---

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



def extract_code_from_response(response_text: str) -> str:
    """Extracts code from LLM responses, handling Markdown code blocks and inline code."""
    code_blocks = re.findall(r"```(?:[a-zA-Z]+)?\n(.*?)\n```", response_text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    lines = response_text.strip().splitlines()
    cleaned_lines = []
    start_collecting = False
    for line in lines:
        line = line.strip()
        if not start_collecting:
            if line.startswith(("import ", "def ", "class ")) or re.match(r"^[a-zA-Z0-9_]+(\(.*\)| =.*):", line):
                start_collecting = True  # Heuristic: start when code-like lines appear
        if start_collecting:
             if line.lower().startswith("return only the"): # Stop at common LLM instructions
                break
             cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# --- LLM Actor Classes ---

class LLMActor:
    """Base class for LLM actors."""
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str, role: str):
        self.client = client
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.prompt_dir = prompt_dir
        self.role = role  # e.g., "backend_developer", "frontend_developer"

    def _get_prompt(self, prompt_name: str, replacements: Dict[str, str] = {}) -> str:
        """Loads and formats a prompt from a file."""
        prompt_file = os.path.join(self.prompt_dir, f"prompt_{prompt_name}.txt")
        if not os.path.exists(prompt_file):
            console.print(f"[red]Prompt file not found: {prompt_file}[/red]")
            return ""

        try:
            with open(prompt_file, "r", encoding=CONFIG_ENCODING) as f:
                prompt_template = f.read()
                for key, value in replacements.items():
                    prompt_template = prompt_template.replace(f"{{{key}}}", value)
                return prompt_template
        except Exception as e:
            console.print(f"[red]Error reading prompt file: {e}[/red]")
            logging.exception(f"Error reading prompt file: {prompt_file}")
            return ""


    def query_llm(self, prompt: str, system_message: str = "You are a helpful coding assistant.") -> str:
        """Queries the LLM and returns the response, handling retries."""
        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=4096,
                    timeout=OPENAI_TIMEOUT,
                )
                return extract_code_from_response(response.choices[0].message.content)
            except Timeout:
                console.print(f"[yellow]Timeout during LLM call (attempt {attempt+1}/{MAX_LLM_RETRIES}). Retrying...[/yellow]")
                logging.warning(f"Timeout during LLM call for {self.role}, attempt {attempt+1}/{MAX_LLM_RETRIES}")
            except Exception as e:
                console.print(f"[red]LLM query failed (attempt {attempt+1}/{MAX_LLM_RETRIES}): {e}. Retrying...[/red]")
                logging.error(f"LLM query failed for {self.role}, attempt {attempt+1}/{MAX_LLM_RETRIES}: {e}")
        console.print(f"[red]Failed to query LLM after {MAX_LLM_RETRIES} retries.[/red]")
        return ""


class BackendDeveloper(LLMActor):
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str):
        super().__init__(client, llm_model, llm_temperature, prompt_dir, "backend_developer")

    def create_backend(self, description: str, existing_code: str = "") -> str:
        prompt = self._get_prompt("backend", {"description": description, "existing_code": existing_code})
        if not prompt: return ""
        return self.query_llm(prompt)

class FrontendDeveloper(LLMActor):
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str):
        super().__init__(client, llm_model, llm_temperature, prompt_dir, "frontend_developer")

    def create_frontend(self, description: str, backend_code: str = "") -> str:
        prompt = self._get_prompt("frontend", {"description": description, "backend_code": backend_code})
        if not prompt: return ""

        return self.query_llm(prompt, system_message="You are a helpful coding assistant creating frontend code.")

class CreativeAssistant(LLMActor):
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str):
        super().__init__(client, llm_model, llm_temperature, prompt_dir, "creative_assistant")

    def generate_description(self, initial_idea: str) -> str:
        prompt = self._get_prompt("description", {"initial_idea": initial_idea})
        if not prompt: return ""
        return self.query_llm(prompt)
    def create_design_asset(self, asset_description: str) -> str:
        # Placeholder - In reality, this might involve image generation or external API calls.
        prompt = self._get_prompt("design_asset", {"asset_description": asset_description})
        if not prompt: return ""
        return self.query_llm(prompt, system_message="You generate descriptions of design assets.")


class SecurityDeveloper(LLMActor):
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str):
        super().__init__(client, llm_model, llm_temperature, prompt_dir, "security_developer")

    def review_code(self, code: str) -> str:
        prompt = self._get_prompt("security_review", {"code": code})
        if not prompt: return ""
        return self.query_llm(prompt, system_message="You are a security expert reviewing code for vulnerabilities.")
    def add_security_measures(self, code: str, vulnerabilities: str) -> str:
        prompt = self._get_prompt("add_security", {"code": code, "vulnerabilities": vulnerabilities})
        if not prompt: return ""
        return self.query_llm(prompt)

class ProjectManager(LLMActor):
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str):
        super().__init__(client, llm_model, llm_temperature, prompt_dir, "project_manager")

    def create_plan(self, description: str) -> str:
        prompt = self._get_prompt("plan", {"description": description})
        if not prompt: return ""
        return self.query_llm(prompt, system_message="You are a project manager creating a development plan.")

    def consolidate_feedback(self, feedback_list: List[str]) -> str:
        """Combines feedback from multiple actors."""
        prompt = self._get_prompt("consolidate", {"feedback_list": "\n".join(feedback_list)})
        return self.query_llm(prompt, system_message="You consolidate feedback and resolve conflicts.")



# --- Main Application Logic ---

def create_application(
    app_description: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    prompt_dir: str,
    repo_path: str,
    debug: bool = False
) -> None:
    """Orchestrates the LLM actors to build the application."""

    backend_dev = BackendDeveloper(client, llm_model, llm_temperature, prompt_dir)
    frontend_dev = FrontendDeveloper(client, llm_model, llm_temperature, prompt_dir)
    creative_assistant = CreativeAssistant(client, llm_model, llm_temperature, prompt_dir)
    security_dev = SecurityDeveloper(client, llm_model, llm_temperature, prompt_dir)
    project_manager = ProjectManager(client, llm_model, llm_temperature, prompt_dir)


    # 1. Refine the application description.
    console.print("[blue]Refining application description...[/blue]")
    refined_description = creative_assistant.generate_description(app_description)
    if not refined_description:
        console.print("[red]Failed to refine application description. Exiting.[/red]")
        return
    console.print(f"[green]Refined Description:\n{refined_description}[/green]")


    # 2. Create a development plan.
    console.print("[blue]Creating development plan...[/blue]")
    plan = project_manager.create_plan(refined_description)
    if not plan:
        console.print("[red]Failed to create a development plan. Exiting.[/red]")
        return
    console.print(f"[green]Development Plan:\n{plan}[/green]")

    # 3. Create backend code.
    console.print("[blue]Creating backend code...[/blue]")
    backend_code = backend_dev.create_backend(refined_description)
    if not backend_code:
        console.print("[red]Failed to create backend code. Exiting.[/red]")
        return

    backend_file_path = os.path.join(repo_path, "backend.py")
    with open(backend_file_path, "w", encoding=CONFIG_ENCODING) as f:
        f.write(backend_code)
    console.print(f"[green]Backend code written to {backend_file_path}[/green]")

    # 4. Create frontend code (if applicable, based on the plan).
    if "frontend" in plan.lower():  # Simple heuristic, improve as needed
        console.print("[blue]Creating frontend code...[/blue]")
        frontend_code = frontend_dev.create_frontend(refined_description, backend_code)
        if frontend_code:
            frontend_file_path = os.path.join(repo_path, "frontend.py")
            with open(frontend_file_path, "w", encoding=CONFIG_ENCODING) as f:
                f.write(frontend_code)
            console.print(f"[green]Frontend code written to {frontend_file_path}[/green]")
        else:
            console.print("[yellow]Frontend creation failed or not applicable.[/yellow]")

     # 5.  Security Review and Improvements.
    console.print("[blue]Performing security review...[/blue]")
    all_code = backend_code
    if 'frontend_code' in locals() and frontend_code:  # Check if frontend_code exists
      all_code += "\n\n" + frontend_code

    vulnerabilities = security_dev.review_code(all_code)
    console.print(f"[green]Identified Vulnerabilities:\n{vulnerabilities}[/green]")

    if vulnerabilities.strip() and vulnerabilities.strip().lower() != "no vulnerabilities found.": # Only improve if there are suggestions
        console.print("[blue]Applying security improvements...[/blue]")
        improved_backend_code = security_dev.add_security_measures(backend_code, vulnerabilities)

        if improved_backend_code:
          with open(backend_file_path, "w", encoding=CONFIG_ENCODING) as f:
            f.write(improved_backend_code)
          console.print(f"[green]Improved backend code written to {backend_file_path}[/green]")
          backend_code = improved_backend_code # Update the code to reflect the changes

        if 'frontend_code' in locals() and frontend_code:
          improved_frontend_code = security_dev.add_security_measures(frontend_code, vulnerabilities)
          if improved_frontend_code:
            with open(frontend_file_path, "w", encoding=CONFIG_ENCODING) as f:
                f.write(improved_frontend_code)
            console.print(f"[green]Improved frontend code written to {frontend_file_path}[/green]")
            frontend_code = improved_frontend_code


    # 6. Project Manager Consolidates and creates a report.
    console.print("[blue]Consolidating feedback and creating report...[/blue]")

    feedback = []
    if 'frontend_code' in locals() and frontend_code: #If frontend exists
      feedback.extend([
          f"Backend Developer: Created backend.py\n{backend_code}",
          f"Frontend Developer: Created frontend.py\n{frontend_code}",
          f"Creative Assistant: Refined description\n{refined_description}",
          f"Security Developer: Reviewed code. Findings:\n{vulnerabilities}",
      ])
    else: #If not frontend
      feedback.extend([
          f"Backend Developer: Created backend.py\n{backend_code}",
          f"Creative Assistant: Refined description\n{refined_description}",
          f"Security Developer: Reviewed code. Findings:\n{vulnerabilities}",
      ])

    consolidated_feedback = project_manager.consolidate_feedback(feedback)
    console.print(f"[green]Consolidated Feedback:\n{consolidated_feedback}[/green]")

    # 7. Create README.md
    readme_content = f"# {app_description}\n\n## Development Plan\n{plan}\n\n## Consolidated Feedback\n{consolidated_feedback}"
    readme_path = os.path.join(repo_path, "README.md")
    with open(readme_path, "w", encoding=CONFIG_ENCODING) as f:
        f.write(readme_content)
    console.print(f"[green]README.md created at {readme_path}[/green]")


@click.command()
@click.option("--app-description", "-d", required=True, help="Description of the application to create.")
@click.option("--llm-model", "-m", default=DEFAULT_LLM_MODEL, help="LLM model to use.")
@click.option("--llm-temperature", "-temp", type=float, default=DEFAULT_LLM_TEMPERATURE, help="Temperature for the LLM.")
@click.option("--llm-custom-prompt", "-p", default=".", help="Path to custom prompt directory.")
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--config", default=None, type=click.Path(exists=True), callback=get_cli_config_priority, is_eager=True, expose_value=False, help="Path to TOML config file.")
@click.option("--openai-api-base", default=None, help="Base URL for OpenAI API (e.g., for LMStudio).")

def theteam_cli(
    app_description: str,
    llm_model: str,
    llm_temperature: float,
    llm_custom_prompt: str,
    debug: bool,
    openai_api_base: Optional[str]

) -> None:
    """Creates an application from scratch using LLM actors."""

    ctx = click.get_current_context()
    config_values = ctx.default_map if ctx.default_map else {}
    api_base = config_values.get("openai_api_base", openai_api_base or os.getenv("OPENAI_API_BASE"))
    api_key = config_values.get("openai_api_key", os.getenv("OPENAI_API_KEY"))

    if debug:
        console.print("[yellow]Debug mode enabled.[/yellow]")
        console.print(f"[yellow]API Base: {api_base}[/yellow]")
        console.print(f"[yellow]API Key from env/config: {api_key is not None}[/yellow]")  # Mask API key
        console.print(f"[yellow]Effective Configuration: {config_values}[/yellow]")


    api_key_provided = api_key and api_key.lower() != "none"
    if not api_base and not api_key_provided:
        console.print(
            "[red]Error: OpenAI API key or base URL not found.\n"
            "Set OPENAI_API_KEY/OPENAI_API_BASE environment variables, or use --config or --openai-api-base/--openai-api-key.[/red]"
        )
        sys.exit(1)


    client = OpenAI(api_key=api_key, base_url=api_base, timeout=OPENAI_TIMEOUT) if api_base or api_key_provided else None


    # Create a temporary directory for the project.
    with tempfile.TemporaryDirectory() as temp_dir:
        console.print(f"[blue]Creating application in temporary directory: {temp_dir}[/blue]")
        create_application(app_description, client, llm_model, llm_temperature, llm_custom_prompt, temp_dir, debug)

        console.print("[green]Application creation completed. Files are in the temporary directory.[/green]")
        # Keep the temporary directory for inspection, do not delete automatically:
        console.print(f"[yellow]Temporary directory will not be deleted: {temp_dir}[/yellow]")

if __name__ == "__main__":
    theteam_cli()