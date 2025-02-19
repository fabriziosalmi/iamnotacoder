import os
import subprocess
import toml
import click
from openai import OpenAI, Timeout, APIConnectionError, AuthenticationError
import datetime
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn
from rich.table import Table
from rich.live import Live
import json
import hashlib
import time
import logging
from rich.logging import RichHandler
import sys
from typing import List, Dict, Tuple, Any, Optional
import re

console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])

# Constants
DEFAULT_LLM_MODEL = "local-model"
DEFAULT_LLM_TEMPERATURE = 0.2
MAX_LLM_RETRIES = 3
MAX_ITERATIONS = 3
OPENAI_TIMEOUT = 120.0
CONFIG_ENCODING = "utf-8"

# --- Helper Functions ---
def run_command(command: List[str], cwd: Optional[str] = None) -> Tuple[str, str, int]:
    """Executes a shell command with timeout and error handling."""
    cmd_str = " ".join(command)
    try:
        process = subprocess.run(command, capture_output=True, text=True, cwd=cwd, check=False, timeout=300)
        return process.stdout, process.stderr, process.returncode
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logging.error(f"Error running command `{cmd_str}`: {e}")
        return "", str(e), 1

def load_config(config_file: str) -> Dict:
    """Loads configuration from a TOML file."""
    try:
        with open(config_file, "r", encoding=CONFIG_ENCODING) as f:
            return toml.load(f)
    except (FileNotFoundError, toml.TomlDecodeError, ValueError) as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        sys.exit(1)

def get_cli_config_priority(ctx: click.Context, param: click.Parameter, value: Any) -> Dict:
    config = {} if not value else load_config(value)
    config.update({k: v for k, v in ctx.params.items() if v is not None})
    ctx.default_map = config
    return config

def extract_code(response_text: str, start_delimiter: str, end_delimiter: str) -> str:
    """Extracts code between specified delimiters and removes Markdown code block syntax."""
    try:
        start_index = response_text.index(start_delimiter) + len(start_delimiter)
        end_index = response_text.index(end_delimiter)
        extracted_code = response_text[start_index:end_index].strip()

        # Remove leading/trailing ``` and language specifier (e.g., ```python)
        extracted_code = re.sub(r"^\s*```\w*\s*", "", extracted_code)  # Start
        extracted_code = re.sub(r"\s*```\s*$", "", extracted_code)  # End
        return extracted_code.strip()
    except ValueError:
        return ""

def clean_code(code: str) -> str:
    return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", code).strip()

# --- Project Context ---
class ProjectContext:
    """Holds the shared state of the project."""
    def __init__(self, app_description: str):
        self.app_description = app_description
        self.refined_description: Optional[str] = None
        self.plan: Optional[str] = None
        self.generated_code: Dict[str, str] = {}
        self.feedback: List[str] = []
        self.test_results: Dict[str, str] = {}
        self.deployment_results: Dict[str, Dict[str, Any]] = {}


# --- LLM Actor Classes ---
class LLMActor:
    def __init__(self, client: OpenAI, llm_model: str, llm_temperature: float, prompt_dir: str, role: str):
        self.client = client
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.prompt_dir = prompt_dir
        self.role = role
        self.prompt_cache: Dict[str, str] = {}

    def _get_prompt(self, prompt_name: str, context: ProjectContext, replacements: Dict[str, str] = {}) -> Optional[str]:
        """Loads and formats a prompt, dynamically adjusting it based on context."""
        prompt_file = os.path.join(self.prompt_dir, f"prompt_{prompt_name}.txt")
        # Create a unique cache key based on prompt name, replacements, and context
        combined_string = f"{prompt_name}_{str(replacements)}_{str(context.app_description)}_{str(context.refined_description)}_{str(context.plan)}_{str(context.generated_code)}_{str(context.test_results)}".encode()
        cache_key = hashlib.md5(combined_string).hexdigest()


        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]

        if not os.path.exists(prompt_file):
            console.print(f"[red]Prompt file not found: {prompt_file}[/red]")
            return None

        try:
            with open(prompt_file, "r", encoding=CONFIG_ENCODING) as f:
                prompt_template = f.read()

            # Dynamic Replacements (Context-Aware)
            replacements["app_description"] = context.app_description
            replacements["initial_idea"] = context.app_description
            replacements["refined_description"] = context.refined_description or ""
            replacements["plan"] = context.plan or ""
            replacements["backend"] = context.generated_code.get("backend", "")
            replacements["frontend"] = context.generated_code.get("frontend", "")
            replacements["database_schema"] = context.generated_code.get("database_schema", "")
            replacements["data_samples"] = context.generated_code.get("data_samples", "")
            replacements["test_results"] = context.test_results.get(self.role, "")

            formatted_prompt = prompt_template.format(**replacements)
            self.prompt_cache[cache_key] = formatted_prompt
            return formatted_prompt

        except KeyError as e:
            console.print(f"[red]KeyError in _get_prompt: {e}[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Error reading or formatting prompt file: {e}[/red]")
            return None

    def query_llm(self, prompt: str, system_message: str = "You are a helpful assistant.") -> Optional[str]:
        """Queries the LLM with retry logic and returns the response."""
        if not prompt:
            return None

        for attempt in range(MAX_LLM_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    timeout=OPENAI_TIMEOUT
                )
                return response.choices[0].message.content.strip()
            except (Timeout, APIConnectionError, AuthenticationError) as e:
                if attempt == MAX_LLM_RETRIES - 1:
                    console.print(f"[red]LLM query failed after {MAX_LLM_RETRIES} attempts: {e}[/red]")
                    return None
                else:
                    console.print(f"[yellow]LLM query failed (attempt {attempt + 1}/{MAX_LLM_RETRIES}): {e}. Retrying...[/yellow]")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                console.print(f"[red]Unexpected error during LLM query: {e}[/red]")
                return None
        return None

class CreativeAssistant(LLMActor):
    def generate_description(self, initial_idea: str) -> Optional[str]:
        context = ProjectContext(initial_idea)
        prompt = self._get_prompt("description", context)
        return self.query_llm(prompt) if prompt else None


class BackendDeveloper(LLMActor):
    def create_backend(self, context: ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("backend", context)
        return self.query_llm(prompt) if prompt else None
    def refine_backend(self, context:ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("backend_refine", context)
        return self.query_llm(prompt) if prompt else None

class FrontendDeveloper(LLMActor):
    def create_frontend(self, context: ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("frontend", context)
        return self.query_llm(prompt) if prompt else None
    def refine_frontend(self, context:ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("frontend_refine", context)
        return self.query_llm(prompt) if prompt else None

class DatabaseDeveloper(LLMActor):
    def create_database_schema(self, context: ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("database_schema", context)
        return self.query_llm(prompt) if prompt else None
    def refine_database_schema(self, context:ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("database_schema_refine", context)
        return self.query_llm(prompt) if prompt else None

class DataSampleGenerator(LLMActor):
    def generate_data_samples(self, context: ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("data_samples", context)
        return self.query_llm(prompt, system_message="You are a helpful assistant generating JSON data samples.") if prompt else None

class SecurityDeveloper(LLMActor):
    def review_code(self, code: str, component_name:str) -> Optional[str]:
        prompt = self._get_prompt("security_review", ProjectContext(""), {"code": code, "component_name": component_name})
        return self.query_llm(prompt) if prompt else None
    def add_security_measures(self, code: str, vulnerabilities: str, component_name: str) -> Optional[str]:
        prompt = self._get_prompt("add_security", ProjectContext(""), {"code": code, "vulnerabilities": vulnerabilities, "component_name": component_name})
        return self.query_llm(prompt) if prompt else None

class ProjectManager(LLMActor):
    def create_plan(self, context: ProjectContext) -> Optional[str]:
        prompt = self._get_prompt("plan", context)
        return self.query_llm(prompt) if prompt else None

    def consolidate_feedback(self, feedback_list: List[str]) -> Optional[str]:
        prompt = self._get_prompt("consolidate", ProjectContext(""), {"feedback_list": "\n".join(feedback_list)})
        return self.query_llm(prompt) if prompt else None

    def _determine_required_agents(self, description: str, data_samples_requested: bool) -> List[type]:
        agents = [FrontendDeveloper]  # Always include Frontend
        description_lower = description.lower()

        # Explicitly check for keywords indicating backend/database needs
        if any(keyword in description_lower for keyword in ["rest api", "server-side", "database connection", "sql", "data persistence"]):
            agents.append(BackendDeveloper)
        if any(keyword in description_lower for keyword in ["sql", "database schema", "relational database", "nosql"]):
            agents.append(DatabaseDeveloper)
        if data_samples_requested:
            agents.append(DataSampleGenerator)
        return agents


class TestDeveloper(LLMActor):
    def create_tests(self, code: str, component_name: str) -> Optional[str]:
        prompt = self._get_prompt("unit_tests", ProjectContext(""), {"code": code, "component_name": component_name})
        return self.query_llm(prompt) if prompt else None

class Deployer(LLMActor):
    def run_in_sandbox(self, repo_path: str) -> Dict[str, Dict[str, Any]]:
        """Runs tests and checks for the existence of frontend files."""
        results = {}

        # Backend Tests
        backend_test_path = os.path.join(repo_path, "test_backend.py")
        if os.path.exists(backend_test_path):
            stdout, stderr, returncode = run_command(["pytest", backend_test_path], cwd=repo_path)
            results["backend"] = {
                "success": returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
            }

        # Frontend (existence check)
        frontend_path = os.path.join(repo_path, "frontend.py")
        if os.path.exists(frontend_path):
            results["frontend"] = {"success": True, "stdout": "", "stderr": ""}

        # Database Tests
        database_test_path = os.path.join(repo_path, "test_database.py")
        if os.path.exists(database_test_path):
            stdout, stderr, returncode = run_command(["pytest", database_test_path], cwd=repo_path)
            results["database"] = {
                "success": returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
            }

        return results

# --- Main Application Logic ---

def create_application(
    app_description: str,
    client: OpenAI,
    llm_model: str,
    llm_temperature: float,
    prompt_dir: str,
    repo_path: str,
    debug: bool = False,
    data_samples_requested: bool = False,
    disable_security_checks: bool = False,
    disable_tests: bool = False,
) -> None:
    """Orchestrates the LLM actors with iterative refinement."""

    context = ProjectContext(app_description)

    creative_assistant = CreativeAssistant(client, llm_model, llm_temperature, prompt_dir, role="creative_assistant")
    project_manager = ProjectManager(client, llm_model, llm_temperature, prompt_dir, role="project_manager")
    security_dev = SecurityDeveloper(client, llm_model, llm_temperature, prompt_dir, role="security_developer")
    test_dev = TestDeveloper(client, llm_model, llm_temperature, prompt_dir, role="test_developer")
    deployer = Deployer(client, llm_model, llm_temperature, prompt_dir, role="deployer")

    with Live(console=console, refresh_per_second=12) as live:
        # 1. Refine description
        live.update(console.render_str("[blue]Refining application description...[/]"))
        context.refined_description = creative_assistant.generate_description(app_description)
        if not context.refined_description:
            console.print("[red]Failed to refine description. Exiting.[/red]")
            return

        # 2. Create plan
        live.update(console.render_str("[blue]Creating development plan...[/]"))
        context.plan = project_manager.create_plan(context)
        if not context.plan:
            console.print("[red]Failed to create plan. Exiting.[/red]")
            return

        # 3. Determine Required Agents
        required_agents = project_manager._determine_required_agents(context.refined_description, data_samples_requested)
        live.update(console.render_str(f"[blue]Required agents: {', '.join([agent.__name__ for agent in required_agents])}[/]"))

        # 4. Agent Execution and Iterative Refinement
        agent_instances = {
            agent_class: agent_class(client, llm_model, llm_temperature, prompt_dir, role=agent_class.__name__.lower())
            for agent_class in required_agents
        }

        for agent_class in required_agents:
            agent_name = agent_class.__name__
            agent = agent_instances[agent_class]
            live.update(console.render_str(f"[blue]Creating {agent_name}...[/]"))
            component_name = agent_name.replace("Developer", "").lower()

            # --- Check if iterations are needed ---
            iterations_needed = not (disable_security_checks and disable_tests)

            num_iterations = MAX_ITERATIONS if iterations_needed else 1 # Only one iteration if no checks

            for iteration in range(num_iterations):
                live.update(console.render_str(f"[blue]{agent_name}: Iteration {iteration + 1}/{num_iterations}[/]"))

                # --- Code Generation ---
                if iteration == 0:
                    if agent_name == "BackendDeveloper":
                        code = agent.create_backend(context)
                    elif agent_name == "FrontendDeveloper":
                        code = agent.create_frontend(context)
                    elif agent_name == "DatabaseDeveloper":
                        code = agent.create_database_schema(context)
                    elif agent_name == "DataSampleGenerator":
                        code = agent.generate_data_samples(context)
                    else:
                        code = None
                else:
                    if agent_name == "BackendDeveloper":
                        code = agent.refine_backend(context)
                    elif agent_name == "FrontendDeveloper":
                        code = agent.refine_frontend(context)
                    elif agent_name == "DatabaseDeveloper":
                        code = agent.refine_database_schema(context)
                    else:
                        code = None


                if not code:
                    console.print(f"[red]Failed to create {component_name} code. Exiting.[/red]")
                    return

                # --- Stricter Code Extraction ---
                if agent_name in ["BackendDeveloper", "FrontendDeveloper", "DatabaseDeveloper"]:
                    code = extract_code(code, "<--CODE_START-->", "<--CODE_END-->")
                elif agent_name == "DataSampleGenerator":
                    code = extract_code(code, "<--JSON_START-->", "<--JSON_END-->")  # or different delimiters if needed
                context.generated_code[component_name] = code


                # --- Security Review (Conditional) ---
                if not disable_security_checks and code:
                    vulnerabilities = security_dev.review_code(code, component_name)
                    if vulnerabilities and "no vulnerabilities found" not in vulnerabilities.lower():
                        console.print(f"[yellow]Identified Vulnerabilities in {component_name}:\n{vulnerabilities}[/yellow]")
                        context.feedback.append(f"{agent_name} Security Feedback: {vulnerabilities}")
                        # Apply security fixes
                        fixed_code = security_dev.add_security_measures(code, vulnerabilities, component_name)
                        if fixed_code:
                            code = extract_code(fixed_code, "<--CODE_START-->", "<--CODE_END-->")
                            context.generated_code[component_name] = code  # Update with fixed code

                # --- Test Creation and Execution (Conditional) ---
                if not disable_tests and code:
                    tests = test_dev.create_tests(code, component_name)
                    if tests:
                        tests = extract_code(tests, "<--TEST_START-->", "<--TEST_END-->")
                        test_file_path = os.path.join(repo_path, f"test_{component_name}.py")
                        with open(test_file_path, "w", encoding=CONFIG_ENCODING) as f:
                            f.write(tests)

                        # Run Tests and Store Results
                        stdout, stderr, returncode = run_command(["pytest", test_file_path], cwd=repo_path)
                        context.test_results[agent.role] = f"Test Results (stdout):\n{stdout}\nTest Results (stderr):\n{stderr}"

                        if returncode == 0:
                            console.print(f"[green]{agent_name}: Tests passed![/green]")
                            break  # Exit refinement loop if tests pass
                        else:
                            console.print(f"[red]{agent_name}: Tests failed. Refining...[/red]")
                            context.feedback.append(f"{agent_name} Test Feedback: Tests Failed")
                    else:
                        console.print(f"[yellow]No tests generated for {component_name}.[/yellow]")

                else:
                    console.print(f"[yellow]Skipping tests for {component_name} as requested.[/yellow]")



            # Write the final code to file.
            if component_name != "data_samples":
                file_path = os.path.join(repo_path, f"{component_name}.py")
                with open(file_path, "w", encoding=CONFIG_ENCODING) as f:
                    f.write(context.generated_code[component_name])
                console.print(f"[green]{agent_name}: Code written to {file_path}[/green]")
            else:
                file_path = os.path.join(repo_path, f"{component_name}.json")
                with open(file_path, "w", encoding=CONFIG_ENCODING) as f:
                    f.write(context.generated_code[component_name])
                console.print(f"[green]{agent_name}: Code written to {file_path}[/green]")

        # Create requirements.txt
        requirements_path = os.path.join(repo_path, "requirements.txt")
        with open(requirements_path, "w", encoding=CONFIG_ENCODING) as f:
            f.write("streamlit\nopenai") # Basic requirements, add more as needed
        console.print(f"[green]requirements.txt created at {requirements_path}[/green]")

        # 5. Deployment (Sandboxed) - Still runs, but will be a no-op if no tests exist
        live.update(console.render_str("[blue]Running in sandbox...[/]"))
        context.deployment_results = deployer.run_in_sandbox(repo_path)
        for component, result in context.deployment_results.items():
            if result["success"]:
                console.print(f"[green]{component.capitalize()} deployment successful.[/green]")
            else:
                console.print(f"[red]{component.capitalize()} deployment failed: {result['stderr']}[/red]")

        # 6. Consolidate Feedback and Create README
        live.update(console.render_str("[blue]Consolidating feedback and creating README...[/]"))
        consolidated_feedback = project_manager.consolidate_feedback(context.feedback)
        readme_content = f"# {context.app_description}\n\n## Development Plan\n{context.plan}\n\n## Consolidated Feedback\n{consolidated_feedback or 'See individual agent outputs above.'}"
        readme_path = os.path.join(repo_path, "README.md")
        with open(readme_path, "w", encoding=CONFIG_ENCODING) as f:
            f.write(readme_content)
        console.print(f"[green]README.md created at {readme_path}[/green]")
        console.print(f"[green]Application creation completed. Files are in {repo_path}.[/green]")


@click.command()
@click.option("--app-description", "-d", required=True, help="Description of the app.")
@click.option("--llm-model", "-m", default=DEFAULT_LLM_MODEL, help="LLM model.")
@click.option("--llm-temperature", "-temp", type=float, default=DEFAULT_LLM_TEMPERATURE, help="LLM temperature.")
@click.option("--llm-custom-prompt", "-p", default=".", help="Path to custom prompts.")
@click.option("--debug", is_flag=True, help="Enable debug logging.")
@click.option("--config", default=None, type=click.Path(exists=False), callback=get_cli_config_priority, is_eager=True, expose_value=False, help="Path to TOML config.")
@click.option("--openai-api-base", default=None, help="Base URL for OpenAI API.")
@click.option("--openai-api-key", default=None, help="OpenAI API Key.")
@click.option("--data-samples", is_flag=True, help="Generate data samples.")
@click.option("--disable-security-checks", is_flag=True, help="Disable security checks.")
@click.option("--disable-tests", is_flag=True, help="Disable tests.")
def theteam_cli(
    app_description: str,
    llm_model: str,
    llm_temperature: float,
    llm_custom_prompt: str,
    debug: bool,
    openai_api_base: Optional[str],
    openai_api_key: Optional[str],
    data_samples: bool,
    disable_security_checks: bool,
    disable_tests: bool,
) -> None:
    """Creates an application from scratch using LLM actors."""
    ctx = click.get_current_context()
    config_values = ctx.default_map if ctx.default_map else {}
    api_base = openai_api_base or config_values.get("openai_api_base") or os.getenv("OPENAI_API_BASE")
    api_key = openai_api_key or config_values.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

    if debug:
        console.print("[yellow]Debug mode enabled.[/yellow]")

    if not api_base:
        if not api_key:
            console.print("[red]Error: OpenAI API key is required when not using a custom base URL.[/red]")
            sys.exit(1)
    else:
        if not api_key:
             api_key = "dummy_key"

    client = OpenAI(api_key=api_key, base_url=api_base, timeout=OPENAI_TIMEOUT)

    script_dir = os.path.abspath(os.path.dirname(__file__))
    prompt_dir = os.path.join(script_dir, "prompts")
    os.makedirs(prompt_dir, exist_ok=True)  # Create prompts directory

    project_path = os.path.join(script_dir, f"project_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(project_path, exist_ok=True)
    console.print(f"[blue]Creating application in directory: {project_path}[/blue]")

    create_application(app_description, client, llm_model, llm_temperature, prompt_dir, project_path, debug, data_samples, disable_security_checks, disable_tests)

if __name__ == "__main__":
    theteam_cli()