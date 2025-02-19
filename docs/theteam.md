# TheTeam: AI-Powered Application Generator

`TheTeam` is a command-line tool that leverages Large Language Models (LLMs) to generate entire applications from scratch.  It simulates a team of specialized AI agents, each playing a distinct role (Frontend Developer, Backend Developer, Database Developer, etc.) to collaboratively build, test, and deploy your application.  The tool supports iterative refinement, security checks, and unit testing to produce robust and functional code.

## Features

*   **Multi-Agent Collaboration:** Employs a team of LLM-powered agents (Frontend, Backend, Database, Security, Project Manager, Tester, Deployer) to handle different aspects of application development.
*   **Iterative Refinement:**  Agents iteratively refine their output based on feedback, test results, and security reviews, improving code quality over multiple cycles.
*   **Code Extraction:**  Uses delimiters (`<--CODE_START-->`, `<--CODE_END-->`, `<--JSON_START-->`, `<--JSON_END-->`, `<--TEST_START-->`, `<--TEST_END-->`) to reliably extract generated code from LLM responses.
*   **Security Reviews:**  Includes a Security Developer agent that analyzes code for vulnerabilities and suggests fixes.
*   **Automated Testing:**  Generates and runs unit tests (using `pytest`) to ensure code functionality.  Test failures trigger refinement.
*   **Sandboxed Deployment:** Simulates a deployment environment to check for basic functionality and file existence (e.g., frontend files).
*   **Customizable Prompts:** Allows users to provide their own prompt files to fine-tune the behavior of the LLM agents.
*   **Configurable LLM:** Supports specifying the LLM model, temperature, and API base URL (for local models or custom deployments).
*   **TOML Configuration:**  Supports loading configuration settings from a TOML file.
*   **Rich CLI:** Uses the `rich` library for beautiful console output, progress bars, and logging.
*   **Caching:** Prompts are cached to avoid redundant calculations and improve performance.

## Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   `pip`

2.  **Install `TheTeam`:**

    ```bash
    pip install theteam
    ```

## Usage

```bash
theteam --app-description "Your application description" [options]
```

**Required Argument:**

*   `--app-description` or `-d`:  A concise description of the application you want to create.  Be as specific as possible, including details about desired functionality, data models, and user interactions.  For example:

    *   "A simple web app to track daily expenses.  Users should be able to add expenses with categories and dates, and view a summary of expenses by month."
    *   "A REST API for a to-do list application.  It should support creating, reading, updating, and deleting tasks.  Tasks should have a title, description, due date, and status (pending, in progress, completed)."

**Options:**

*   `--llm-model` or `-m`:  The LLM model to use (default: `local-model`).  You can specify any compatible OpenAI model or a local model endpoint.
*   `--llm-temperature` or `-temp`:  The LLM temperature (default: 0.2).  Higher values (e.g., 0.7) result in more creative but potentially less coherent output.  Lower values (e.g., 0.2) are generally preferred for code generation.
*   `--llm-custom-prompt` or `-p`:  Path to a directory containing custom prompt files (default: `.`, meaning the script's directory).  See the "Custom Prompts" section below.
*   `--debug`:  Enable debug logging.
*   `--config`: Path to a TOML configuration file. This file can contain any of the command-line options. Command-line options take precedence over configuration file values.
*   `--openai-api-base`:  Base URL for the OpenAI API (or a compatible endpoint).  If not provided, it defaults to the standard OpenAI API.
*   `--openai-api-key`:  Your OpenAI API key.  Required if `--openai-api-base` is not provided. Can also be set via the `OPENAI_API_KEY` environment variable.
*   `--data-samples`:  Generate sample data for the application (e.g., example JSON data).
*   `--disable-security-checks`:  Disable security vulnerability checks and fixes.
*   `--disable-tests`:  Disable unit test generation and execution.

**Example (using a local LLM with `ollama`):**
first run:
```
ollama run llama2
```

```bash
theteam -d "A simple web app that displays a list of tasks.  Tasks have a title and a due date." --llm-model "local-model" --openai-api-base "http://localhost:11434/v1" --data-samples --disable-tests
```

**Example (using OpenAI's API):**
first set enviroment vars
```
export OPENAI_API_KEY="sk-..."
```
then run:
```bash
theteam -d "A simple web app that displays a list of tasks.  Tasks have a title and a due date." --llm-model "gpt-3.5-turbo" --data-samples --disable-tests
```

## Configuration File (TOML)

You can create a TOML file (e.g., `config.toml`) to store your preferred settings:

```toml
# config.toml
app_description = "A web app to manage a recipe collection."
llm_model = "local-model"
llm_temperature = 0.3
openai_api_base = "http://localhost:11434/v1"
data_samples = true
disable_tests = true # Disable tests for faster iteration, enable later for robust code.

```

Then, run `TheTeam` with the `--config` option:

```bash
theteam --config config.toml
```
Any options provided directly on the command line will override the corresponding values in the configuration file. The `--app-description` is required, even if present in config.

## Project Structure

`TheTeam` creates a project directory with the following structure:

```
project_<timestamp>/
├── backend.py
├── frontend.py
├── database_schema.py  (if applicable)
├── data_samples.json (if --data-samples is used)
├── test_backend.py (if tests are enabled)
├── test_frontend.py (if tests are enabled)
├── test_database.py (if tests are enabled and a database is used)
├── requirements.txt
└── README.md
```

## Custom Prompts

The `prompts` directory (located in the same directory as the script) contains text files that define the prompts used by each LLM agent.  You can modify these prompts or create new ones to customize the behavior of the agents.

Each prompt file is named `prompt_<agent_role>.txt`.  For example:

*   `prompt_creative_assistant.txt`:  Prompt for the CreativeAssistant.
*   `prompt_backenddeveloper.txt`:  Prompt for the BackendDeveloper.
*   `prompt_frontenddeveloper.txt`:  Prompt for the FrontendDeveloper.
*   ...and so on.

**Prompt Formatting:**

Prompts are formatted using Python's string formatting syntax (e.g., `{app_description}`, `{plan}`). The available placeholders are:

*   `app_description`:  The initial application description provided by the user.
*   `initial_idea`: The same as `app_description`.
*   `refined_description`: The refined application description generated by the CreativeAssistant.
*   `plan`: The development plan created by the ProjectManager.
*   `backend`: The generated backend code.
*   `frontend`: The generated frontend code.
*   `database_schema`: The generated database schema.
*    `data_samples`: The generated sample data.
*   `test_results`:  The results of the unit tests for the specific agent's role.
*   `code`: The code to be analyzed (used in `security_review` and `unit_tests`).
*   `component_name`: The name of the component being analyzed (e.g., "backend", "frontend").
*   `vulnerabilities`:  The identified vulnerabilities (used in `add_security`).
*    `feedback_list`: (Used in Project Manager consolidate).

**Example Prompt (prompt_backenddeveloper.txt):**

```
You are a Backend Developer.  Your task is to create the backend code for a web application based on the following description and plan:

Description: {refined_description}

Plan: {plan}

Create the Python backend code.  Use a suitable framework (e.g., Flask, FastAPI) and provide clear instructions for running the application.  Include any necessary setup or configuration steps.

<--CODE_START-->
# Your generated backend code will go here
<--CODE_END-->
```

**Important:**

*   The `<--CODE_START-->` and `<--CODE_END-->` delimiters (or other delimiters like `<--JSON_START-->`, `<--JSON_END-->`, `<--TEST_START-->`, `<--TEST_END-->`) are *crucial* for code extraction.  Make sure they are present in your prompts where appropriate.
*  The prompts `security_review`, `add_security`, and `unit_tests` receive the `code` and the `component_name` to perform its tasks.

## How it Works (Internal Logic)

1.  **Initialization:**
    *   Loads configuration from command-line arguments and/or a TOML file.
    *   Initializes the OpenAI client.
    *   Creates a project directory.

2.  **Description Refinement:** The `CreativeAssistant` refines the initial user-provided application description, making it more detailed and specific.

3.  **Planning:** The `ProjectManager` creates a development plan based on the refined description.

4.  **Agent Selection:** The `ProjectManager` determines which agents are required based on keywords in the refined description (e.g., "REST API" implies a `BackendDeveloper`).

5.  **Iterative Code Generation:**
    *   The required agents (Frontend, Backend, Database, DataSampleGenerator) are instantiated.
    *   Each agent generates code based on its role and the current project context (description, plan, previously generated code).
    *   **Security Review (optional):** The `SecurityDeveloper` analyzes the generated code for vulnerabilities. If found, it suggests fixes, and the code is updated.
    *   **Testing (optional):** The `TestDeveloper` creates unit tests for the generated code. The tests are executed. If tests fail, the agent refines the code and the process repeats (up to `MAX_ITERATIONS`).
    *   The final code is written to files in the project directory.

6.  **Deployment Simulation:** The `Deployer` runs a basic check to see if the necessary files exist and if the tests (if generated) pass.

7.  **Feedback Consolidation and README Generation:** The `ProjectManager` consolidates feedback from the various agents, and a README file is created, summarizing the project and the development process.

## Limitations and Future Improvements

*   **Complexity:**  `TheTeam` is best suited for relatively simple applications.  Very complex applications may require more manual intervention.
*   **Framework Choices:**  The current implementation makes some default framework choices (e.g., Flask for backend).  Future versions could allow for more user control over framework selection.
*   **Error Handling:**  The error handling could be improved to provide more specific guidance to the user in case of failures.
*   **Integration Tests:**  Currently, only unit tests are generated.  Adding integration tests would further enhance code quality.
*   **Interactive Mode:**  An interactive mode where the user can provide feedback and guidance to the agents during the development process would be a valuable addition.
* **Deployment options:** Currently the tests and deployments are very limited. Add more options for deployments and test.

## Contributing

Contributions are welcome!  Please submit pull requests or open issues to discuss proposed changes.
