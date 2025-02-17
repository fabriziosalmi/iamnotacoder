# ✨ FabGPT: Automated Python Code Improvement and Generation Suite 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

FabGPT is a powerful toolkit, fueled by Large Language Models (LLMs), that automates the process of improving and generating Python code. It's designed to be flexible, robust, and easy to integrate into your workflow. The suite includes:

*   **`fabgpt.py` (The Optimizer 🛠️):** Your primary tool for enhancing existing Python code within GitHub repositories.  It combines static analysis, LLM-powered refactoring, automated testing, and seamless GitHub integration (including safe forking).
*   **`scraper.py` (The Finder 🔍):** Discovers and filters Python repositories on GitHub based on your criteria (e.g., code quality, lines of code, update date).
*   **`process.py` (The Orchestrator ⚙️):** Automates the execution of `fabgpt.py` on multiple repositories identified by `scraper.py`.
*   **`create_app_from_scratch.py` (The Creator 🏗️):** A code generation tool that builds basic Python applications from natural language descriptions. It leverages a team of specialized LLM "actors" to handle different development tasks.

This README focuses primarily on `fabgpt.py` (the optimizer) and provides an overview of `create_app_from_scratch.py` (the creator).

## 📑 Table of Contents

1.  [I. `fabgpt.py`: The Code Optimizer 🛠️](#i-fabgptpy-the-code-optimizer-)
    *   [Overview](#overview)
    *   [Features](#features-optimizer)
    *   [Installation](#installation-optimizer)
    *   [Usage](#usage-optimizer)
    *   [Configuration File (`config.toml`)](#configuration-file-configtoml---example)
    *   [Custom Prompts](#custom-prompts)
    *   [Examples](#examples-optimizer)
    *   [Workflow](#workflow-optimizer)
2.  [II. `scraper.py` and `process.py`: The Finder and Orchestrator 🔍⚙️](#ii-scraperpy-and-processpy-the-finder-and-orchestrator-)
    *    [Overview](#overview-1)
    *    [Usage](#usage-scraper-and-process)
3.  [III. `create_app_from_scratch.py`: The Application Creator 🏗️](#iii-create_app_from_scratchpy-the-application-creator-)
    *   [Overview](#overview-2)
    *   [Usage](#usage-creator)
    *   [Example Prompts](#example-prompts-creator)
    *   [Example](#example-creator)
4.  [Troubleshooting 🐛](#troubleshooting-)
5.  [Contributing 🤝](#contributing-)
6.  [License 📜](#license-)

## I. `fabgpt.py`: The Code Optimizer 🛠️

### Overview

`fabgpt.py` is your go-to tool for automatically improving the quality of existing Python code. It's designed to be a comprehensive solution for:

*   **Enhancing Code Style and Readability:** Makes your code more consistent, readable, and maintainable.
*   **Identifying and Fixing Potential Issues:** Detects potential bugs, security vulnerabilities, and performance bottlenecks.
*   **Generating and Running Tests:** Helps you increase test coverage and ensure code correctness.
*   **Integrating with GitHub:** Streamlines your workflow by automating the process of cloning, branching, committing, and creating pull requests (with safe forking).

### Features (Optimizer)

*   **🤖 Automated Code Improvement:**
    *   Leverages LLMs to suggest and apply code refactorings.
    *   Includes robust retry mechanisms for handling LLM calls and Git operations.
    *   Adapts to different LLM response formats.
*   **🔍 Static Analysis Integration:**
    *   Seamlessly integrates with popular static analysis tools:
        *   [Black](https://github.com/psf/black) (code formatting)
        *   [isort](https://pycqa.github.io/isort/) (import sorting)
        *   [Pylint](https://www.pylint.org/) (code analysis)
        *   [Flake8](https://flake8.pycqa.org/en/latest/) (style guide enforcement)
        *   [Mypy](http://mypy-lang.org/) (static typing)
    *   Allows you to configure which tools to use and exclude.
    *   Caches analysis results for improved performance.
*   **🧪 Automated Test Generation and Execution:**
    *   Generates unit tests using an LLM, increasing your code coverage.
    *   Automatically corrects syntax errors in generated tests.
    *   Executes tests using `pytest`.
    *   Reports code coverage and lets you set minimum coverage thresholds.
*   **🐙 GitHub Integration:**
    *   Clones repositories (using shallow clones for efficiency).
    *   Creates new branches for your improvements.
    *   Generates detailed, customizable commit messages.
    *   Creates pull requests directly on GitHub.
    *   **🛡️ Forking Support:** Automatically forks the target repository for a safe, non-destructive workflow.
*   **⚙️ Configuration:**
    *   Supports configuration via TOML files for easy setup.
    *   Command-line options override configuration file settings.
*   **✨ Other Key Features:**
    *   **Dry Run Mode:** Performs all analysis and improvement steps, but doesn't commit, push, or create pull requests.
    *   **Local Commit Mode:** Makes changes and commits locally, but skips pull request creation.
    *   **Customizable LLM Prompts:** Tailor the LLM's behavior with custom prompts for different improvement categories (style, maintenance, security, performance, and tests).
    *   **Comprehensive Reports:** Generates detailed text and JSON reports summarizing changes, analysis results, and test outcomes.
    *   **Robust Error Handling:** Includes extensive error handling and informative error messages.
    *   **Progress Indicators:** Provides visual progress bars and console output using the `rich` library.
    *   **Fast Mode:** Reduces delays for faster execution (useful for quick checks).
    *   **Local LLM Support:** Integrates with local LLMs (e.g., LM Studio) via the OpenAI API.
    *   **Configurable Line Length:** Enforces a maximum line length (defaults to the PEP 8 standard of 79 characters).
    *   **Output File Options:** Save modified files to a different path than the original.

### Installation (Optimizer)

1.  **Clone the Repository:**

    ```bash
    git clone <your_fabgpt_repo_url>
    cd FabGPT
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install black isort pylint flake8 mypy pytest pytest-cov  # Static analysis & testing tools
    ```

    Your `requirements.txt` should include at least:

    ```
    click
    openai
    toml
    GitPython
    rich
    PyGithub
    requests
    ```

### Usage (Optimizer)

**Basic Command Structure:**

```bash
python fabgpt.py --repo <repository_url> --files <file_paths> --branch <target_branch> --token <github_token> [options]
```

**Required Arguments:**

*   `--repo` (`-r`): 🐙 The URL of the GitHub repository (e.g., `https://github.com/user/repo`).
*   `--files` (`-f`): 📄 Comma-separated paths to the Python files you want to improve (e.g., `src/module1.py,src/module2.py`).
*   `--branch` (`-b`): 🌿 The target branch in the repository (e.g., `main`, `develop`).
*   `--token` (`-t`): 🔑 Your GitHub Personal Access Token (PAT) with the `repo` scope.

**Common Options:**

*   `--config` (`-c`): ⚙️ Path to a TOML configuration file.
*   `--tools` (`-T`): 🛠️ Comma-separated list of static analysis tools to use (default: `black,isort,pylint,flake8,mypy`).
*   `--exclude-tools` (`-e`): ❌ Comma-separated list of tools to exclude.
*   `--llm-model` (`-m`): 🧠 The LLM model to use (default: `qwen2.5-coder-14b-instruct-mlx`). Supports both OpenAI and local LLM models.
*   `--llm-temperature` (`-temp`): 🔥 The temperature for the LLM (default: 0.2). Higher values increase randomness.
*   `--llm-custom-prompt` (`-p`): 📝 Path to a directory containing your custom prompt files (e.g., `prompt_style.txt`).
*   `--min-coverage` (`-c`): 📊 Minimum code coverage threshold (as a percentage).
*   `--no-dynamic-analysis`: 🚫 Disable test generation and execution.
*   `--dry-run`: 👀 Perform a dry run without making any actual changes (no commits, pushes, or pull requests).
*   `--local-commit`: 💾 Only commit changes locally, don't create a pull request.
*   `--openai-api-base`: 🌐 Base URL for local LLMs (e.g., `http://localhost:1234/v1` for LM Studio).
*   `--categories` (`-C`): 🏷️ Comma-separated list of improvement categories (default: `style,maintenance,security,performance`).
*   `--output-file` (`-o`): 💾 Path to save the modified file. Defaults to overwriting the original file.
*   `--output-info`: 📝 Path to save the text report (default: `report.txt`).
*   `--fork-repo`: 🍴 Automatically fork the repository to your account before making changes.
*   `--fork-user`: 👤 Your GitHub username (if forking and your username cannot be inferred from the token).
*   `--line-length`: 📏 Maximum line length for code formatting (default: 79).
*   `--debug`: 🐛 Prints verbose logs for troubleshooting.
*    `--force-push`: 💪 Force push the branch if it exists on remote.

### Configuration File (`config.toml` - Example)

```toml
openai_api_key = "none"  # Set to "none" when using a local LLM
openai_api_base = "http://localhost:1234/v1"  # For LM Studio, if applicable
llm_model = "qwen2.5-coder-14b-instruct-mlx"
llm_temperature = 0.2
tools = "black,isort,pylint,flake8"
min_coverage = 80
coverage_fail_action = "warn"
```

### Custom Prompts

To customize the LLM's behavior, create a directory (default: `.`) and place text files named `prompt_<category>.txt` within it. Use `{code}` as a placeholder for the code to be improved, and `{file_base_name}` for the file base name.

Example (`prompt_style.txt`):

```
You are a coding assistant tasked with improving the style of the following Python code.
Focus on PEP 8 compliance, readability, and clarity. Return only the improved code,
without any introductory or concluding text. Do not include markdown code fences.

{code}
```

Example (`prompt_tests.txt`):

```
You are a coding assistant tasked with writing test for the following Python code.
Focus on readability, and clarity. Return only the test code,
without any introductory or concluding text. Do not include markdown code fences.
Write test for {file_base_name}.py file:

{code}
```

### Examples (Optimizer)

1.  **Basic Usage (with a configuration file):**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN -c config.toml
    ```

2.  **Dry Run with Debug Logging:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --dry-run --debug
    ```

3.  **Using a Local LLM with LM Studio:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --openai-api-base http://localhost:1234/v1 --llm-model qwen2.5-coder-14b-instruct-mlx
    ```

    (and set `openai_api_key = "none"` in `config.toml`)

4.  **Forking and Automatically Generating Tests:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --fork-repo --fork-user yourusername
    ```

5.  **Running on Multiple Files and exclude mypy tool:**

    ```bash
    python fabgpt.py --repo https://github.com/user/repo --files "src/module1.py,src/module2.py,tests/test_module1.py" --branch development --token YOUR_GITHUB_TOKEN --exclude-tools mypy --fork-repo
    ```

### Workflow (Optimizer)

1.  **Clone:** 🐙 The repository is cloned (using a shallow clone) to a temporary directory.
2.  **Branch:** 🌿 A new branch is created with a unique, descriptive name.
3.  **Static Analysis:** 🔍 The selected static analysis tools are run, and the results are cached (if `--cache-dir` is specified).
4.  **Test Generation (Optional):** 🧪 If enabled (`--no-dynamic-analysis` is *not* set), tests are generated using the LLM. Syntax errors are automatically corrected.
5.  **Test Execution (Optional):** 🚦 Tests are run, and code coverage is checked if `--min-coverage` is specified.
6.  **LLM Improvement:** 🧠 The LLM is used to improve the code, guided by custom prompts. Retries are performed if necessary.
7.  **Change Verification:** ✅ The script checks if any actual code changes were made.
8.  **Commit:** 💾 Changes (including generated tests) are committed to the new branch.
9.  **Push:** 🚀 The new branch is pushed to the remote repository (your fork, if `--fork-repo` is used).
10. **Pull Request:** 🎁 A pull request is created on GitHub.
11. **Reporting:** 📝 A text report (`report.txt` or as specified by `--output-info`) and a JSON log file are generated.
12. **Cleanup:** 🧹 The temporary directory is removed (unless `--debug` is enabled).

## II. `scraper.py` and `process.py`: The Finder and Orchestrator 🔍⚙️

### Overview

`scraper.py` is a script engineered to locate and sift through Python repositories on GitHub based on a set of criteria you define. These criteria can include aspects like the number of lines of code in the files, the ratio of comments to code (an indicator of code quality), and the last update date of the repository.  It outputs a JSON file containing a list of repositories and files that meet your criteria.

The `process.py` script acts as a conductor, efficiently running `fabgpt.py` across a multitude of Python repositories. These repositories are typically discovered using `scraper.py`, making the process streamlined and automated.  It reads the JSON output from `scraper.py` and runs `fabgpt.py` on each identified repository and file.

### Usage (scraper and process)

1.  **Run the Scraper:**

    ```bash
    python scraper.py --token YOUR_GITHUB_TOKEN --max-repos 10 --quality-threshold 50 --output output.json
    ```
    This command searches for up to 10 repositories, includes Python files with a "quality score" (lines of code in this example) of 50 or less, and saves the results to `output.json`.  You can adjust parameters like `--min-lines`, `--max-lines`, `--start-date`, and `--end-date` to refine your search.

2.  **Process the Results with process.py:**

    ```bash
    python process.py --input output.json --token YOUR_GITHUB_TOKEN --config config.toml --branch main --output results.json --fork
    ```

    This command reads the `output.json` file (generated by `scraper.py`), uses your GitHub token, applies the settings from `config.toml`, targets the `main` branch, saves the processing results to `results.json`, and forks each repository before making changes. It effectively automates running `fabgpt.py` on the repositories and files found by the scraper.

## III. `create_app_from_scratch.py`: The Application Creator 🏗️

### Overview

`create_app_from_scratch.py` is your AI-powered coding assistant for generating basic Python applications from scratch. It leverages a team of specialized LLM "actors," each with a specific role:

*   **Backend Developer 🧠:** Creates the core application logic, typically generating a `backend.py` file.
*   **Frontend Developer 🎨:** Creates a simple frontend (e.g., `frontend.py`), *if* the Project Manager determines it's necessary based on the application description.
*   **Creative Assistant ✨:** Refines the initial application description, providing more detail and clarity.
*   **Security Developer 🛡️:** Reviews the generated code for potential vulnerabilities and suggests improvements.
*   **Project Manager 📝:** Creates a development plan and consolidates feedback from all the actors.

This collaborative approach helps to produce more complete and well-structured applications.

### Usage (Creator)

```bash
python create_app_from_scratch.py --app-description "A simple web app to track tasks." [options]
```

**Required Argument:**

*   `--app-description` (`-d`): 💬 A clear and concise description of the application you want to create. The more detail you provide, the better the results.

**Common Options:**

*   `--llm-model` (`-m`): 🧠 The LLM model to use (default: `qwen2.5-coder-7b-instruct-mlx`).
*   `--llm-temperature` (`-temp`): 🔥 The temperature for the LLM (default: 0.2).
*   `--llm-custom-prompt` (`-p`): 📝 Path to a directory containing your custom prompt files for each actor.
*   `--openai-api-base`: 🌐 Base URL for local LLMs (e.g., `http://localhost:1234/v1` for LM Studio).
*    `--config`: Path to TOML configuration file.
*   `--debug`: 🐛 Enable debug logging.

### Example Prompts (Creator)

You'll need to create prompt files (e.g., `prompt_backend.txt`, `prompt_frontend.txt`, `prompt_security_review.txt`, etc.) in the custom prompt directory. These prompts will guide the different LLM actors. Use placeholders like `{description}` and `{existing_code}` to inject the relevant information.

### Example (Creator)

```bash
python create_app_from_scratch.py -d "A command-line tool to convert Markdown files to HTML, with support for custom templates." --openai-api-base http://localhost:1234/v1 --config config.toml --debug
```

The generated application files will be created in a temporary directory. The script is designed *not* to automatically delete this directory, so you can inspect and modify the generated code.

## Troubleshooting 🐛

*   **API Key Issues:** Ensure your `OPENAI_API_KEY` or `OPENAI_API_BASE` environment variables are set correctly. You can also provide these values via the `--config` file or the `--openai-api-key` / `--openai-api-base` command-line options. If you're using a local LLM, set `openai_api_key = "none"` in your `config.toml`.
*   **GitHub Token Permissions:** Your GitHub PAT must have the `repo` scope.
*   **Tool Not Found:** If a static analysis tool is not found, make sure it's installed (`pip install <tool_name>`).
*   **LLM Errors:** If the LLM consistently fails, consider adjusting the `--llm-temperature` or using a different `--llm-model`.
*   **Pull Request Creation Failures:** Double-check your token, repository URL, and branch names. Make sure you have write access to the repository (or use the forking workflow).
* **Forking error**: Ensure you insert your github username with `--fork-user`

## Contributing 🤝

Contributions are highly welcome! Please submit pull requests or open issues to discuss proposed changes or report bugs.

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
