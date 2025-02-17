# FabGPT: Automated Python Code Improvement Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

FabGPT is a command-line tool designed to automatically improve the quality of Python code within GitHub repositories. It leverages static analysis tools, Large Language Models (LLMs), and automated testing to enhance code style, maintainability, security, and performance.  FabGPT can operate on individual files or process multiple files from a JSON list, making it suitable for both targeted improvements and broader code quality initiatives.  It integrates seamlessly with GitHub, offering both local commit and pull request creation capabilities, and it fully supports forking for a non-destructive workflow.

## Features

*   **Automated Code Improvement:**  Uses LLMs to suggest and apply improvements in various categories (style, maintainability, security, performance).  Provides a robust retry mechanism and handles diverse LLM response formats.
*   **Static Analysis Integration:** Integrates with popular static analysis tools:
    *   [Black](https://github.com/psf/black) (code formatting)
    *   [isort](https://pycqa.github.io/isort/) (import sorting)
    *   [Pylint](https://www.pylint.org/) (code analysis)
    *   [Flake8](https://flake8.pycqa.org/en/latest/) (style guide enforcement)
    *   [Mypy](http://mypy-lang.org/) (static typing)
    *   Tool selection and exclusion are configurable.
*   **Automated Test Generation:**  Can generate unit tests using an LLM, helping to improve code coverage.  Includes a syntax error correction loop for generated tests.
*   **Test Execution and Coverage Reporting:**  Runs tests using `pytest` and reports code coverage. Configurable minimum coverage thresholds and actions (fail/warn) for insufficient coverage.
*   **GitHub Integration:**
    *   Clones repositories (shallow clone for efficiency).
    *   Creates new branches for improvements.
    *   Creates commits with detailed, customizable messages.
    *   Creates pull requests directly on GitHub.
    *   **Forking Support:** Automatically forks the target repository, allowing for a safe, non-destructive workflow.  Changes are pushed to the fork, and pull requests are created from the fork to the original repository.
*   **Configuration:**
    *   Supports configuration via TOML files.
    *   Command-line options override configuration file settings.
*   **Caching:** Caches static analysis results to improve performance on subsequent runs.
*   **Dry Run Mode:**  Performs all analysis and improvement steps but doesn't commit, push, or create pull requests.
*   **Local Commit Mode:**  Makes changes and commits locally, but does not create a pull request.
*   **Customizable Prompts:** Allows users to provide custom prompt templates for the LLM, enabling fine-grained control over the improvement process.  Prompts are separated by category (style, maintenance, etc.).
*   **Detailed Reporting:** Generates a comprehensive text report summarizing changes, static analysis results, test results, and LLM improvements.  Also generates a JSON log file for tracking.
*   **Robust Error Handling:** Includes extensive error handling, retries for LLM calls and Git operations, and informative error messages.
*   **Progress Indicators:** Uses `rich` library for visually appealing progress bars and console output.
*   **Fast Mode:** Option to reduce delays for faster execution (useful for quick checks).
*   **Support for Local LLMs:** Can be configured to use local LLMs via the OpenAI API (e.g., with LM Studio).
*   **Line Length Control:**  Enforces a configurable maximum line length, defaulting to the PEP 8 standard of 79 characters.
* **Output File Options:** Allows user to save the modified file to another path respect the original.
* **Output Info File:** Create a complete report in txt format.

## Installation

1.  **Clone the FabGPT repository:**

    ```bash
    git clone https://github.com/your-username/FabGPT.git  # Replace with your repository URL
    cd FabGPT
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    
    *requirements.txt* should contain at least:
    ```
    click
    openai
    toml
    GitPython
    rich
    PyGithub
    requests
    ```

    You'll also need to install the static analysis tools you intend to use.  For example:

    ```bash
    pip install black isort pylint flake8 mypy pytest pytest-cov
    ```

## Usage

### Basic Command Structure

```bash
python fabgpt.py --repo <repository_url> --files <file_paths> --branch <target_branch> --token <github_token> [options]
```

### Required Arguments

*   `--repo` (`-r`):  The URL of the GitHub repository (e.g., `https://github.com/user/repo`).
*   `--files` (`-f`): Comma-separated relative paths to the Python files to be improved (e.g., `src/module1.py,src/module2.py`).
*   `--branch` (`-b`): The target branch in the repository (e.g., `main`, `develop`).
*   `--token` (`-t`): Your GitHub Personal Access Token (PAT) with appropriate permissions (repo scope is generally required).

### Common Options

*   `--config` (`-c`): Path to a TOML configuration file. Command-line options override config file settings.
*   `--tools` (`-T`):  Comma-separated list of static analysis tools to use (default: `black,isort,pylint,flake8,mypy`).
*   `--exclude-tools` (`-e`): Comma-separated list of tools to exclude.
*   `--llm-model` (`-m`):  The LLM model to use (default: `qwen2.5-coder-14b-instruct-mlx`).  This can be an OpenAI model name or a model name compatible with your local LLM setup.
*   `--llm-temperature` (`-temp`):  The temperature for the LLM (default: 0.2).
*   `--llm-optimization-level` (`-l`): Optimization level for the LLM (default: `balanced`).
*   `--llm-custom-prompt` (`-p`): Path to a directory containing custom prompt files (default: `.`).
*   `--test-framework` (`-F`):  The test framework to use (default: `pytest`).  Currently, only `pytest` is supported.
*   `--min-coverage` (`-c`):  Minimum code coverage threshold (as a percentage).
*   `--coverage-fail-action` : Action to take if coverage is below the threshold (default: `fail`). Choices: `fail`, `warn`.
*   `--no-dynamic-analysis`: Disable dynamic analysis (testing).
*   `--cache-dir`: Directory to store cached analysis results.
*   `--debug`: Enable debug logging.
*   `--dry-run`: Perform a dry run without making any actual changes.
*   `--local-commit`: Only commit changes locally, don't create a pull request.
*   `--fast`: Enable fast mode (reduces delays).
*   `--openai-api-base`:  Base URL for the OpenAI API (for local LLMs, e.g., `http://localhost:1234/v1`).
*   `--no-output`: Disable all console output.
*   `--categories` (`-C`): Comma-separated list of improvement categories (default: `style,maintenance,security,performance`).
*   `--force-push`: Force push the branch if it already exists.
*   `--output-file` (`-o`):  Path to save the modified file. Defaults to overwriting the original.
*   `--output-info`: Path to save the TEXT report (default: `report.txt`).
*   `--line-length`: Maximum line length for code formatting (default: 79).
* `--fork-repo`: Automatically fork the repository.
* `--fork-user`: Your GitHub username (if different from what can be inferred from the token).

### Configuration File (config.toml)

The `config.toml` file allows you to set default values for most options.  Example:

```toml
openai_api_key = "none"  # Or your OpenAI API key, or use an environment variable
openai_api_base = "http://localhost:1234/v1"  # For LM Studio, if applicable
llm_model = "qwen2.5-coder-14b-instruct-mlx"
llm_temperature = 0.2
llm_optimization_level = "balanced"
tools = "black,isort,pylint,flake8"
exclude_tools = ""
test_framework = "pytest"
min_coverage = 80
coverage_fail_action = "warn"
```

### Custom Prompts

Create a directory (default: `.`) and place text files named `prompt_<category>.txt` within it.  For example:

*   `prompt_style.txt`:  Contains the prompt for style improvements.
*   `prompt_maintenance.txt`: Contains the prompt for maintainability improvements.
*   `prompt_security.txt`:  Contains the prompt for security improvements.
*   `prompt_performance.txt`: Contains the prompt for performance improvements.
* `prompt_tests.txt`: Contains the prompt for tests generation.

Within the prompt files, use `{code}` as a placeholder for the code to be improved and `{file_base_name}` for the file base name.

Example `prompt_style.txt`:

```
You are a coding assistant tasked with improving the style of the following Python code.
Focus on PEP 8 compliance, readability, and clarity.  Return only the improved code,
without any introductory or concluding text. Do not include markdown code fences.

{code}
```
Example 'prompt_tests.txt'
```
You are a coding assistant tasked with writing test for the following Python code.
Focus on  readability, and clarity.  Return only the test code,
without any introductory or concluding text.  Do not include markdown code fences.
Write test for {file_base_name}.py file:

{code}
```

### Examples

1.  **Basic usage with a configuration file:**

    ```bash
    python fabgpt.py --repo https://github.com/user/repo --files src/my_module.py --branch main --token YOUR_GITHUB_TOKEN --config config.toml
    ```

2.  **Dry run with debug logging:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --debug --dry-run
    ```

3.  **Using a local LLM with LM Studio:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --openai-api-base http://localhost:1234/v1 --llm-model qwen2.5-coder-14b-instruct-mlx
    ```
    (and set `openai_api_key = "none"` in `config.toml`)

4.  **Using forking, and automatically generating tests:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --fork-repo --fork-user yourusername
    ```
5. **Save the modified file to another path:**

    ```bash
    python fabgpt.py -r https://github.com/user/repo -f src/my_module.py -b main -t YOUR_GITHUB_TOKEN --fork-repo --fork-user yourusername --output-file improved_module.py
    ```
6.  **Running on Multiple Files and exclude mypy tool:**

    ```bash
    python fabgpt.py --repo https://github.com/user/repo --files "src/module1.py,src/module2.py,tests/test_module1.py" --branch development --token YOUR_GITHUB_TOKEN --exclude-tools mypy --fork-repo
    ```
7.  **Using scraper.py and process.py to run FabGPT on multiple Repositories**
  * **Run the Scraper:**
   First, use `scraper.py` to find repositories and files that meet your criteria. This will output a JSON file containing a list of repositories and files.

    ```bash
    python scraper.py --token YOUR_GITHUB_TOKEN --max-repos 10 --quality-threshold 50 --output output.json
    ```
    This command searches for up to 10 repositories, includes Python files with a "quality score" (lines of code in this example) of 50 or less, and saves the results to `output.json`.

   * **Process the Results with process.py:**
  Next, use `process.py` to run `fabgpt.py` on the repositories and files listed in the JSON file created by the scraper.
    ```bash
    python process.py --input output.json --token YOUR_GITHUB_TOKEN --config config.toml --branch main --output results.json --fork
    ```
    This command reads the `output.json` file, uses your GitHub token, applies the settings from `config.toml`, targets the `main` branch, saves the processing results to `results.json`, and forks each repository before making changes.

## Workflow

1.  **Cloning:** The repository is cloned (shallow clone) to a temporary directory.
2.  **Branching:**  A new branch is created with a unique, descriptive name based on the file name, purpose, timestamp, and a UUID.  The target branch is checked out first, and then the new branch is created from it.
3.  **Static Analysis:** The selected static analysis tools are run, and the results are cached (if `--cache-dir` is specified).
4.  **Test Generation:** If enabled (`--no-dynamic-analysis` is *not* set), tests are generated using the LLM.  Any syntax errors in the generated tests are automatically corrected (with retries).
5.  **Test Execution:**  Tests are run (if generated and the test framework is available).  Code coverage is checked if `--min-coverage` is specified.
6.  **LLM Improvement:** The LLM is used to improve the code based on the specified categories and custom prompts.  The code is formatted with Black and isort *before* being sent to the LLM.  Retries are performed if the LLM fails or returns invalid code.
7. **Check for Changes:** Before creating any commit, the script verify if there are effective changes. If no changes, skip the commit.
8.  **Commit:**  Changes (including generated tests) are committed to the new branch.  The commit message is automatically generated and includes details about the improvements made.
9.  **Push:** The new branch is pushed to the remote repository (your fork if `--fork-repo` is used).
10. **Pull Request:**  A pull request is created on GitHub (from your fork to the original repository if forking is enabled). The PR title and body summarize the changes.
11. **Reporting:** A text report (`report.txt` or as specified by `--output-info`) is generated with a summary of the changes. A JSON log file is also created in the `logs` directory.
12. **Cleanup:** The temporary directory is removed (unless `--debug` is enabled).

## Troubleshooting

*   **OpenAI API Key Issues:** Ensure your `OPENAI_API_KEY` environment variable is set correctly, or provide the key via `--openai-api-key` or the `config.toml` file.  If using a local LLM, set `openai_api_key = "none"` in `config.toml` and provide the `--openai-api-base` URL.
*   **GitHub Token Permissions:** Make sure your GitHub PAT has the necessary permissions (usually `repo` scope).
*   **Tool Not Found:**  If a static analysis tool is not found, it will be skipped.  Make sure all required tools are installed.
*   **Test Failures:**  If tests fail, review the test output and the generated tests.  You may need to manually adjust the tests.
*   **LLM Errors:**  If the LLM consistently fails, try adjusting the `--llm-temperature` or using a different `--llm-model`.
*   **Pull Request Creation Failures:** Double-check your token, repository URL, and branch names.  Ensure you have write access to the repository (or use the forking workflow).
* **Forking error**: Ensure you insert your github username with `--fork-user`

## Contributing

Contributions are welcome! Please submit pull requests or open issues to discuss proposed changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
