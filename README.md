# FabGPT

[![GitHub version](https://badge.fury.io/gh/efraimgentil%2FFabGPT.svg)](https://badge.fury.io/gh/efraimgentil%2FFabGPT)
[![PyPI version](https://badge.fury.io/py/FabGPT.svg)](https://badge.fury.io/py/FabGPT)
![GitHub contributors](https://img.shields.io/github/contributors/efraimgentil/FabGPT)
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/efraimgentil/FabGPT/main)
![GitHub pull requests](https://img.shields.io/github/issues-pr/efraimgentil/FabGPT)
[![Downloads](https://static.pepy.tech/badge/fabgpt)](https://pepy.tech/project/fabgpt)

FabGPT is a command-line tool designed to enhance Python code quality in GitHub repositories.  It leverages Large Language Models (LLMs) to analyze, improve, and generate tests for your code. FabGPT automates the process of code review, refactoring, and test generation, making it easier to maintain high-quality code. It can integrate with various static analysis tools, testing frameworks, and provides flexible configuration options, including custom prompts for fine-grained control over LLM behavior.  The tool supports local-only commits or full pull request creation.  It also includes robust error handling, caching, and reporting.

## Index

1.  [Features](#features)
2.  [Installation](#installation)
3.  [Configuration](#configuration)
    *   [TOML Configuration File](#toml-configuration-file)
    *   [Environment Variables](#environment-variables)
4.  [Usage](#usage)
    *  [Command-Line Options](#command-line-options)
    *   [Examples](#examples)
5.  [Workflow](#workflow)
6.  [Custom Prompts](#custom-prompts)
7.  [Static Analysis Tools](#static-analysis-tools)
8.  [Test Frameworks](#test-frameworks)
9.  [Caching](#caching)
10. [Logging](#logging)
11. [Error Handling](#error-handling)
12. [Contributing](#contributing)
13. [License](#license)

## 1. Features <a name="features"></a>

*   **Code Improvement:** Uses LLMs to suggest and apply improvements to Python code based on specified categories (style, maintenance, security, performance).
*   **Static Analysis Integration:** Integrates with popular static analysis tools like `pylint`, `flake8`, `black`, `isort`, and `mypy`.
*   **Test Generation:** Automatically generates unit tests for improved code using a specified testing framework (currently supports `pytest`).
*   **Dynamic Analysis:**  Runs generated tests and checks code coverage.
*   **Configurable:**  Supports configuration via TOML files, command-line options, and environment variables.
*   **Custom Prompts:** Allows custom prompts to guide the LLM's behavior for different improvement categories and test generation.
*   **Caching:** Caches static analysis results to speed up subsequent runs.
*   **Pull Request Creation:**  Automatically creates a pull request on GitHub with the improvements.
*   **Local Commits:**  Option to only commit changes locally without creating a pull request.
*   **Backup and Restore:** Creates backups before making changes and restores them if errors occur.
*   **Reporting:** Generates a detailed report of the changes, analysis results, and test results.
*   **Error Handling:** Robustly handles errors during command execution, LLM interaction, and Git operations.
* **OpenAI and Local LLM Support**: Use OpenAI's API or local LLM backends (like LM Studio)

## 2. Installation <a name="installation"></a>

FabGPT requires Python 3.12.8 or later.

```bash
pip install FabGPT
```
or
```bash
git clone https://github.com/efraimgentil/FabGPT
cd FabGPT
pip install -r requirements.txt
```
This will install the required dependencies, including:

*   `gitpython`: For Git operations.
*   `toml`: For TOML configuration parsing.
*   `click`: For command-line interface creation.
*   `openai`: For interacting with the OpenAI API.
*   `rich`: For enhanced console output.
*   `PyGithub`: To interact with Github API.
* `difflib`:  For calculating differences between code versions.
*  `pytest`: (Optional, but recommended) for running tests.
*  Static analysis tools (Optional, but recommended): `black`, `isort`, `pylint`, `flake8`, `mypy`.

You should install any of the optional, recommended tools you plan to use.  For example:

```bash
pip install black isort pylint flake8 mypy pytest
```

## 3. Configuration <a name="configuration"></a>

FabGPT can be configured using a combination of a TOML configuration file, environment variables, and command-line options. Command-line options take precedence over environment variables, which in turn take precedence over the TOML file.

### 3.A. TOML Configuration File <a name="toml-configuration-file"></a>

Create a `config.toml` file in the project directory or specify its path using the `--config` option.

Example `config.toml`:

```toml
openai_api_key = "none"  # Or your OpenAI API Key
openai_api_base = "http://localhost:1234/v1"  # For LM Studio, or similar
llm_model = "qwen2.5-coder-14b-instruct-mlx"
llm_temperature = 0.2
llm_optimization_level = "balanced"
tools = "black,isort,pylint,flake8"
exclude_tools = ""
test_framework = "pytest"
min_coverage = 80
coverage_fail_action = "warn"
```

*   `openai_api_key`: Your OpenAI API key.  Set to `"none"` if using a local LLM server.
*   `openai_api_base`: The base URL for the OpenAI API.  Use this for local LLM servers (e.g., LM Studio).
*   `llm_model`:  The LLM model to use.
*   `llm_temperature`:  The temperature for the LLM (controls randomness).
*   `llm_optimization_level`:  The optimization level for LLM improvements (`balanced`, etc.).  This setting is passed to your custom prompts.
*   `tools`:  A comma-separated list of static analysis tools to use.
*   `exclude_tools`: A comma-separated list of tools to exclude.
*   `test_framework`:  The testing framework to use (`pytest`).
*   `min_coverage`:  The minimum code coverage percentage required.
*   `coverage_fail_action`:  Action to take if coverage is below the minimum (`fail` or `warn`).

### 3.B. Environment Variables <a name="environment-variables"></a>

You can also set configuration options using environment variables.

*   `OPENAI_API_KEY`: Your OpenAI API key.
*   `OPENAI_API_BASE`:  The base URL for the OpenAI API (e.g., for LM Studio).

## 4. Usage <a name="usage"></a>

### 4.A. Command-Line Options <a name="command-line-options"></a>

```
Usage: fabgpt.py [OPTIONS]

  Improves a Python file in a GitHub repository, generates tests, and creates a
  Pull Request.

Options:
  -r, --repo TEXT                 GitHub repository URL.  [required]
  -f, --files TEXT                Comma-separated relative paths to files to
                                  improve.  [required]
  -b, --branch TEXT               Target branch name.  [required]
  -t, --token TEXT                GitHub Personal Access Token (PAT).
                                  [required]
  -T, --tools TEXT                Static analysis tools (comma-separated).
  -e, --exclude-tools TEXT        Tools to exclude (comma-separated).
  -m, --llm-model TEXT            LLM model to use.
  -temp, --llm-temperature FLOAT  Temperature for the LLM.
  -l, --llm-optimization-level TEXT
                                  LLM optimization level.
  -p, --llm-custom-prompt TEXT    Path to a custom prompt directory.
  -F, --test-framework TEXT       Test framework.
  -c, --min-coverage FLOAT        Minimum code coverage threshold.
  --coverage-fail-action [fail|warn]
                                  Action on insufficient coverage.
  -cm, --commit-message TEXT      Custom commit message.
  --no-dynamic-analysis           Disable dynamic analysis (testing).
  --cache-dir TEXT                Directory for caching.
  --debug                         Enable debug logging.
  --dry-run                       Run without making changes.
  --local-commit                  Only commit locally, don't create a Pull
                                  Request.
  --fast                          Enable fast mode by reducing delays.
  --openai-api-base TEXT          Base URL for OpenAI API (for LMStudio, e.g.,
                                  http://localhost:1234/v1).
  --config PATH                   Configuration file.
  --no-output                     Disable console output.
  -C, --categories TEXT           Comma-separated list of improvement
                                  categories.  Defaults to
                                  'style,maintenance,security,performance'.
  --force-push                    Force push the branch if it already exists.
  -o, --output-file TEXT          Path to save the modified file. Defaults to
                                  overwriting the original.
  --output-info TEXT               Path to save the TEXT report. Defaults to
                                  report.txt
  --line-length INTEGER           Maximum line length for code formatting.
  --help                          Show this message and exit.

```

### 4.B. Examples <a name="examples"></a>

1.  **Basic usage with default settings:**

    ```bash
    fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token
    ```

2.  **Using a configuration file:**

    ```bash
    fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token --config config.toml
    ```

3.  **Specifying static analysis tools and a custom prompt directory:**

    ```bash
    fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token -T "black,isort,pylint" -p /path/to/your/prompts
    ```

4.  **Running in dry-run mode:**

    ```bash
    fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token --dry-run
    ```

5. **Using local commit only:**
    ```bash
    fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token --local-commit
    ```
6.  **Using a local LLM server (e.g., LM Studio):**

    ```bash
    fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token --openai-api-base http://localhost:1234/v1
    ```

    Make sure the `config.toml` also sets `openai_api_key = "none"`.
7. **Improve multiple files**
   ```bash
    fabgpt -r https://github.com/your-username/your-repo -f file1.py,file2.py,utils/helper.py -b main -t your_github_token
   ```
8. **Specifying Improvement Categories**
    ```bash
   fabgpt -r https://github.com/your-username/your-repo -f your_file.py -b main -t your_github_token -C "style,performance"
    ```

## 5. Workflow <a name="workflow"></a>

1.  **Clone Repository:** Clones the specified GitHub repository (shallow clone) to a temporary directory.
2.  **Checkout Branch:** Checks out the specified target branch.  Fetches if necessary.
3.  **Create Branch:** Creates a new, uniquely-named branch for the improvements.
4.  **Analyze Project (Optional):** Runs specified static analysis tools on the file.  Results are cached if a cache directory is provided.
5.  **Improve File:**  Reads the file content, then iterates through the specified improvement categories:
    *   Loads the appropriate custom prompt for the category (if available).
    *   Sends the code and prompt to the LLM.
    *   Cleans the LLM response to extract the improved code.
    *   Checks the improved code for syntax errors.
    *   Retries LLM calls and syntax fixes up to a maximum number of attempts.
    *  Formats code with `black` and `isort` before sending to the LLM.
6.  **Generate Tests (Optional):**  Sends the (potentially improved) code to the LLM to generate tests.
    *   Uses a custom prompt for test generation (if available).
    *   Checks generated tests for syntax errors.
    *   Retries LLM calls and syntax fixes.
    *  Writes tests to a `tests/` directory.
7.  **Run Tests (Optional):** Runs the generated tests (currently only `pytest` is supported).
    *   Checks code coverage if `--min-coverage` is specified.
    *  Handles test failures and coverage failures based on `--coverage-fail-action`.
8.  **Create Commit:** Creates a Git commit with the changes (improved code and generated tests).  A custom commit message can be provided.
9.  **Create Pull Request (Optional):**  Pushes the new branch to GitHub and creates a pull request. The PR title and body describe the changes made.
10. **Create Info File:** Generates a `report.txt` (or custom-named) file with details of the changes, analysis results, and test results.
11. **Cleanup:**  Deletes the temporary directory (unless `--debug` is enabled).

## 6. Custom Prompts <a name="custom-prompts"></a>

Custom prompts allow fine-grained control over the LLM's behavior.  You can provide a directory containing prompt files using the `--llm-custom-prompt` option.

*   **Improvement Prompts:**  Create text files named `prompt_{category}.txt` (e.g., `prompt_style.txt`, `prompt_maintenance.txt`) in the custom prompt directory.  These prompts should instruct the LLM on how to improve the code for that specific category.

    The prompt template should contain a placeholder `{code}` which will be replaced with the current code.  You can also use the `--llm-optimization-level` option to pass a hint to the prompt.  This value is available in the prompt, but you must include it in the prompt text itself.

    Example `prompt_style.txt`:

    ```text
    You are a coding assistant tasked with improving the style of Python code.  
    Refactor the following code to improve its style, following PEP 8 guidelines.
    Optimization level: {llm_optimization_level}
    
    Maintain a maximum line length of 79 characters.

    {code}

    Return only the improved code, without any introductory or concluding text.
    Do not include markdown code fences.
    ```

*   **Test Generation Prompt:** Create a text file named `prompt_tests.txt` in the custom prompt directory. This prompt should instruct the LLM how to generate tests.

    The prompt template should include `{code}` (replaced with the file's code) and `{file_base_name}` (replaced with the file's base name without extension).

    Example `prompt_tests.txt`:

    ```text
    You are a coding assistant tasked with generating unit tests for Python code.

    Write unit tests for the following code using pytest.  
    The file name is {file_base_name}.py.
    Maintain a maximum line length of 79 characters.

    {code}

    Return only the test code, without any introductory or concluding text.
    Do not include markdown code fences.
    ```

## 7. Static Analysis Tools <a name="static-analysis-tools"></a>

FabGPT supports the following static analysis tools:

*   `pylint`
*   `flake8`
*   `black`
*   `isort`
*   `mypy`

The `--tools` option specifies which tools to use. The `--exclude-tools` option specifies which tools *not* to use.  The tool must be installed in your environment for FabGPT to use it.

## 8. Test Frameworks <a name="test-frameworks"></a>

Currently, only `pytest` is supported as a test framework.

## 9. Caching <a name="caching"></a>

Static analysis results are cached to improve performance.  Use the `--cache-dir` option to specify a directory for storing the cache files.  The cache key is based on the file path, the selected tools, excluded tools, and the configured line length. If the cache directory is not specified, caching is disabled.

## 10. Logging <a name="logging"></a>

FabGPT uses the `rich` library for enhanced console logging.  The `--debug` option enables more verbose logging.  Log messages indicate progress, errors, and warnings.  A JSON log file is also created in a `logs/` directory within the FabGPT directory, containing detailed information about each run.

## 11. Error Handling <a name="error-handling"></a>

FabGPT handles various errors gracefully:

*   **Command Execution Errors:** Errors from running shell commands (e.g., static analysis tools) are caught and reported.
*   **LLM Errors:**  Errors from the LLM (including timeouts) are caught, and the operation is retried up to `MAX_LLM_RETRIES` times.
*   **Syntax Errors:**  Syntax errors in the LLM-generated code are caught, and the LLM is prompted to fix them (up to `MAX_SYNTAX_RETRIES` times).
*   **Git Errors:** Errors during Git operations (cloning, branching, committing, pushing) are caught and reported.
*   **File I/O Errors:** Errors reading or writing files are caught and handled.

If errors occur during file modification, the code is restored from a backup.

## 12. Contributing <a name="contributing"></a>

Contributions are welcome! Please submit issues or pull requests on the [GitHub repository](https://github.com/efraimgentil/FabGPT).

## 13. License <a name="license"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
