import os
import sys
import toml
import shutil
import ast
import re
import uuid
import json
import datetime
import logging
from rich.console import Console
from io import StringIO
from rich.table import Table
from collections import Counter
from rich import box

console = Console()

# Load configuration from a TOML file.
def load_config(config_file: str) -> dict:
    try:
        config = toml.load(config_file)
        if "prompts" in config:
            for category, prompt_data in config["prompts"].items():
                if isinstance(prompt_data, str):
                    with open(prompt_data, "r", encoding="utf-8") as f:
                        config["prompts"][category] = f.read()
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        logging.error(f"Error decoding TOML: {e}")
        sys.exit(1)
    except Exception as e:
        logging.exception(f"Error loading configuration file: {e}")
        sys.exit(1)

# Retrieve prompt string from config or custom file.
def get_prompt(config: dict, category: str, custom_prompt_dir: str) -> str:
    prompt = ""
    if "prompts" in config and category in config["prompts"]:
        prompt = config["prompts"][category]
    prompt_file = os.path.join(custom_prompt_dir, f"prompt_{category}.txt")
    if os.path.exists(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        except Exception as e:
            logging.error(f"Error loading custom prompt from {prompt_file}: {e}")
    if not prompt:
        logging.warning(f"No prompt found for category '{category}'. Using default.")
        prompt = f"Improve the following code in terms of {category}:\n{{code}}"
    return prompt

# Create a backup copy of a file.
def create_backup(file_path: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.bak.{timestamp}"
    try:
        shutil.copy2(file_path, backup_path)
        console.print(f"[green]Backup created: {backup_path}[/green]")
        return backup_path
    except Exception as e:
        logging.exception(f"Backup creation failure for {file_path}: {e}")
        return None

# Restore from a backup file.
def restore_backup(file_path: str, backup_path: str) -> None:
    try:
        shutil.copy2(backup_path, file_path)
        console.print(f"[yellow]File restored from: {backup_path}[/yellow]")
    except FileNotFoundError:
        logging.error(f"Backup file not found: {backup_path}")
    except Exception as e:
        logging.exception(f"Restore backup failure for {file_path} from {backup_path}: {e}")

# Callback to prioritize CLI config.
def get_cli_config_priority(ctx, param, value) -> dict:
    config = ctx.default_map or {}
    if value:
        config.update(load_config(value))
    config.update({k: v for k, v in ctx.params.items() if v is not None})
    ctx.default_map = config
    return config

# Validate Python syntax.
def validate_python_syntax(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# Extract code from LLM response.
def extract_code_from_response(response_text: str) -> str:
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
                start_collecting = True
        if start_collecting:
            if line.lower().startswith("return only the"):
                break
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()

# Format LLM improvements summary.
def format_llm_summary(improvements_summary: dict) -> str:
    unique_improvements = set()
    for improvements in improvements_summary.values():
        if improvements and improvements != ["Error retrieving improvements."]:
            unique_improvements.update(improvements)
    if unique_improvements:
        return "\n".join(f"- {improvement}" for improvement in unique_improvements) + "\n"
    return "No LLM-driven improvements were made.\n"
