import subprocess
import json
import os
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.text import Text  # Import Text for styled text


def run_fabgpt(repo_url, python_file, github_token, config_file, default_branch="main", console=None, fork_repo=False):
    """Runs fabgpt.py with the given parameters, including forking support."""
    if console is None:
        console = Console()

    try:
        command = [
            "python3",
            "fabgpt.py",
            "--repo", repo_url,
            "--files", python_file,
            "--branch", default_branch,
            "-t", github_token,
            "--config", config_file
        ]

        # Add fork-related options if needed
        if fork_repo:
            command.extend(["--fork-repo"])
            #  We can get the username from the token if needed, but it's better
            #  to have it as separate argument if we're submitting to other's repo
            # command.extend(["--fork-user", <your_github_username>])  # Add if needed

        if not os.path.exists("fabgpt.py"):
            console.print("[red]Error: fabgpt.py not found in the current directory.[/red]")
            return False, "", "fabgpt.py not found"

        console.print(f"[blue]Running command for {python_file}:[/blue] {' '.join(command)}")  # More specific command log
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        console.print(f"[green]fabgpt output for {python_file}:[/green]\n{result.stdout}") #Show output per file.
        if result.stderr:
            console.print(f"[yellow]fabgpt errors for {python_file}:[/yellow]\n{result.stderr}")
        return True, result.stdout, result.stderr

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running fabgpt for {python_file}: {e}[/red]")
        console.print(f"  [red]Return code:[/red] {e.returncode}")
        console.print(f"  [red]Stdout:[/red] {e.stdout}")
        console.print(f"  [red]Stderr:[/red] {e.stderr}")
        return False, e.stdout, e.stderr
    except FileNotFoundError:
        console.print(f"[red]Error: fabgpt.py not found.  Make sure it's in the current directory.[/red]")
        return False, "", "fabgpt.py not found"
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")
        return False, "", str(e)



def main():
    parser = argparse.ArgumentParser(description="Run fabgpt.py on multiple repositories.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file.")
    parser.add_argument("-t", "--token", required=True, help="GitHub Personal Access Token.")
    parser.add_argument("-c", "--config", required=True, help="Path to the config.toml file.")
    parser.add_argument("-b", "--branch", default="main", help="Default branch to use (default: main).")
    parser.add_argument("-o", "--output", help="Path to the output JSON file (optional).")
    parser.add_argument("--fork", action="store_true", help="Fork the repository before making changes.") # Add fork option
    # parser.add_argument("--fork-user", default=None, help="GitHub username for forking (if needed).")  #Optional, we can obtain it from the token.

    args = parser.parse_args()

    console = Console()

    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: Input JSON file '{args.input}' not found.[/red]")
        return
    except json.JSONDecodeError:
        console.print(f"[red]Error: Invalid JSON format in '{args.input}'.[/red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Repo URL", style="dim")
    table.add_column("File")
    table.add_column("Status")
    table.add_column("Output/Error")

    results = []

    for item in track(data, description="Processing repositories...", console=console):
        repo_url = item.get("repo_url")
        python_file = item.get("python_file")

        if not repo_url or not python_file:
            # Use Text for consistent styling within the table
            repo_url_text = Text(str(repo_url))  # Convert to string
            python_file_text = Text(str(python_file))
            table.add_row(repo_url_text, python_file_text, Text("Skipped", style="yellow"), Text("Missing repo_url or python_file"))

            results.append({"repo_url": repo_url, "file": python_file, "status": "Skipped", "output": "Missing repo_url or python_file"})
            continue

        python_file = python_file.replace(" ", "\\ ")

        console.print(f"[cyan]Processing repo: {repo_url}, file: {python_file}[/cyan]")  # Indicate current repo/file


        success, stdout, stderr = run_fabgpt(repo_url, python_file, args.token, args.config, args.branch, console=console, fork_repo=args.fork) #Pass fork_repo

        if success:
            # Use Text for consistent styling within the table
            table.add_row(Text(repo_url), Text(python_file), Text("Success", style="green"), Text(stdout))
            results.append({"repo_url": repo_url, "file": python_file, "status": "Success", "output": stdout})
        else:
            # Use Text for consistent styling
            table.add_row(Text(repo_url), Text(python_file), Text("Failed", style="red"), Text(stderr))
            results.append({"repo_url": repo_url, "file": python_file, "status": "Failed", "output": stderr})


    console.print(table)  # Print the table after processing

    if args.output:
        try:
            with open(args.output, 'w') as outfile:
                json.dump(results, outfile, indent=4)
            console.print(f"[green]Results saved to {args.output}[/green]")
        except Exception as e:
            console.print(f"[red]Error writing to output file: {e}[/red]")

if __name__ == "__main__":
    main()