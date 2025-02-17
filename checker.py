import json
import requests
import base64
import argparse
from rich.console import Console
from rich.table import Table
from rich.progress import track
import concurrent.futures
import time
import os

console = Console()

GITHUB_API_URL = "https://api.github.com"
MAX_RETRIES = 3
RETRY_DELAY = 5

class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    pass

def make_github_request(url, headers, params=None, method="GET", data=None):
    """Makes a request to the GitHub API with retries and error handling."""
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=data)  # Corrected: data=data
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()  # Raises HTTPError for bad requests
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise GitHubAPIError(f"Failed to make request to {url} after {MAX_RETRIES} attempts: {e}") from e
            console.print(f"[yellow]Request failed: {e}. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})[/yellow]")
            time.sleep(RETRY_DELAY)
    return None  # Should not be reached, but kept for type hinting and completeness

def evaluate_file_quality(token, repo_url, file_path):
    """Evaluate the quality of a Python file (current version from GitHub)."""
    headers = {"Authorization": f"token {token}"} if token else {}  # Use token if provided
    repo_full_name = repo_url.replace("https://github.com/", "")
    if not repo_full_name:
        console.print(f"[red]Invalid repo URL: {repo_url}[/red]")
        return float('inf'), "Invalid URL"  # Return infinite quality (bad) and reason

    try:
        file_data = make_github_request(f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}", headers=headers)

        if "content" in file_data:
            decoded_content = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
            lines = decoded_content.splitlines()
            quality_score = len(lines)
            return quality_score, "OK" #Success and quality
        else:
            return float('inf'), f"No content found: {file_data.get('message', 'Unknown error')}"

    except GitHubAPIError as e:
        console.print(f"[red]Error during file quality evaluation for {repo_full_name}/{file_path}: {e}[/red]")
        return float('inf'), str(e) #Failed request reason
    except Exception as e:
         console.print(f"[red]Unexpected error during file quality evaluation: {e}[/red]")
         return float('inf'), str(e)

def process_entry(entry, token):
    """Process a single entry from the JSON file and return verification results."""
    repo_url = entry["repo_url"]
    file_path = entry["python_file"]
    original_quality_score = entry["quality_score"]

    current_quality_score, status_message = evaluate_file_quality(token, repo_url, file_path)
    status = "CHANGED" if current_quality_score != original_quality_score else "UNCHANGED"
    if status == "CHANGED" and current_quality_score==float('inf'):
        status = "ERROR"

    return {
        "repo_url": repo_url,
        "python_file": file_path,
        "original_quality_score": original_quality_score,
        "current_quality_score": current_quality_score,
        "status": status,
        "status_message": status_message,  # Add status message
    }

def main():
    parser = argparse.ArgumentParser(description="Verify quality of Python files listed in a JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file.")
    parser.add_argument("--token", help="GitHub Personal Access Token (optional, but recommended for higher rate limits).")
    parser.add_argument("--output", default="verification_results.json", help="Path to save the output JSON results.")
    args = parser.parse_args()

    input_file = args.input_file
    token = args.token
    output_file = args.output
    
    if not os.path.exists(input_file):
        console.print(f"[red]Error: Input file '{input_file}' not found.[/red]")
        return

    try:
        with open(input_file, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        console.print(f"[red]Error: Input file '{input_file}' is not a valid JSON file.[/red]")
        return

    all_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_entry, entry, token) for entry in data]
        for future in track(concurrent.futures.as_completed(futures), total=len(data), description="Verifying files..."):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                console.print(f"[red]Error processing an entry: {e}[/red]")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    # Create and display a Rich table for a summary
    table = Table(title="Verification Summary")
    table.add_column("Repo URL", style="cyan")
    table.add_column("File Path", style="magenta")
    table.add_column("Original Quality", justify="right")
    table.add_column("Current Quality", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Status Message", style="yellow") #add status message


    for result in all_results:
        status_style = "green" if result["status"] == "UNCHANGED" else "red" if result["status"] == "ERROR" else "yellow"
        table.add_row(
            result["repo_url"],
            result["python_file"],
            str(result["original_quality_score"]),
            str(result["current_quality_score"]),
            f"[{status_style}]{result['status']}[/{status_style}]",
            result["status_message"]  # Display status message
        )

    console.print(table)
    console.print(f"[bold green]Verification results saved to {output_file}[/bold green]")

if __name__ == "__main__":
    main()