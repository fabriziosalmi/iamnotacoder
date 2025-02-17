import argparse
import requests
import base64
import time
import json
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Initialize Rich console
console = Console()

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"

def search_repositories(token, max_repos):
    """Search for Python repositories with latest commit older than one year."""
    headers = {"Authorization": f"token {token}"}
    query_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    query = f"language:python pushed:<{query_date}"

    repos = []
    page = 1
    per_page = 30  # Maximum items per page allowed by GitHub API

    while len(repos) < max_repos:
        remaining = max_repos - len(repos)
        current_page_size = min(per_page, remaining)

        response = requests.get(
            f"{GITHUB_API_URL}/search/repositories",
            headers=headers,
            params={"q": query, "sort": "updated", "order": "desc", "per_page": current_page_size, "page": page},
        )

        if response.status_code != 200:
            console.print(f"[red]Error:[/red] {response.status_code} - {response.json().get('message')}")
            break

        data = response.json()
        if "items" not in data or not data["items"]:
            break

        repos.extend(data["items"])
        page += 1
        time.sleep(1)  # To avoid hitting API rate limits

    return repos[:max_repos]

def find_python_files(token, repo_full_name, quality_threshold):
    """Find Python files in the repository and evaluate their quality."""
    headers = {"Authorization": f"token {token}"}
    response = requests.get(f"{GITHUB_API_URL}/repos/{repo_full_name}/contents", headers=headers)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["type"] == "file" and file["name"].endswith(".py"):
                # Perform quality check on the file
                quality_score = evaluate_file_quality(token, repo_full_name, file["path"])
                if quality_score <= quality_threshold:
                    return file["path"], quality_score

    return None, None

def evaluate_file_quality(token, repo_full_name, file_path):
    """Evaluate the quality of a Python file based on simple metrics."""
    headers = {"Authorization": f"token {token}"}
    response = requests.get(f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}", headers=headers)

    if response.status_code == 200:
        file_content = response.json().get("content")
        if file_content:
            decoded_content = base64.b64decode(file_content).decode("utf-8")
            # Example quality metric: Line count
            lines = decoded_content.splitlines()
            quality_score = len(lines)  # Simple metric: total lines in the file
            return quality_score

    return float("inf")  # Default to poor quality if file cannot be evaluated

def create_unique_filename(base_name, max_repos, quality_threshold, extension):
    """Create a unique filename using the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    return f"{timestamp}-{max_repos}repos-MaxQuality{quality_threshold}.{extension}"

def main():
    parser = argparse.ArgumentParser(description="GitHub Python Repo Scraper")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--max-repos", type=int, default=10, help="Number of repositories to retrieve")
    parser.add_argument("--output", default="output", help="Base name for the output JSON file")
    parser.add_argument("--quality-threshold", type=int, default=50, help="Maximum allowed quality score for a file to be included")
    args = parser.parse_args()

    token = args.token
    max_repos = args.max_repos
    output_base_name = args.output
    quality_threshold = args.quality_threshold

    output_file = create_unique_filename(output_base_name, max_repos, quality_threshold, "json")

    console.print(f"[cyan]Searching for up to {max_repos} Python repositories...[/cyan]")
    repos = search_repositories(token, max_repos)

    results = []
    for repo in track(repos, description="Processing repositories..."):
        repo_name = repo["full_name"]
        console.print(f"[green]Processing repository:[/green] {repo_name}")

        python_file, quality_score = find_python_files(token, repo_name, quality_threshold)
        if python_file:
            results.append({"repo_url": repo["html_url"], "python_file": python_file, "quality_score": quality_score})

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    console.print(f"[bold green]Results saved to {output_file}[/bold green]")

if __name__ == "__main__":
    main()
