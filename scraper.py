import argparse
import requests
import base64
import time
import json
import os
import threading
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
import concurrent.futures

# Initialize Rich console
console = Console()

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"
MAX_RETRIES = 3
RETRY_DELAY = 5  # Initial delay, will be adjusted


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    pass


def make_github_request(url, headers, params=None, method="GET", data=None):
    """Makes a request to the GitHub API with retries and rate limit handling."""
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()

            if response.status_code in (403, 429):
                remaining = response.headers.get('X-RateLimit-Remaining')
                reset_timestamp = response.headers.get('X-RateLimit-Reset')

                if remaining == '0' and reset_timestamp:
                    reset_time = datetime.fromtimestamp(int(reset_timestamp))
                    wait_time = (reset_time - datetime.now()).total_seconds() + 1
                    wait_time = max(0, wait_time)
                    console.print(f"[yellow]Rate limit exceeded. Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_time:.0f} seconds)[/yellow]")
                    time.sleep(wait_time)
                    continue
                else:
                    wait_time = 60 * (attempt + 1)  # Exponential backoff
                    console.print(f"[yellow]Rate limit or other error (status {response.status_code}). Waiting {wait_time} seconds.[/yellow]")
                    time.sleep(wait_time)
                    continue

            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise GitHubAPIError(f"Failed to make request to {url} after {MAX_RETRIES} attempts: {e}") from e
            # Use exponential backoff for other request exceptions as well
            wait_time = RETRY_DELAY * (2 ** attempt)  # 5, 10, 20 seconds
            console.print(f"[yellow]Request failed: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})[/yellow]")
            time.sleep(wait_time)

    return None  # Should never reach here


def search_repositories(token, max_repos):
    """Search for Python repositories."""
    headers = {"Authorization": f"token {token}"}
    query_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    query = f"language:python pushed:<{query_date}"

    repos = []
    page = 1
    per_page = 30

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Searching repositories...", total=max_repos)
        while len(repos) < max_repos:
            remaining = max_repos - len(repos)
            current_page_size = min(per_page, remaining)

            try:
                data = make_github_request(
                    f"{GITHUB_API_URL}/search/repositories",
                    headers=headers,
                    params={
                        "q": query,
                        "sort": "updated",
                        "order": "desc",
                        "per_page": current_page_size,
                        "page": page,
                    },
                )

                if "items" not in data or not data["items"]:
                    break

                for item in data["items"]:
                    if item.get("default_branch") == "main":
                        repos.append(item)
                        progress.update(task, advance=1)
                    if len(repos) >= max_repos:
                        break

                page += 1
                time.sleep(0.5)  # Basic delay to avoid immediate rate limiting

            except GitHubAPIError as e:
                console.print(f"[red]Error during repository search: {e}[/red]")
                break
        progress.update(task, completed=len(repos))

    return repos


def find_python_files(token, repo_full_name, quality_threshold):
    """Find Python files in the repository."""
    headers = {"Authorization": f"token {token}"}
    results = []

    try:
        files = make_github_request(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/contents", headers=headers
        )
        if not isinstance(files, list):
            return []

        for file in files:
            if file["type"] == "file" and file["name"].endswith(".py"):
                quality_score = evaluate_file_quality(token, repo_full_name, file["path"])
                if quality_score <= quality_threshold:
                    results.append((file["path"], quality_score))

    except GitHubAPIError as e:
        console.print(f"[red]Error during file search in {repo_full_name}: {e}[/red]")

    return results


def evaluate_file_quality(token, repo_full_name, file_path):
    """Evaluate the quality of a Python file."""
    headers = {"Authorization": f"token {token}"}
    try:
        file_data = make_github_request(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}",
            headers=headers,
        )

        if "content" in file_data:
            decoded_content = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
            lines = decoded_content.splitlines()
            quality_score = len(lines)
            return quality_score

    except GitHubAPIError as e:
        console.print(f"[red]Error during file quality evaluation for {repo_full_name}/{file_path}: {e}[/red]")

    return float("inf")


def create_unique_filename(base_name, max_repos, quality_threshold, extension):
    """Create a unique filename."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{max_repos}repos-MaxQuality{quality_threshold}.{extension}"


def load_existing_data(directory):
    """Loads existing data from JSON files."""
    existing_data = set()
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if "repo_url" in item and "python_file" in item:
                                existing_data.add((item["repo_url"], item["python_file"]))
            except (json.JSONDecodeError, FileNotFoundError) as e:
                console.print(f"[yellow]Warning: Could not read {filename}: {e}[/yellow]")
            except Exception as e:
                console.print(f"[red]Error could not read{filename}: {e}[/red] ")
    return existing_data


def process_repository(token, repo, quality_threshold, initial_existing_data, processed_files, processed_files_lock, progress, task_id):
    """Process a single repository, thread-safe."""
    repo_name = repo["full_name"]
    repo_url = repo["html_url"]
    results = []
    skipped_count = 0
    python_files = find_python_files(token, repo_name, quality_threshold)

    for file_path, quality_score in python_files:
        if (repo_url, file_path) not in initial_existing_data:
            with processed_files_lock:
                if (repo_url, file_path) not in processed_files:
                    results.append({
                        "repo_url": repo_url,
                        "python_file": file_path,
                        "quality_score": quality_score,
                    })
                    processed_files.add((repo_url, file_path))
        else:
            skipped_count += 1

    progress.update(task_id, advance=1)
    return results, skipped_count


def main():
    parser = argparse.ArgumentParser(description="GitHub Python Repo Scraper")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--max-repos", type=int, default=10, help="Number of repositories")
    parser.add_argument("--output", default="output", help="Base name for output")
    parser.add_argument("--quality-threshold", type=int, default=50, help="Max quality score")
    args = parser.parse_args()

    token = args.token
    max_repos = args.max_repos
    output_base_name = args.output
    quality_threshold = args.quality_threshold

    output_file = create_unique_filename(output_base_name, max_repos, quality_threshold, "json")
    script_directory = os.path.dirname(os.path.abspath(__file__))

    processed_files_lock = threading.Lock()
    processed_files = set()
    existing_data = load_existing_data(script_directory)

    repos = search_repositories(token, max_repos)
    all_results = []
    total_skipped = 0

    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Processing repositories...", total=len(repos))
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for repo in repos:
                initial_existing_data_copy = existing_data.copy()
                futures.append(
                    executor.submit(
                        process_repository, token, repo, quality_threshold,
                        initial_existing_data_copy, processed_files,
                        processed_files_lock, progress, task_id
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                try:
                    results, skipped_count = future.result()
                    all_results.extend(results)
                    total_skipped += skipped_count
                except Exception as e:
                    console.print(f"[red]Error processing a repository: {e}[/red]")

    console.print(f"[cyan]{len(all_results)} Items added to JSON[/cyan]")
    console.print(f"[yellow]{total_skipped} Items skipped (already present)[/yellow]")

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    console.print(f"[bold green]Results saved to {output_file}[/bold green]")


if __name__ == "__main__":
    main()