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
import concurrent.futures
import re
import random

# Initialize Rich console
console = Console()

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"
MAX_RETRIES = 3
RETRY_DELAY = 5


class GitHubAPIError(Exception):
    """Custom exception for GitHub API errors."""
    pass


def make_github_request(url, headers, params=None, method="GET", data=None):
    """Makes a request to the GitHub API, with basic error handling."""
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=headers, data=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise GitHubAPIError(f"Failed to make request to {url} after {MAX_RETRIES} attempts: {e}") from e
            wait_time = RETRY_DELAY * (2 ** attempt)
            console.print(f"[yellow]Request failed: {e}. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})[/yellow]")
            time.sleep(wait_time)

    return None


def get_rate_limit_status(token):
    """Fetches and returns the current rate limit status."""
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "GitHub-Repo-Scraper/1.0"
    }
    try:
        data = make_github_request(f"{GITHUB_API_URL}/rate_limit", headers=headers)
        if data:
            return {
                "core": {
                    "remaining": data["resources"]["core"]["remaining"],
                    "reset": data["resources"]["core"]["reset"]
                },
                "search": {
                    "remaining": data["resources"]["search"]["remaining"],
                    "reset": data["resources"]["search"]["reset"]
                }
            }
    except GitHubAPIError as e:
        console.print(f"[red]Error getting rate limit status: {e}[/red]")
    return None


def search_repositories(token, max_repos):
    """Search for Python repositories, respecting search rate limits."""
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "GitHub-Repo-Scraper/1.0"
    }
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

            # Check search rate limit *before* making a request
            rate_limit = get_rate_limit_status(token)
            if rate_limit and rate_limit["search"]["remaining"] <= 1:
                reset_time = datetime.fromtimestamp(rate_limit["search"]["reset"])
                wait_time = (reset_time - datetime.now()).total_seconds() + 1
                wait_time = max(0, wait_time)
                console.print(f"[yellow]Search rate limit low. Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_time:.0f} seconds)[/yellow]")
                time.sleep(wait_time)


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

            except GitHubAPIError as e:
                console.print(f"[red]Error during repository search: {e}[/red]")
                break
        progress.update(task, completed=len(repos))

    return repos


def get_file_content_and_stats(token, repo_full_name, file_path):
    """Gets file content, line count, and comment ratio."""
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "GitHub-Repo-Scraper/1.0"
    }
    try:
        file_data = make_github_request(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}",
            headers=headers,
        )

        if "content" in file_data:
            decoded_content = base64.b64decode(file_data["content"]).decode("utf-8", errors="replace")
            lines = decoded_content.splitlines()
            num_lines = len(lines)

            comment_lines = 0
            for line in lines:
                if re.match(r"^\s*#", line):
                    comment_lines += 1

            code_lines = num_lines - comment_lines
            if code_lines > 0:
                comment_ratio = (comment_lines / code_lines) * 100
            else:
                comment_ratio = 0.0

            return decoded_content, num_lines, comment_ratio

    except GitHubAPIError as e:
        console.print(f"[red]Error getting file stats for {repo_full_name}/{file_path}: {e}[/red]")
    return "", 0, 0.0


def find_python_files(token, repo_full_name, min_lines, max_lines, quality_threshold):
    """Finds Python files meeting line count and quality criteria."""
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "GitHub-Repo-Scraper/1.0"
    }
    results = []

    try:
        files = make_github_request(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/contents", headers=headers
        )
        if not isinstance(files, list):
            return []

        for file in files:
            if file["type"] == "file" and file["name"].endswith(".py"):
                _, num_lines, comment_ratio = get_file_content_and_stats(token, repo_full_name, file["path"])

                if min_lines <= num_lines <= max_lines and comment_ratio >= quality_threshold:
                    results.append((file["path"], num_lines, comment_ratio))

    except GitHubAPIError as e:
        console.print(f"[red]Error during file search in {repo_full_name}: {e}[/red]")

    return results


def create_unique_filename(base_name, max_repos, min_lines, max_lines, quality_threshold, extension):
    """Creates a unique filename including all filter parameters."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M:%S")
    return f"{timestamp}-{max_repos}repos-Min{min_lines}-Max{max_lines}-Quality{quality_threshold}.{extension}"


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
    return existing_data


def process_repository(token, repo, min_lines, max_lines, quality_threshold, initial_existing_data,
                       processed_files, processed_files_lock, progress, task_id):
    """Processes a single repository, thread-safe, with combined filters and adaptive delays."""
    repo_name = repo["full_name"]
    repo_url = repo["html_url"]
    results = []
    skipped_count = 0

    # Check *core* rate limit before fetching file list
    rate_limit = get_rate_limit_status(token)
    if rate_limit and rate_limit["core"]["remaining"] <= 10:  # More conservative threshold
        reset_time = datetime.fromtimestamp(rate_limit["core"]["reset"])
        wait_time = (reset_time - datetime.now()).total_seconds() + 1
        wait_time = max(0, wait_time)
        console.print(f"[yellow]Core rate limit low. Waiting until {reset_time.strftime('%Y-%m-%d %H:%M:%S')} ({wait_time:.0f} seconds)[/yellow]")
        time.sleep(wait_time)


    python_files = find_python_files(token, repo_name, min_lines, max_lines, quality_threshold)

    for file_path, num_lines, comment_ratio in python_files:
        if (repo_url, file_path) not in initial_existing_data:
            with processed_files_lock:
                if (repo_url, file_path) not in processed_files:
                    results.append({
                        "repo_url": repo_url,
                        "python_file": file_path,
                        "num_lines": num_lines,
                        "comment_ratio": comment_ratio,
                    })
                    processed_files.add((repo_url, file_path))
        else:
            skipped_count += 1

    progress.update(task_id, advance=1)

    # Adaptive delay based on remaining core rate limit
    rate_limit = get_rate_limit_status(token)
    if rate_limit:
        remaining_ratio = rate_limit["core"]["remaining"] / 5000  # Assuming default limit of 5000
        delay = (1 - remaining_ratio) * 2  # Scale delay up to 2 seconds
        delay = max(0.1, delay)  # Ensure minimum delay
        time.sleep(delay)

    return results, skipped_count


def main():
    parser = argparse.ArgumentParser(description="GitHub Python Repo Scraper")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument("--max-repos", type=int, default=10, help="Number of repositories")
    parser.add_argument("--output", default="output", help="Base name for output")
    parser.add_argument("--min-lines", type=int, default=1, help="Minimum number of lines")
    parser.add_argument("--max-lines", type=int, default=100, help="Maximum number of lines")
    parser.add_argument("--quality-threshold", type=float, default=0.0,
                        help="Minimum comment-to-code ratio (percentage)")
    parser.add_argument("--max-workers", type=int, default=5,
                        help="Maximum number of concurrent workers (threads)")
    args = parser.parse_args()

    token = args.token
    max_repos = args.max_repos
    output_base_name = args.output
    min_lines = args.min_lines
    max_lines = args.max_lines
    quality_threshold = args.quality_threshold
    max_workers = args.max_workers

    if min_lines > max_lines:
        console.print("[red]Error: --min-lines cannot be greater than --max-lines[/red]")
        return

    output_file = create_unique_filename(output_base_name, max_repos, min_lines, max_lines, quality_threshold, "json")
    script_directory = os.path.dirname(os.path.abspath(__file__))

    processed_files_lock = threading.Lock()
    processed_files = set()
    existing_data = load_existing_data(script_directory)
    # Check rate limits *before* starting the main loop
    rate_limit_status = get_rate_limit_status(token)
    if rate_limit_status is None:
        console.print("[red]Failed to retrieve initial rate limit status. Exiting.[/red]")
        return

    console.print(f"[green]Initial Rate Limit Status: Core Remaining: {rate_limit_status['core']['remaining']}, Search Remaining: {rate_limit_status['search']['remaining']}[/green]")
    repos = search_repositories(token, max_repos)

    all_results = []
    total_skipped = 0

    with Progress(console=console) as progress:
        task_id = progress.add_task("[green]Processing repositories...", total=len(repos))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for repo in repos:
                initial_existing_data_copy = existing_data.copy()
                futures.append(
                    executor.submit(
                        process_repository, token, repo, min_lines, max_lines, quality_threshold,
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