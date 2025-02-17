import argparse
import requests
import base64
import time
import json
from datetime import datetime, timedelta
from rich.console import Console
from rich.progress import track, Progress
from rich.table import Table
import concurrent.futures

# Initialize Rich console
console = Console()

# GitHub API base URL
GITHUB_API_URL = "https://api.github.com"
MAX_RETRIES = 3  # Maximum number of retries for API requests
RETRY_DELAY = 5  # Seconds to wait between retries


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
                response = requests.post(url, headers=headers, data=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()  # Raises HTTPError for bad requests (4xx or 5xx)
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                raise GitHubAPIError(
                    f"Failed to make request to {url} after {MAX_RETRIES} attempts: {e}"
                ) from e
            console.print(
                f"[yellow]Request failed: {e}. Retrying in {RETRY_DELAY} seconds... (Attempt {attempt + 1}/{MAX_RETRIES})[/yellow]"
            )
            time.sleep(RETRY_DELAY)
    return None  # should never reach here, but for type hinting


def search_repositories(token, max_repos):
    """Search for Python repositories with latest commit older than one year
    and 'main' as the default branch.
    """
    headers = {"Authorization": f"token {token}"}
    query_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    query = f"language:python pushed:<{query_date}"

    repos = []
    page = 1
    per_page = 30  # Maximum items per page allowed by GitHub API

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

                # Filter for repos with 'main' as the default branch
                for item in data["items"]:
                    if item.get("default_branch") == "main":
                        repos.append(item)
                        progress.update(task, advance=1)  # Increment for each *added* repo
                    if len(repos) >= max_repos:
                        break  # Stop once we have enough

                page += 1
                time.sleep(0.5)  # Add small delay

            except GitHubAPIError as e:
                console.print(f"[red]Error during repository search: {e}[/red]")
                break  # Stop searching
        progress.update(task, completed=len(repos)) #ensure the progress reach the total.

    return repos



def find_python_files(token, repo_full_name, quality_threshold):
    """Find Python files in the repository and evaluate their quality."""
    headers = {"Authorization": f"token {token}"}
    results = []

    try:
        files = make_github_request(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/contents", headers=headers
        )
        if not isinstance(files, list):
            # Sometimes github returns a single file, not a list
            return []

        for file in files:
            if file["type"] == "file" and file["name"].endswith(".py"):
                # Perform quality check on the file
                quality_score = evaluate_file_quality(token, repo_full_name, file["path"])
                if quality_score <= quality_threshold:
                    results.append((file["path"], quality_score))

    except GitHubAPIError as e:
        console.print(
            f"[red]Error during file search in {repo_full_name}: {e}[/red]"
        )  # Specific repo error

    return results


def evaluate_file_quality(token, repo_full_name, file_path):
    """Evaluate the quality of a Python file based on simple metrics."""
    headers = {"Authorization": f"token {token}"}
    try:
        file_data = make_github_request(
            f"{GITHUB_API_URL}/repos/{repo_full_name}/contents/{file_path}",
            headers=headers,
        )

        if "content" in file_data:
            decoded_content = base64.b64decode(file_data["content"]).decode(
                "utf-8", errors="replace"
            )  # handle decode errors
            # Example quality metric: Line count
            lines = decoded_content.splitlines()
            quality_score = len(lines)  # Simple metric: total lines in the file
            return quality_score

    except GitHubAPIError as e:
        console.print(
            f"[red]Error during file quality evaluation for {repo_full_name}/{file_path}: {e}[/red]"
        )

    return float("inf")  # Default to poor quality if file cannot be evaluated


def create_unique_filename(base_name, max_repos, quality_threshold, extension):
    """Create a unique filename using the current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # More precise timestamp
    return f"{timestamp}-{max_repos}repos-MaxQuality{quality_threshold}.{extension}"


def process_repository(token, repo, quality_threshold):
    """Process a single repository and return results."""
    repo_name = repo["full_name"]
    console.print(f"[green]Processing repository:[/green] {repo_name}")
    results = []
    python_files = find_python_files(token, repo_name, quality_threshold)
    for file_path, quality_score in python_files:
        results.append(
            {
                "repo_url": repo["html_url"],
                "python_file": file_path,
                "quality_score": quality_score,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="GitHub Python Repo Scraper")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token")
    parser.add_argument(
        "--max-repos", type=int, default=10, help="Number of repositories to retrieve"
    )
    parser.add_argument(
        "--output", default="output", help="Base name for the output JSON file"
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=50,
        help="Maximum allowed quality score for a file to be included",
    )
    args = parser.parse_args()

    token = args.token
    max_repos = args.max_repos
    output_base_name = args.output
    quality_threshold = args.quality_threshold

    output_file = create_unique_filename(
        output_base_name, max_repos, quality_threshold, "json"
    )

    console.print(f"[cyan]Searching for up to {max_repos} Python repositories...[/cyan]")
    repos = search_repositories(token, max_repos)


    all_results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=5
    ) as executor:  # Use a ThreadPool for I/O-bound tasks
        futures = [
            executor.submit(process_repository, token, repo, quality_threshold)
            for repo in repos
        ]
        for future in track(
            concurrent.futures.as_completed(futures),
            total=len(repos),
            description="Processing repositories...",
        ):
            try:
                results = future.result()  # Get results from the completed future
                all_results.extend(results)
            except Exception as e:
                console.print(f"[red]Error processing a repository: {e}[/red]")


    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

    console.print(f"[bold green]Results saved to {output_file}[/bold green]")


if __name__ == "__main__":
    main()