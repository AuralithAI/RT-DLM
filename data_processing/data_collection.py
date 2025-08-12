import os
import requests
import time
import random
from data_processing.data_utils import DataProcessor

DATA_PATH_TRAIN = os.path.join("data", "train_data.txt")
DATA_PATH_VALIDATION = os.path.join("data", "validation_data.txt")
WIKI_ARTICLES = 500000  
ARXIV_DATA = 500000
GITHUB_READMES = 500000
REDDIT_POSTS = 500000
CC_SAMPLES = 500  
VALIDATION_SPLIT = 0.2  

def fetch_wikipedia_data(num_articles=500):
    """Fetches Wikipedia articles dynamically from Wikipedia API."""
    print("[INFO] Fetching Wikipedia dataset...")

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnlimit": num_articles,
        "rnnamespace": 0 
    }

    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception("Failed to fetch Wikipedia data")

    data = response.json()
    articles = []

    for page in data["query"]["random"]:
        title = page["title"]
        content_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + title.replace(" ", "_")
        content_response = requests.get(content_url)

        if content_response.status_code == 200:
            content_data = content_response.json()
            if "extract" in content_data:
                articles.append(content_data["extract"])

    print(f"[INFO] Fetched {len(articles)} Wikipedia articles.")
    return articles

import time

def fetch_arxiv_data(num_papers=500000, batch_size=2000):
    """Fetch abstracts from arXiv in batches."""
    print("[INFO] Fetching arXiv dataset in batches...")
    
    all_abstracts = []
    base_url = "http://export.arxiv.org/api/query?search_query=cat:cs.AI"

    for start in range(0, num_papers, batch_size):
        max_results = min(batch_size, num_papers - len(all_abstracts))
        url = f"{base_url}&start={start}&max_results={max_results}"
        
        retries = 3  # Retry up to 3 times if fetching fails
        while retries > 0:
            response = requests.get(url)
            if response.status_code == 200:
                abstracts = []
                for entry in response.text.split("<summary>")[1:]:
                    abstract = entry.split("</summary>")[0].strip()
                    abstracts.append(abstract)

                if abstracts:
                    all_abstracts.extend(abstracts)
                    print(f"[INFO] Collected {len(all_abstracts)}/{num_papers} abstracts so far...")
                    break  # Exit retry loop if successful
                else:
                    print("[ERROR] No <summary> tags found in response, retrying...")
                    retries -= 1
                    time.sleep(3)  # Wait before retrying
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}, retrying...")
                retries -= 1
                time.sleep(3)

        if retries == 0:
            print("[ERROR] Max retries reached. Stopping arXiv data collection.")
            break

    print(f"[INFO] Successfully fetched {len(all_abstracts)} abstracts from arXiv.")
    return all_abstracts

def fetch_github_readmes(num_repos=200):
    """Fetches README.md files from trending AI/ML repositories on GitHub."""
    print("[INFO] Fetching GitHub dataset...")
    url = "https://api.github.com/search/repositories?q=machine+learning&sort=stars&order=desc&per_page=" + str(num_repos)
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception("Failed to fetch GitHub repositories")
    
    repos = response.json().get("items", [])
    readmes = []
    for repo in repos:
        readme_url = repo.get("html_url") + "/blob/main/README.md"
        readmes.append(readme_url)
    
    print(f"[INFO] Fetched {len(readmes)} GitHub README.md URLs.")
    return readmes


def fetch_reddit_posts(subreddit="artificial", num_posts=500, batch_size=10):
    """Fetches top posts from a subreddit in batches."""
    print(f"[INFO] Fetching {num_posts} posts from r/{subreddit} in batches of {batch_size}...")

    all_posts = []
    last_post_time = None  # For pagination

    while len(all_posts) < num_posts:
        remaining = num_posts - len(all_posts)
        fetch_count = min(batch_size, remaining)

        url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size={fetch_count}"
        if last_post_time:
            url += f"&before={last_post_time}"

        response = requests.get(url)

        if response.status_code != 200:
            print(f"[ERROR] HTTP {response.status_code}: {response.text}")
            raise Exception("Failed to fetch Reddit data")

        data = response.json().get("data", [])
        if not data:
            print("[ERROR] No more posts found, stopping early.")
            break

        for post in data:
            text = post.get("title", "") + " " + post.get("selftext", "")
            if text.strip():
                all_posts.append(text)

        last_post_time = data[-1]["created_utc"]  # Get timestamp of the last post
        print(f"[INFO] Collected {len(all_posts)}/{num_posts} posts so far...")

        time.sleep(1)  # Avoid hitting rate limits

    print(f"[INFO] Successfully fetched {len(all_posts)} Reddit posts from r/{subreddit}.")
    return all_posts

def save_datasets(train_data, val_data):
    """Saves training and validation datasets to separate files."""
    os.makedirs("data", exist_ok=True)

    with open(DATA_PATH_TRAIN, "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(line + "\n")
    print(f"[INFO] Training dataset saved to {DATA_PATH_TRAIN}")

    with open(DATA_PATH_VALIDATION, "w", encoding="utf-8") as f:
        for line in val_data:
            f.write(line + "\n")
    print(f"[INFO] Validation dataset saved to {DATA_PATH_VALIDATION}")


if __name__ == "__main__":
    processor = DataProcessor()
    
    print("[INFO] Collecting dataset...")
    wiki_data = fetch_wikipedia_data(num_articles=WIKI_ARTICLES)
    arxiv_data = fetch_arxiv_data(num_papers=ARXIV_DATA, batch_size=2000)
    github_data = fetch_github_readmes(num_repos=GITHUB_READMES)
    #reddit_data = fetch_reddit_posts(subreddit="machinelearning", num_posts=REDDIT_POSTS, batch_size=1000)
    
    # Combine all sources
    all_data = wiki_data + arxiv_data + github_data #+ reddit_data
    all_data = [processor.preprocess_text(text) for text in all_data]
    
    # Shuffle data
    random.shuffle(all_data)
    
    # Split into train and validation sets
    split_idx = int(len(all_data) * (1 - VALIDATION_SPLIT))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # Save datasets
    save_datasets(train_data, val_data)
    print("[INFO] Data collection and validation dataset creation completed successfully.")
