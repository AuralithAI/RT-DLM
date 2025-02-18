import os
import requests
import gzip
import random
from data_utils import DataProcessor

DATA_PATH_TRAIN = os.path.join("data", "train_data.txt")
DATA_PATH_VALIDATION = os.path.join("data", "validation_data.txt")
WIKI_ARTICLES = 1000  
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


def fetch_commoncrawl_data(num_samples=500):
    """Fetches raw text data from Common Crawl."""
    print("[INFO] Fetching Common Crawl dataset paths...")

    url = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2023-40/wet.paths.gz"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to fetch Common Crawl dataset paths.")

    wet_paths = response.text.strip().split("\n")
    if not wet_paths or len(wet_paths[0]) < 10:
        raise Exception("Invalid WET file paths received.")

    wet_file_path = wet_paths[0].strip()  # First available WET file path
    wet_file_url = f"https://data.commoncrawl.org/{wet_file_path}"

    print(f"[INFO] Fetching Common Crawl text data from: {wet_file_url}")

    wet_response = requests.get(wet_file_url, stream=True)
    if wet_response.status_code != 200:
        raise Exception("Failed to fetch Common Crawl text data.")

    raw_texts = []
    with gzip.open(wet_response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith(("WARC", "Content-Length", "Content-Type", "WARC-Target-URI")):
                raw_texts.append(line.strip())

            if len(raw_texts) >= num_samples:
                break

    print(f"[INFO] Successfully fetched {len(raw_texts)} text samples from Common Crawl.")
    return raw_texts


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
    #cc_data = fetch_commoncrawl_data(num_samples=CC_SAMPLES)

    all_data = wiki_data #+ cc_data
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
