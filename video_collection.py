import os
import requests
from pytube import YouTube, Search
import logging
import time
from urllib.error import HTTPError
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_video_with_retry(url, output_dir, filename, max_retries=3):
    """Attempt to download a video with retries."""
    for attempt in range(max_retries):
        try:
            yt = YouTube(url)
            stream = yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first()
            if stream:
                file_path = stream.download(output_path=output_dir, filename=filename)
                return file_path
            else:
                logger.warning(f"No suitable stream for {yt.title}")
                return None
        except HTTPError as e:
            if e.code == 403:
                logger.warning(f"HTTP 403 on attempt {attempt + 1}/{max_retries} for {url}. Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"HTTP Error {e.code} downloading {url}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None
    logger.error(f"Failed to download {url} after {max_retries} retries")
    return None

def scrape_youtube_urls(query: str, limit: int = 5):
    """Scrape YouTube video URLs as a fallback."""
    search_url = "https://www.youtube.com/results"
    params = {"search_query": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to scrape YouTube: HTTP {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    video_urls = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/watch?v=' in href and len(video_urls) < limit:
            video_urls.append(f"https://youtube.com{href}")
    return video_urls[:limit]

def download_videos(query: str, limit: int = 5, output_dir: str = "data/videos"):
    """Download videos from YouTube based on a search query."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    count = 0
    try:
        # Attempt pytube Search
        search = Search(query)
        videos = search.results[:limit]
        for idx, video in enumerate(videos):
            if count >= limit:
                break
            file_path = download_video_with_retry(
                video.watch_url, 
                output_dir, 
                f"video_{count}.mp4"
            )
            if file_path:
                logger.info(f"Downloaded: {file_path}")
                count += 1
    except Exception as e:
        logger.warning(f"Pytube search failed: {e}. Falling back to scraping...")

    # Fallback to scraping if pytube fails
    if count < limit:
        logger.info("Switching to YouTube URL scraping fallback...")
        video_urls = scrape_youtube_urls(query, limit - count)
        for idx, url in enumerate(video_urls, start=count):
            if count >= limit:
                break
            file_path = download_video_with_retry(
                url, 
                output_dir, 
                f"video_{idx}.mp4"
            )
            if file_path:
                logger.info(f"Downloaded (fallback): {file_path}")
                count += 1

    logger.info(f"Downloaded {count} videos for query '{query}'")

if __name__ == "__main__":
    search_query = "nature documentary short"
    download_limit = 5
    output_directory = "data/videos"
    download_videos(search_query, download_limit, output_directory)