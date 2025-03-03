import os
import requests
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import urlencode
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_images(query: str, limit: int = 10, output_dir: str = "data/images"):
    """Download JPEG and PNG images from Google Images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    # Google Images search URL
    search_url = "https://www.google.com/search"
    params = {
        "q": query,
        "tbm": "isch",
        "tbs": "ift:jpg,ift:png"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(search_url, params=params, headers=headers)
    if response.status_code != 200:
        logger.error(f"Failed to fetch images: HTTP {response.status_code}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    img_tags = soup.find_all('img')

    count = 0
    for idx, img in enumerate(img_tags):
        if count >= limit:
            break
        src = img.get('src')
        if not src or not (src.endswith('.jpg') or src.endswith('.png')):
            continue
        
        try:
            if src.startswith('data:'):
                continue
            
            file_ext = '.jpg' if src.endswith('.jpg') else '.png'
            file_path = os.path.join(output_dir, f"{query.replace(' ', '_')}_{idx}{file_ext}")
            
            urllib.request.urlretrieve(src, file_path)
            logger.info(f"Downloaded: {file_path}")
            count += 1
        except Exception as e:
            logger.error(f"Error downloading {src}: {e}")

    logger.info(f"Downloaded {count} images for query '{query}'")

if __name__ == "__main__":
    search_query = "nature scenery"
    download_limit = 10
    output_directory = "data/images"
    download_images(search_query, download_limit, output_directory)