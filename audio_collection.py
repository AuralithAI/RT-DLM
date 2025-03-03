import os
import requests
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_audio_files(source_urls: list, limit: int = 10, output_dir: str = "data/audio"):
    """Download audio files from provided URLs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    count = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
    }

    for idx, url in enumerate(source_urls):
        if count >= limit:
            break
        if not url.endswith('.mp3'):
            continue

        try:
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                file_name = f"audio_{idx}.mp3"
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded: {file_path}")
                count += 1
            else:
                logger.warning(f"Failed to download {url}: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")

    logger.info(f"Downloaded {count} audio files")

if __name__ == "__main__":
    audio_urls = [
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Scott_Holmes/Inspirational_Background_Music/Scott_Holmes_-_Inspiring_Corporate.mp3",
        "https://files.freemusicarchive.org/storage-freemusicarchive-org/music/no_curator/Scott_Holmes/Corporate_Uplifting_Music/Scott_Holmes_-_Upbeat_Corporate.mp3",
    ]
    download_limit = 5
    output_directory = "data/audio"
    download_audio_files(audio_urls, download_limit, output_directory)