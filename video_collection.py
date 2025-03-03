import os
from pytube import YouTube, Search
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_videos(query: str, limit: int = 5, output_dir: str = "data/videos"):
    """Download videos from YouTube based on a search query."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created directory: {output_dir}")

    search = Search(query)
    videos = search.results[:limit]
    count = 0

    for video in videos:
        try:
            yt = YouTube(video.watch_url)
            stream = yt.streams.filter(file_extension='mp4', progressive=True).order_by('resolution').desc().first()
            if stream:
                file_path = stream.download(output_path=output_dir, filename=f"video_{count}.mp4")
                logger.info(f"Downloaded: {file_path}")
                count += 1
            else:
                logger.warning(f"No suitable stream for {yt.title}")
        except Exception as e:
            logger.error(f"Error downloading {video.watch_url}: {e}")

    logger.info(f"Downloaded {count} videos for query '{query}'")

if __name__ == "__main__":
    search_query = "nature documentary short"
    download_limit = 5
    output_directory = "data/videos"
    download_videos(search_query, download_limit, output_directory)