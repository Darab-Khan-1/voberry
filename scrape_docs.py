#!/usr/bin/env python3
import asyncio
import logging
import re
from pathlib import Path
from typing import List, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("docs-scraper")

# Load environment variables
load_dotenv()

BASE_URL = "https://farmdar.ai/"
OUTPUT_FILE = Path(__file__).parent / "data/raw_data.txt"
EXCLUDED_PATHS = ["/reference"]


class DocsScraper:
    def __init__(self):
        self.visited_urls: Set[str] = set()
        self.content: List[str] = []
        self.session = None

    async def init_session(self):
        self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()

    def should_exclude_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return any(parsed.path.startswith(path) for path in EXCLUDED_PATHS)

    async def crawl_links(self, url: str):
        """Recursively crawl and scrape all internal links starting from the given URL."""
        if url in self.visited_urls or self.should_exclude_url(url):
            return

        logger.info(f"Crawling {url}")
        self.visited_urls.add(url)

        try:
            async with self.session.get(url) as response:
                if response.status != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                    logger.warning(f"Skipping non-HTML or failed URL: {url}")
                    return

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Try to find <main>, fallback to <body>
                main_content = soup.find("main") or soup.find("body")
                if main_content:
                    for tag in main_content.find_all(["script", "style", "footer", "nav", "header"]):
                        tag.decompose()
                    text = main_content.get_text(separator="\n", strip=True)
                    text = re.sub(r"\n\s*\n", "\n\n", text)
                    if text:
                        logger.info(
                            f"Scraped {len(text)} characters from {url}")
                        self.content.append(
                            f"Content from {url}:\n\n{text}\n\n")
                    else:
                        logger.info(f"No usable text found in {url}")
                else:
                    logger.info(f"No <main> or <body> tag found in {url}")

                # Discover more internal links
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    joined_url = urljoin(BASE_URL, href)
                    parsed_base = urlparse(BASE_URL)
                    parsed_href = urlparse(joined_url)

                    # Stay on same domain
                    if parsed_base.netloc == parsed_href.netloc:
                        await self.crawl_links(joined_url)

        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")

    async def scrape(self):
        await self.init_session()
        try:
            logger.info(f"Starting crawl from {BASE_URL}")
            await self.crawl_links(BASE_URL)
        finally:
            await self.close_session()

    def save_content(self):
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(self.content))
        logger.info(f"Saved content to {OUTPUT_FILE}")


async def main():
    scraper = DocsScraper()
    await scraper.scrape()
    scraper.save_content()


if __name__ == "__main__":
    asyncio.run(main())
