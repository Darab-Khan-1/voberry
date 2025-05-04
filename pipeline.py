import sys
import asyncio
from pathlib import Path

async def run_pipeline(urls):
    from scrape_docs import main
    from build_rag_data import build_rag

    for url in urls:
        await main(url)
    await build_rag()
if __name__ == "__main__":
    url_file = Path(sys.argv[1])
    with url_file.open() as f:
        urls = [line.strip() for line in f if line.strip()]

    asyncio.run(run_pipeline(urls))
