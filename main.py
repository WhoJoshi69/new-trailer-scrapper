from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import re
import asyncio
import aiohttp
import logging
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import time
from supabase import create_client, Client
import os

# Load Supabase credentials from environment or define directly
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dsdfw.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-key")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Movie Trailers API",
    description="Dynamic movie trailer scraper from FirstShowing.net",
    version="1.0.0"
)


# Pydantic models
class TrailerInfo(BaseModel):
    name: str
    youtube_link: str
    poster_url: Optional[str] = None
    source_url: str


class TrailerResponse(BaseModel):
    success: bool
    page: int
    count: int
    trailers: List[TrailerInfo]
    total_pages_scraped: int
    scrape_time_seconds: float


class ErrorResponse(BaseModel):
    success: bool
    error: str
    page: Optional[int] = None


class TrailerScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def get_page_url(self, page: int) -> str:
        """Generate URL for specific page"""
        if page == 1:
            return "https://www.firstshowing.net/category/trailers/"
        else:
            return f"https://www.firstshowing.net/category/trailers/page/{page}/"

    def get_page_content(self, url: str) -> Optional[str]:
        """Fetch page content with error handling"""
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def extract_article_links(self, html_content: str) -> List[str]:
        """Extract article links from category page"""
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        article_links = []

        # Find all article divs
        articles = soup.find_all('div', class_='article')

        for article in articles:
            # Find the title link in h2
            h2_tag = article.find('h2')
            if h2_tag:
                link_tag = h2_tag.find('a', href=True)
                if link_tag:
                    article_links.append(link_tag['href'])

        logger.info(f"Found {len(article_links)} article links")
        return article_links

    def extract_trailer_info(self, article_url: str) -> Optional[TrailerInfo]:
        """Extract trailer information from individual article"""
        html_content = self.get_page_content(article_url)
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')

        try:
            # Extract title
            title_tag = soup.find('h2', id=re.compile(r'^post-\d+'))
            if not title_tag:
                # Fallback to other title selectors
                title_tag = soup.find('h1') or soup.find('h2')

            title = title_tag.get_text().strip() if title_tag else "Unknown Title"

            # Extract YouTube link - multiple strategies
            youtube_link = None

            # Strategy 1: Direct YouTube links in HTML
            youtube_patterns = [
                r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
                r'https?://youtu\.be/[\w-]+'
            ]

            for pattern in youtube_patterns:
                matches = re.findall(pattern, html_content)
                if matches:
                    youtube_link = matches[0]
                    break

            # Strategy 2: Check for embedded YouTube iframes
            if not youtube_link:
                iframe = soup.find('iframe', src=re.compile(r'youtube\.com/embed/'))
                if iframe:
                    embed_src = iframe.get('src', '')
                    video_id_match = re.search(r'/embed/([^?&/]+)', embed_src)
                    if video_id_match:
                        youtube_link = f"https://www.youtube.com/watch?v={video_id_match.group(1)}"

            # Strategy 3: Look for YouTube links in specific sections
            if not youtube_link:
                # Check in source links
                source_links = soup.find_all('a', href=re.compile(r'youtube\.com'))
                if source_links:
                    youtube_link = source_links[0].get('href')

            # Extract poster image
            poster_url = None
            content_images = soup.find_all('img')

            for img in content_images:
                src = img.get('src', '')
                alt = img.get('alt', '').lower()

                # Prioritize images that look like posters
                if src and 'firstshowing.net' in src:
                    if 'poster' in alt or 'trailer' in alt:
                        poster_url = src
                        break

            # Fallback: get first FirstShowing image
            if not poster_url:
                for img in content_images:
                    src = img.get('src', '')
                    if src and 'firstshowing.net' in src and any(
                            ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        poster_url = src
                        break

            # Only return if we have a YouTube link
            if youtube_link:
                return TrailerInfo(
                    name=title,
                    youtube_link=youtube_link,
                    poster_url=poster_url,
                    source_url=article_url
                )
            else:
                logger.warning(f"No YouTube link found for: {title}")

        except Exception as e:
            logger.error(f"Error extracting trailer info from {article_url}: {e}")

        return None

    def scrape_pages(self, pages: List[int]) -> List[TrailerInfo]:
        """Scrape multiple pages and return all trailers"""
        all_article_links = []

        # Collect article links from all requested pages
        for page in pages:
            page_url = self.get_page_url(page)
            logger.info(f"Scraping page {page}: {page_url}")

            html_content = self.get_page_content(page_url)
            if html_content:
                article_links = self.extract_article_links(html_content)
                all_article_links.extend(article_links)

            # Small delay between page requests
            time.sleep(0.5)

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in all_article_links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        logger.info(f"Found {len(unique_links)} unique articles across {len(pages)} pages")

        # Extract trailer info using threading for better performance
        trailers = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.extract_trailer_info, url) for url in unique_links]

            for future in futures:
                try:
                    trailer_info = future.result(timeout=30)
                    if trailer_info:
                        trailers.append(trailer_info)
                        logger.info(f"Successfully scraped: {trailer_info.name}")
                except Exception as e:
                    logger.error(f"Error processing article: {e}")

                # Small delay between requests
                time.sleep(0.3)

        return trailers


# Initialize scraper
scraper = TrailerScraper()


@app.get("/",
         summary="API Information",
         description="Get basic information about the Movie Trailers API")
async def root():
    return {
        "message": "Movie Trailers API",
        "version": "1.0.0",
        "endpoints": {
            "/fetch-trailers": "GET - Fetch trailers from specific pages",
            "/health": "GET - Health check",
            "/docs": "API documentation"
        }
    }
@app.get("/fetch-trailers",
         response_model=TrailerResponse,
         summary="Fetch Movie Trailers",
         description="Dynamically scrape movie trailers from FirstShowing.net based on page numbers")
async def fetch_trailers(
        page: int = Query(1, ge=1, le=50, description="Single page number to scrape (1-50)"),
        pages: Optional[str] = Query(None, description="Comma-separated page numbers (e.g., '1,2,3')"),
        save_to_db: bool = Query(False, description="If true, save results to Supabase")
):
    """
    Fetch movie trailers from FirstShowing.net

    - **page**: Single page number (default: 1)
    - **pages**: Multiple pages as comma-separated string (overrides single page)
    - **save_to_db**: If true, save results to Supabase (default: False)

    Examples:
    - `/fetch-trailers?page=1` - Scrape page 1
    - `/fetch-trailers?pages=1,2,3` - Scrape pages 1, 2, and 3
    - `/fetch-trailers?pages=1,2,3&save_to_db=true` - Scrape pages 1, 2, and 3 and save to Supabase
    """
    start_time = time.time()

    try:
        # Determine which pages to scrape
        if pages:
            try:
                page_list = [int(p.strip()) for p in pages.split(',') if p.strip().isdigit()]
                page_list = [p for p in page_list if 1 <= p <= 50]  # Validate page range
                if not page_list:
                    raise ValueError("No valid page numbers provided")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid pages parameter: {str(e)}")
        else:
            page_list = [page]

        logger.info(f"Scraping pages: {page_list}")

        # Scrape trailers
        trailers = scraper.scrape_pages(page_list)

        # Save to Supabase if requested
        # Save to Supabase if requested
        if save_to_db and trailers:
            logger.info(f"Preparing to save {len(trailers)} trailers to Supabase...")

            # Step 1: Get all existing youtube_links from DB
            try:
                youtube_links = [t.youtube_link for t in trailers]
                existing_response = supabase.table("trailers").select("youtube_link").in_("youtube_link", youtube_links).execute()
                existing_links = set(item['youtube_link'] for item in existing_response.data)
            except Exception as e:
                logger.error(f"Error checking existing records in Supabase: {e}")
                existing_links = set()

            # Step 2: Filter only new trailers
            new_trailers = [t for t in trailers if t.youtube_link not in existing_links]

            if new_trailers:
                insert_data = [{
                    "name": t.name,
                    "youtube_link": t.youtube_link,
                    "poster_url": t.poster_url,
                    "source_url": t.source_url,
                    "is_watched": False
                } for t in new_trailers]

                try:
                    supabase.table("trailers").insert(insert_data).execute()
                    logger.info(f"Inserted {len(insert_data)} new trailers into Supabase.")
                except Exception as e:
                    logger.error(f"Error inserting new trailers into Supabase: {e}")
            else:
                logger.info("No new trailers to insert.")


        scrape_time = time.time() - start_time

        return TrailerResponse(
            success=True,
            page=page_list[0] if len(page_list) == 1 else -1,
            count=len(trailers),
            trailers=trailers,
            total_pages_scraped=len(page_list),
            scrape_time_seconds=round(scrape_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fetch_trailers: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
@app.get("/health",
         summary="Health Check",
         description="Check if the API server is running properly")
async def health_check():
    """Health check endpoint"""
    try:
        # Test a simple request to FirstShowing.net
        test_url = "https://www.firstshowing.net/category/trailers/"
        response = requests.get(test_url, timeout=10)
        external_status = "healthy" if response.status_code == 200 else "degraded"
    except:
        external_status = "unavailable"

    return {
        "status": "healthy",
        "external_site_status": external_status,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)