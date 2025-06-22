#!/usr/bin/env python3
"""
Basketball Data Scraper using BeautifulSoup + Requests
Lighter and faster alternative to Selenium
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import csv
import shutil
import json
from urllib.parse import urljoin
import re
import time
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BasketballScraperBS4:
    """BeautifulSoup-based scraper for TBF basketball data."""
    
    def __init__(self, download_dir: str = "./stats/downloads", processed_dir: str = "./stats/processed"):
        self.download_dir = Path(download_dir)
        self.processed_dir = Path(processed_dir)
        self.leagues_file = Path("./stats/leagues.csv")
        
        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.leagues = self.load_leagues()
        logger.info(f"✅ BeautifulSoup scraper initialized with {len(self.leagues)} leagues")
    
    def load_leagues(self) -> List[Dict]:
        """Load league information from CSV."""
        leagues = []
        try:
            with open(self.leagues_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    leagues.append({
                        'link': row['link'].strip(),
                        'acronym': row['acronym'].strip(),
                        'name': row['name_of_the_league'].strip(),
                        'gender': row['gender'].strip(),
                        'sponsor': row['sponsor'].strip(),
                        'season': row['season'].strip()
                    })
            return leagues
        except Exception as e:
            logger.error(f"❌ Failed to load leagues: {e}")
            return []
    
    def find_excel_download_url(self, page_url: str) -> Optional[str]:
        """Find Excel download URL on the page."""
        try:
            logger.info(f"🔍 Searching for Excel download on: {page_url}")
            
            response = self.session.get(page_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Search patterns for Excel downloads
            patterns = [
                # Direct links to Excel files
                'a[href*=".xlsx"]',
                'a[href*=".xls"]',
                'a[href*="excel"]',
                'a[href*="Excel"]',
                # Turkish download terms
                'a[href*="indir"]',
                'a[title*="Excel"]',
                'a[title*="excel"]',
                # Button patterns
                'button[onclick*="excel"]',
                'input[onclick*="excel"]'
            ]
            
            for pattern in patterns:
                elements = soup.select(pattern)
                for element in elements:
                    href = element.get('href')
                    if href and isinstance(href, str):
                        full_url = urljoin(page_url, href)
                        logger.info(f"   🎯 Found Excel link: {full_url}")
                        return full_url
                    
                    # Check onclick for JavaScript downloads
                    onclick = element.get('onclick')
                    if onclick and 'excel' in str(onclick).lower():
                        # Try to extract URL from onclick
                        url_match = re.search(r"['\"]([^'\"]*\.xlsx?[^'\"]*)['\"]", str(onclick))
                        if url_match:
                            url = url_match.group(1)
                            full_url = urljoin(page_url, url)
                            logger.info(f"   🎯 Found Excel URL in onclick: {full_url}")
                            return full_url
            
            # Try text-based search
            excel_links = soup.find_all('a', string=re.compile(r'[Ee]xcel|İndir|Download'))
            for link in excel_links:
                if link.get('href'):
                    href = link.get('href')
                    full_url = urljoin(page_url, href)
                    logger.info(f"   🎯 Found text-based Excel link: {full_url}")
                    return full_url
            
            logger.warning(f"   ⚠️  No Excel download found on {page_url}")
            return None
            
        except Exception as e:
            logger.error(f"   ❌ Error searching for Excel link: {e}")
            return None
    
    def download_excel_file(self, url: str, league: Dict) -> Optional[Path]:
        """Download Excel file from URL."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{league['acronym']}_{timestamp}.xlsx"
            file_path = self.download_dir / filename
            
            logger.info(f"📥 Downloading {filename} from {url}")
            
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Save file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify download
            if file_path.exists() and file_path.stat().st_size > 0:
                size = file_path.stat().st_size
                logger.info(f"   ✅ Downloaded: {filename} ({size:,} bytes)")
                
                # Create latest link
                latest_link = self.download_dir / f"{league['acronym']}_latest.xlsx"
                if latest_link.exists():
                    latest_link.unlink()
                
                try:
                    latest_link.symlink_to(filename)
                except OSError:
                    shutil.copy2(file_path, latest_link)
                
                return file_path
            else:
                logger.error(f"   ❌ Download failed or file empty")
                return None
                
        except Exception as e:
            logger.error(f"   ❌ Download error: {e}")
            return None
    
    def excel_to_csv(self, excel_path: Path, league: Dict) -> Optional[Path]:
        """Convert Excel to CSV."""
        try:
            logger.info(f"📊 Converting {excel_path.name} to CSV")
            
            df = pd.read_excel(excel_path)
            csv_filename = excel_path.stem + ".csv"
            csv_path = self.processed_dir / csv_filename
            
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Create metadata
            metadata = {
                'league': league,
                'file_info': {
                    'excel_file': str(excel_path),
                    'csv_file': str(csv_path),
                    'processed_at': datetime.now().isoformat(),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()
                }
            }
            
            metadata_path = csv_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"   ✅ CSV: {csv_path.name} ({len(df)} rows, {len(df.columns)} cols)")
            return csv_path
            
        except Exception as e:
            logger.error(f"   ❌ CSV conversion error: {e}")
            return None
    
    def scrape_league(self, league: Dict) -> Dict:
        """Scrape a single league."""
        result = {
            'league': league['acronym'],
            'success': False,
            'excel_file': None,
            'csv_file': None,
            'download_url': None,
            'error': None
        }
        
        try:
            logger.info(f"🏀 Processing {league['acronym'].upper()} - {league['name']}")
            
            # Find download URL
            download_url = self.find_excel_download_url(league['link'])
            if not download_url:
                result['error'] = "Excel download URL not found"
                return result
            
            result['download_url'] = download_url
            
            # Download Excel
            excel_file = self.download_excel_file(download_url, league)
            if not excel_file:
                result['error'] = "Excel download failed"
                return result
            
            result['excel_file'] = str(excel_file)
            
            # Convert to CSV
            csv_file = self.excel_to_csv(excel_file, league)
            if csv_file:
                result['csv_file'] = str(csv_file)
                result['success'] = True
            else:
                result['error'] = "CSV conversion failed"
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error processing {league['acronym']}: {e}")
        
        return result
    
    def scrape_all_leagues(self, delay: float = 2.0) -> Dict:
        """Scrape all leagues."""
        logger.info(f"🚀 Starting BS4 scraping of {len(self.leagues)} leagues")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'BeautifulSoup + Requests',
            'total_leagues': len(self.leagues),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for i, league in enumerate(self.leagues):
            logger.info(f"📊 League {i+1}/{len(self.leagues)}: {league['acronym'].upper()}")
            
            result = self.scrape_league(league)
            results['details'].append(result)
            
            if result['success']:
                results['successful'] += 1
                logger.info(f"   ✅ {league['acronym'].upper()} completed")
            else:
                results['failed'] += 1
                logger.error(f"   ❌ {league['acronym'].upper()} failed: {result['error']}")
            
            if i < len(self.leagues) - 1:
                logger.info(f"   ⏳ Waiting {delay}s...")
                time.sleep(delay)
        
        # Save summary
        summary_file = self.processed_dir / f"bs4_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🎉 BS4 scraping completed: {results['successful']}/{results['total_leagues']}")
        return results
    
    def close(self):
        """Close session."""
        self.session.close()

def main():
    """Main function."""
    logger.info("🏀 BeautifulSoup Basketball Scraper Starting...")
    
    try:
        scraper = BasketballScraperBS4()
        results = scraper.scrape_all_leagues(delay=2.0)
        
        print("\n" + "="*50)
        print("🏀 BS4 SCRAPING SUMMARY")
        print("="*50)
        print(f"✅ Successful: {results['successful']}/{results['total_leagues']}")
        print(f"❌ Failed: {results['failed']}")
        print(f"⚡ Method: BeautifulSoup + Requests")
        print(f"📁 Files in: ./stats/downloads/ & ./stats/processed/")
        print("="*50)
        
        scraper.close()
        
    except Exception as e:
        logger.error(f"💥 Error: {e}")

if __name__ == "__main__":
    main() 