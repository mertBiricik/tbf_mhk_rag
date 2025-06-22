#!/usr/bin/env python3
"""
Basketball Data Scraper for Turkish Basketball Federation
Downloads league data from TBF website and processes it for RAG system
"""

import os
import sys
import time
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from pathlib import Path
import logging
from datetime import datetime
import csv
import shutil
from typing import Dict, List, Optional
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/basketball_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BasketballDataScraper:
    """Scrapes basketball league data from Turkish Basketball Federation website."""
    
    def __init__(self, download_dir: str = "./stats/downloads", processed_dir: str = "./stats/processed"):
        """Initialize the scraper with directories."""
        self.download_dir = Path(download_dir)
        self.processed_dir = Path(processed_dir)
        self.leagues_file = Path("./stats/leagues.csv")
        
        # Create directories
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Browser setup
        self.driver = None
        self.setup_browser()
        
        # Load league information
        self.leagues = self.load_leagues()
        
        logger.info(f"✅ Basketball Data Scraper initialized")
        logger.info(f"   Download dir: {self.download_dir}")
        logger.info(f"   Processed dir: {self.processed_dir}")
        logger.info(f"   Leagues loaded: {len(self.leagues)}")
    
    def setup_browser(self):
        """Setup Chrome browser with download preferences."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Set download directory
        prefs = {
            "download.default_directory": str(self.download_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✅ Chrome browser initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    def load_leagues(self) -> List[Dict]:
        """Load league information from CSV file."""
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
            logger.info(f"✅ Loaded {len(leagues)} leagues from {self.leagues_file}")
            return leagues
        except Exception as e:
            logger.error(f"❌ Failed to load leagues: {e}")
            return []
    
    def download_league_data(self, league: Dict, timeout: int = 30) -> Optional[Path]:
        """Download Excel data for a specific league."""
        logger.info(f"🏀 Downloading data for {league['acronym'].upper()} - {league['name']}")
        
        try:
            # Navigate to league page
            self.driver.get(league['link'])
            time.sleep(3)  # Wait for page to load
            
            # Look for Excel download button - try multiple selectors
            download_selectors = [
                "//a[contains(text(), 'Excel')]",
                "//button[contains(text(), 'Excel')]", 
                "//a[contains(@href, 'excel')]",
                "//button[contains(@class, 'excel')]",
                "//a[contains(text(), 'İndir')]",  # Turkish for "Download"
                "//button[contains(text(), 'İndir')]",
                "//a[contains(@title, 'Excel')]",
                "//button[contains(@title, 'Excel')]"
            ]
            
            download_button = None
            for selector in download_selectors:
                try:
                    download_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    logger.info(f"   Found download button with selector: {selector}")
                    break
                except TimeoutException:
                    continue
            
            if not download_button:
                logger.error(f"   ❌ Could not find download button for {league['acronym']}")
                return None
            
            # Get current files in download directory before download
            files_before = set(self.download_dir.glob("*.xlsx"))
            
            # Click download button
            download_button.click()
            logger.info(f"   📥 Download initiated for {league['acronym']}")
            
            # Wait for download to complete
            max_wait = timeout
            wait_interval = 1
            elapsed = 0
            
            while elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                # Check for new Excel files
                files_after = set(self.download_dir.glob("*.xlsx"))
                new_files = files_after - files_before
                
                if new_files:
                    downloaded_file = list(new_files)[0]
                    logger.info(f"   ✅ Download completed: {downloaded_file.name}")
                    return downloaded_file
            
            logger.error(f"   ❌ Download timeout for {league['acronym']} after {timeout}s")
            return None
            
        except Exception as e:
            logger.error(f"   ❌ Error downloading {league['acronym']}: {e}")
            return None
    
    def rename_and_organize_file(self, file_path: Path, league: Dict) -> Path:
        """Rename downloaded file with proper naming convention."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create new filename: acronym_timestamp.xlsx
        new_filename = f"{league['acronym']}_{timestamp}.xlsx"
        new_path = self.download_dir / new_filename
        
        try:
            # Rename the file
            shutil.move(str(file_path), str(new_path))
            logger.info(f"   📝 Renamed to: {new_filename}")
            
            # Also create a "latest" symlink for easy access
            latest_link = self.download_dir / f"{league['acronym']}_latest.xlsx"
            if latest_link.exists():
                latest_link.unlink()
            
            # Create symlink to latest file
            try:
                latest_link.symlink_to(new_path.name)
                logger.info(f"   🔗 Created latest link: {latest_link.name}")
            except OSError:
                # Symlinks might not work on all systems, create a copy instead
                shutil.copy2(str(new_path), str(latest_link))
                logger.info(f"   📄 Created latest copy: {latest_link.name}")
            
            return new_path
            
        except Exception as e:
            logger.error(f"   ❌ Error renaming file: {e}")
            return file_path
    
    def excel_to_csv(self, excel_path: Path, league: Dict) -> Optional[Path]:
        """Convert Excel file to CSV format."""
        logger.info(f"📊 Converting {excel_path.name} to CSV")
        
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            
            # Create CSV filename
            csv_filename = excel_path.stem + ".csv"
            csv_path = self.processed_dir / csv_filename
            
            # Save as CSV with UTF-8 encoding
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Add metadata
            metadata = {
                'league': league,
                'file_info': {
                    'original_excel': str(excel_path),
                    'csv_file': str(csv_path),
                    'processed_at': datetime.now().isoformat(),
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()
                }
            }
            
            # Save metadata
            metadata_path = csv_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"   ✅ CSV created: {csv_path.name} ({len(df)} rows, {len(df.columns)} columns)")
            logger.info(f"   📋 Metadata saved: {metadata_path.name}")
            
            return csv_path
            
        except Exception as e:
            logger.error(f"   ❌ Error converting to CSV: {e}")
            return None
    
    def scrape_league(self, league: Dict) -> Dict:
        """Scrape data for a single league."""
        result = {
            'league': league['acronym'],
            'success': False,
            'excel_file': None,
            'csv_file': None,
            'error': None
        }
        
        try:
            # Download Excel file
            excel_file = self.download_league_data(league)
            if not excel_file:
                result['error'] = "Failed to download Excel file"
                return result
            
            # Rename and organize
            organized_file = self.rename_and_organize_file(excel_file, league)
            result['excel_file'] = str(organized_file)
            
            # Convert to CSV
            csv_file = self.excel_to_csv(organized_file, league)
            if csv_file:
                result['csv_file'] = str(csv_file)
                result['success'] = True
            else:
                result['error'] = "Failed to convert to CSV"
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error scraping {league['acronym']}: {e}")
        
        return result
    
    def scrape_all_leagues(self, delay_between_requests: float = 5.0) -> Dict:
        """Scrape data for all leagues."""
        logger.info(f"🚀 Starting to scrape {len(self.leagues)} leagues")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_leagues': len(self.leagues),
            'successful': 0,
            'failed': 0,
            'details': []
        }
        
        for i, league in enumerate(self.leagues):
            logger.info(f"📊 Processing league {i+1}/{len(self.leagues)}: {league['acronym'].upper()}")
            
            result = self.scrape_league(league)
            results['details'].append(result)
            
            if result['success']:
                results['successful'] += 1
                logger.info(f"   ✅ {league['acronym'].upper()} completed successfully")
            else:
                results['failed'] += 1
                logger.error(f"   ❌ {league['acronym'].upper()} failed: {result['error']}")
            
            # Add delay between requests to be respectful
            if i < len(self.leagues) - 1:  # Don't wait after the last league
                logger.info(f"   ⏳ Waiting {delay_between_requests}s before next league...")
                time.sleep(delay_between_requests)
        
        # Save summary report
        summary_file = self.processed_dir / f"scraping_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🎉 Scraping completed!")
        logger.info(f"   ✅ Successful: {results['successful']}/{results['total_leagues']}")
        logger.info(f"   ❌ Failed: {results['failed']}/{results['total_leagues']}")
        logger.info(f"   📊 Summary saved: {summary_file.name}")
        
        return results
    
    def cleanup(self):
        """Clean up resources."""
        if self.driver:
            self.driver.quit()
            logger.info("🔧 Browser cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def main():
    """Main function to run the scraper."""
    logger.info("🏀 Basketball Data Scraper Starting...")
    
    try:
        with BasketballDataScraper() as scraper:
            results = scraper.scrape_all_leagues(delay_between_requests=3.0)
            
            print("\n" + "="*60)
            print("🏀 BASKETBALL DATA SCRAPING SUMMARY")
            print("="*60)
            print(f"📊 Total Leagues: {results['total_leagues']}")
            print(f"✅ Successful: {results['successful']}")
            print(f"❌ Failed: {results['failed']}")
            print(f"📅 Completed: {results['timestamp']}")
            
            if results['successful'] > 0:
                print(f"\n📁 Files saved to:")
                print(f"   Excel: ./stats/downloads/")
                print(f"   CSV: ./stats/processed/")
            
            print("="*60)
            
    except KeyboardInterrupt:
        logger.info("🛑 Scraping interrupted by user")
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 