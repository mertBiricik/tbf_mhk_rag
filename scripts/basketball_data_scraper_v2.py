#!/usr/bin/env python3
"""
TBF Basketball Data Scraper v2.0 - Based on HTML Analysis
"""

import os
import time
import csv
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TBFScraperV2:
    """Improved TBF Basketball Data Scraper using DataTables Excel export."""
    
    def __init__(self):
        """Initialize the scraper."""
        self.setup_directories()
        self.driver = None
        
    def setup_directories(self):
        """Create necessary directories."""
        self.base_dir = Path("./stats")
        self.downloads_dir = self.base_dir / "downloads"
        self.processed_dir = self.base_dir / "processed"
        self.logs_dir = Path("./logs")
        
        for dir_path in [self.downloads_dir, self.processed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_chrome_driver(self):
        """Setup Chrome driver with download preferences."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in background
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Set download directory
        prefs = {
            "download.default_directory": str(self.downloads_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✅ Chrome driver initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize Chrome driver: {e}")
            return False
    
    def wait_for_download(self, timeout=30):
        """Wait for download to complete."""
        logger.info("⏳ Waiting for download to complete...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check for .crdownload files (Chrome partial downloads)
            crdownload_files = list(self.downloads_dir.glob("*.crdownload"))
            if not crdownload_files:
                # Check for Excel files
                excel_files = list(self.downloads_dir.glob("*.xlsx"))
                if excel_files:
                    latest_file = max(excel_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"✅ Download completed: {latest_file.name}")
                    return latest_file
            
            time.sleep(1)
        
        logger.warning(f"⚠️ Download timeout after {timeout} seconds")
        return None
    
    def scrape_league(self, league_name, league_url):
        """Scrape data for a specific league."""
        logger.info(f"🏀 Scraping league: {league_name}")
        logger.info(f"🔗 URL: {league_url}")
        
        try:
            # Navigate to the page
            self.driver.get(league_url)
            time.sleep(5)  # Wait for page load and JS execution
            
            # Wait for DataTable to be initialized
            logger.info("⏳ Waiting for DataTable to load...")
            wait = WebDriverWait(self.driver, 20)
            
            # Wait for the table to be present
            table_locator = (By.ID, "TableMaclar")
            wait.until(EC.presence_of_element_located(table_locator))
            logger.info("✅ DataTable found")
            
            # Try to find the Excel button using the information from HTML analysis
            excel_button = None
            button_strategies = [
                # Based on our analysis - DataTables button with Excel export
                (By.XPATH, "//button[contains(@class, 'buttons-excel')]"),
                # Button with Excel text "Excel'e Aktar"
                (By.XPATH, "//button[contains(text(), \"Excel'e Aktar\")]"),
                # Button with Excel icon fa-file-excel-o
                (By.XPATH, "//button[.//i[contains(@class, 'fa-file-excel-o')]]"),
                # Any button containing "Excel"
                (By.XPATH, "//button[contains(text(), 'Excel')]"),
                # Any button containing "Aktar" (Export)
                (By.XPATH, "//button[contains(text(), 'Aktar')]"),
                # Look for div with DataTables buttons
                (By.XPATH, "//div[contains(@class, 'dt-buttons')]//button[contains(@class, 'buttons-excel')]"),
            ]
            
            for i, (by, selector) in enumerate(button_strategies, 1):
                try:
                    logger.info(f"🔍 Strategy {i}: {selector}")
                    excel_button = wait.until(EC.element_to_be_clickable((by, selector)))
                    logger.info(f"✅ Found Excel button with strategy {i}")
                    break
                except TimeoutException:
                    logger.info(f"⚠️ Strategy {i} failed")
                    continue
            
            if not excel_button:
                logger.error("❌ Could not find Excel export button")
                # Let's debug - find all buttons on the page
                all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
                logger.info(f"🔍 Found {len(all_buttons)} buttons on page")
                for i, btn in enumerate(all_buttons[:10]):  # Show first 10
                    try:
                        text = btn.text.strip()
                        classes = btn.get_attribute("class")
                        if text or "excel" in classes.lower():
                            logger.info(f"Button {i}: text='{text}', class='{classes}'")
                    except:
                        pass
                return None
            
            # Clear downloads directory of old files
            for old_file in self.downloads_dir.glob("*.xlsx"):
                try:
                    old_file.unlink()
                    logger.info(f"🗑️ Removed old file: {old_file.name}")
                except:
                    pass
            
            # Click the Excel export button
            logger.info("📥 Clicking Excel export button...")
            self.driver.execute_script("arguments[0].click();", excel_button)
            
            # Wait for download
            downloaded_file = self.wait_for_download()
            
            if downloaded_file:
                # Rename file with league and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{league_name}_{timestamp}.xlsx"
                new_path = self.downloads_dir / new_filename
                
                downloaded_file.rename(new_path)
                logger.info(f"✅ File renamed to: {new_filename}")
                
                return new_path
            else:
                logger.error("❌ Download failed or timed out")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error scraping {league_name}: {e}")
            return None
    
    def process_excel_to_csv(self, excel_path, league_name):
        """Convert Excel file to CSV with proper encoding."""
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            logger.info(f"📊 Excel file loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Create CSV filename
            csv_filename = excel_path.stem + ".csv"
            csv_path = self.processed_dir / csv_filename
            
            # Save as CSV with UTF-8 encoding
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"💾 CSV saved: {csv_path}")
            
            # Create metadata
            metadata = {
                'league': league_name,
                'scraped_at': datetime.now().isoformat(),
                'excel_file': excel_path.name,
                'csv_file': csv_filename,
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': list(df.columns)
            }
            
            # Save metadata
            metadata_path = self.processed_dir / f"{excel_path.stem}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📋 Metadata saved: {metadata_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"❌ Error processing {excel_path}: {e}")
            return None
    
    def load_leagues(self):
        """Load league configuration from CSV."""
        try:
            leagues_file = self.base_dir / "leagues.csv"
            with open(leagues_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                leagues = [row for row in reader]
            logger.info(f"📋 Loaded {len(leagues)} leagues from configuration")
            return leagues
        except Exception as e:
            logger.error(f"❌ Error loading leagues: {e}")
            return []
    
    def test_single_league(self, league_acronym="BSL"):
        """Test scraping a single league for debugging."""
        if not self.setup_chrome_driver():
            return False
        
        leagues = self.load_leagues()
        test_league = None
        
        for league in leagues:
            if league['acronym'].strip() == league_acronym:
                test_league = league
                break
        
        if not test_league:
            logger.error(f"❌ League {league_acronym} not found")
            return False
        
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"🧪 TESTING LEAGUE: {test_league['acronym']}")
            logger.info(f"{'='*50}")
            
            result = self.scrape_league(test_league['acronym'], test_league['link'])
            
            if result:
                logger.info(f"✅ Test successful! Downloaded: {result}")
                return True
            else:
                logger.error("❌ Test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            return False
        finally:
            if self.driver:
                self.driver.quit()


def main():
    """Main execution function."""
    print("🏀 TBF Basketball Data Scraper v2.0")
    print("Based on HTML Analysis")
    print("=" * 50)
    
    scraper = TBFScraperV2()
    
    # Test with a single league first
    print("🧪 Testing with BSL league...")
    success = scraper.test_single_league("BSL")
    
    if success:
        print("\n🎉 Test successful! Scraper is working.")
    else:
        print("\n⚠️ Test failed. Check logs for details.")

if __name__ == "__main__":
    main() 