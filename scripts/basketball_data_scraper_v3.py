#!/usr/bin/env python3
"""
TBF Basketball Data Scraper v3.0 - JavaScript-aware version
"""

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
from selenium.common.exceptions import TimeoutException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TBFScraperV3:
    """JavaScript-aware TBF Basketball Data Scraper."""
    
    def __init__(self):
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
        """Setup Chrome driver."""
        chrome_options = Options()
        # Keep browser visible for debugging
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
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
            logger.info("✅ Chrome driver initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Chrome driver failed: {e}")
            return False
    
    def wait_for_download(self, timeout=30):
        """Wait for download to complete."""
        logger.info("⏳ Waiting for download...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            crdownload_files = list(self.downloads_dir.glob("*.crdownload"))
            if not crdownload_files:
                excel_files = list(self.downloads_dir.glob("*.xlsx"))
                if excel_files:
                    latest_file = max(excel_files, key=lambda f: f.stat().st_mtime)
                    logger.info(f"✅ Download completed: {latest_file.name}")
                    return latest_file
            time.sleep(1)
        
        logger.warning(f"⚠️ Download timeout")
        return None
    
    def scrape_league(self, league_name, league_url):
        """Scrape data for a specific league."""
        logger.info(f"🏀 Scraping: {league_name}")
        logger.info(f"🔗 URL: {league_url}")
        
        try:
            # Navigate and wait
            self.driver.get(league_url)
            logger.info("📄 Page loaded, waiting for JavaScript...")
            time.sleep(10)  # Wait for JS execution
            
            # Wait for table
            wait = WebDriverWait(self.driver, 30)
            table_locator = (By.ID, "TableMaclar")
            wait.until(EC.presence_of_element_located(table_locator))
            logger.info("✅ DataTable found")
            
            # Wait for buttons to be generated
            time.sleep(5)
            
            # Try JavaScript approach directly
            logger.info("🎯 Attempting JavaScript Excel export...")
            
            # Clear old downloads
            for old_file in self.downloads_dir.glob("*.xlsx"):
                try:
                    old_file.unlink()
                except:
                    pass
            
            js_code = """
            try {
                // Get the DataTable instance
                var table = $('#TableMaclar').DataTable();
                
                if (!table) {
                    return {success: false, error: 'DataTable not found'};
                }
                
                // Get all buttons
                var buttonCount = 0;
                try {
                    var buttons = table.buttons();
                    buttonCount = buttons.length || 0;
                } catch(e) {
                    return {success: false, error: 'No buttons method: ' + e.message};
                }
                
                // Try to find Excel button
                for (var i = 0; i < buttonCount; i++) {
                    try {
                        var button = table.button(i);
                        var text = button.text() || '';
                        
                        if (text.toLowerCase().includes('excel') || text.toLowerCase().includes('aktar')) {
                            button.trigger();
                            return {success: true, message: 'Excel button found and clicked at index ' + i + ': ' + text};
                        }
                    } catch(e) {
                        // Continue to next button
                    }
                }
                
                // Try direct class selector
                try {
                    var excelBtn = table.button('.buttons-excel');
                    if (excelBtn && excelBtn.length > 0) {
                        excelBtn.trigger();
                        return {success: true, message: 'Excel button triggered via .buttons-excel'};
                    }
                } catch(e) {
                    // Continue
                }
                
                // Try excelHtml5 specifically
                try {
                    var excelBtn = table.button('.buttons-excelHtml5');
                    if (excelBtn && excelBtn.length > 0) {
                        excelBtn.trigger();
                        return {success: true, message: 'Excel button triggered via .buttons-excelHtml5'};
                    }
                } catch(e) {
                    // Continue
                }
                
                return {success: false, error: 'No Excel button found among ' + buttonCount + ' buttons'};
                
            } catch (e) {
                return {success: false, error: 'JavaScript error: ' + e.message};
            }
            """
            
            result = self.driver.execute_script(js_code)
            logger.info(f"JavaScript result: {result}")
            
            if result and result.get('success'):
                logger.info(f"✅ {result.get('message')}")
                
                # Wait for download
                downloaded_file = self.wait_for_download()
                
                if downloaded_file:
                    # Rename file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_filename = f"{league_name}_{timestamp}.xlsx"
                    new_path = self.downloads_dir / new_filename
                    
                    downloaded_file.rename(new_path)
                    logger.info(f"✅ File renamed to: {new_filename}")
                    return new_path
                else:
                    logger.error("❌ Download failed")
                    return None
            else:
                logger.error(f"❌ JavaScript failed: {result.get('error') if result else 'Unknown error'}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error scraping {league_name}: {e}")
            return None
    
    def load_leagues(self):
        """Load league configuration from CSV."""
        try:
            leagues_file = self.base_dir / "leagues.csv"
            with open(leagues_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                leagues = [row for row in reader]
            logger.info(f"📋 Loaded {len(leagues)} leagues")
            return leagues
        except Exception as e:
            logger.error(f"❌ Error loading leagues: {e}")
            return []
    
    def test_single_league(self, league_acronym="bsl"):
        """Test scraping a single league."""
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
            logger.info(f"🧪 TESTING: {test_league['acronym']}")
            
            result = self.scrape_league(test_league['acronym'], test_league['link'])
            
            if result:
                logger.info(f"✅ Success! Downloaded: {result}")
                return True
            else:
                logger.error("❌ Test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            return False
        finally:
            if self.driver:
                input("Press Enter to close browser...")
                self.driver.quit()


def main():
    """Main execution."""
    print("🏀 TBF Basketball Data Scraper v3.0")
    print("JavaScript-aware version")
    print("=" * 50)
    
    scraper = TBFScraperV3()
    success = scraper.test_single_league("bsl")
    
    if success:
        print("\n🎉 Test successful!")
    else:
        print("\n⚠️ Test failed.")

if __name__ == "__main__":
    main() 