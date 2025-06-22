#!/usr/bin/env python3
"""
Comprehensive TBF Basketball Data Scraper
"""

import requests
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

class ComprehensiveBasketballScraper:
    """Comprehensive scraper for all TBF basketball data."""
    
    def __init__(self):
        self.setup_directories()
        self.driver = None
        self.base_url = "https://www.tbf.org.tr/ligler/"
        
        # Seasons to try (most recent first)
        self.seasons_to_try = [
            "2024-2025", "2023-2024", "2022-2023", "2021-2022", 
            "2020-2021", "2019-2020", "2018-2019", "2017-2018",
            "2016-2017", "2015-2016", "2014-2015", "2013-2014"
        ]
        
    def setup_directories(self):
        """Create directories."""
        self.base_dir = Path("./stats")
        self.downloads_dir = self.base_dir / "downloads"
        self.processed_dir = self.base_dir / "processed"
        self.logs_dir = Path("./logs")
        
        for dir_path in [self.downloads_dir, self.processed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_chrome_driver(self):
        """Setup headless Chrome driver."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        
        prefs = {
            "download.default_directory": str(self.downloads_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("✅ Chrome driver initialized (headless)")
            return True
        except Exception as e:
            logger.error(f"❌ Chrome driver failed: {e}")
            return False
    
    def check_url_exists(self, url):
        """Check if URL exists."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def discover_seasons_for_league(self, league_acronym):
        """Discover available seasons for a league."""
        available_seasons = []
        
        logger.info(f"🔍 Discovering seasons for {league_acronym}...")
        
        for season in self.seasons_to_try:
            url = f"{self.base_url}{league_acronym}-{season}/maclar"
            logger.info(f"  🔍 Checking: {url}")
            
            if self.check_url_exists(url):
                available_seasons.append(season)
                logger.info(f"  ✅ Found: {season}")
            else:
                logger.info(f"  ❌ Not found: {season}")
            
            time.sleep(0.5)
        
        logger.info(f"📊 {league_acronym}: {len(available_seasons)} seasons")
        return available_seasons
    
    def wait_for_download(self, timeout=30):
        """Wait for download completion."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            crdownload_files = list(self.downloads_dir.glob("*.crdownload"))
            if not crdownload_files:
                excel_files = list(self.downloads_dir.glob("*.xlsx"))
                if excel_files:
                    return max(excel_files, key=lambda f: f.stat().st_mtime)
            time.sleep(1)
        return None
    
    def scrape_league_season(self, league_info, season):
        """Scrape specific league and season."""
        league_acronym = league_info['acronym'].strip()
        gender = league_info['gender'].strip()
        url = f"{self.base_url}{league_acronym}-{season}/maclar"
        
        logger.info(f"🏀 Scraping {league_acronym} {gender} {season}")
        
        try:
            # Clear old downloads
            for old_file in self.downloads_dir.glob("*.xlsx"):
                try:
                    old_file.unlink()
                except:
                    pass
            
            # Navigate and wait
            self.driver.get(url)
            time.sleep(8)
            
            # Wait for table
            wait = WebDriverWait(self.driver, 20)
            table_locator = (By.ID, "TableMaclar")
            wait.until(EC.presence_of_element_located(table_locator))
            time.sleep(3)
            
            # JavaScript Excel export
            js_code = """
            try {
                var table = $('#TableMaclar').DataTable();
                if (!table) return {success: false, error: 'DataTable not found'};
                
                var buttonCount = 0;
                try {
                    buttonCount = table.buttons().length || 0;
                } catch(e) {
                    return {success: false, error: 'No buttons: ' + e.message};
                }
                
                for (var i = 0; i < buttonCount; i++) {
                    try {
                        var button = table.button(i);
                        var text = button.text() || '';
                        if (text.toLowerCase().includes('excel') || text.toLowerCase().includes('aktar')) {
                            button.trigger();
                            return {success: true, message: 'Excel button clicked'};
                        }
                    } catch(e) {}
                }
                
                try {
                    var btn = table.button('.buttons-excel');
                    if (btn && btn.length > 0) {
                        btn.trigger();
                        return {success: true, message: 'Excel via .buttons-excel'};
                    }
                } catch(e) {}
                
                return {success: false, error: 'No Excel button found'};
                
            } catch (e) {
                return {success: false, error: 'JS error: ' + e.message};
            }
            """
            
            result = self.driver.execute_script(js_code)
            
            if result and result.get('success'):
                downloaded_file = self.wait_for_download()
                
                if downloaded_file:
                    # Create filename: league_gender_season.xlsx
                    new_filename = f"{league_acronym}_{gender}_{season}.xlsx"
                    new_path = self.downloads_dir / new_filename
                    downloaded_file.rename(new_path)
                    
                    # Convert to CSV
                    csv_path = self.process_excel_to_csv(new_path, league_info, season)
                    
                    return {
                        'league': league_acronym,
                        'gender': gender,
                        'season': season,
                        'status': 'success',
                        'excel_file': new_filename,
                        'csv_file': csv_path.name if csv_path else None
                    }
                else:
                    return {
                        'league': league_acronym,
                        'gender': gender,
                        'season': season,
                        'status': 'download_failed'
                    }
            else:
                return {
                    'league': league_acronym,
                    'gender': gender,
                    'season': season,
                    'status': 'js_failed',
                    'error': result.get('error') if result else 'Unknown error'
                }
                
        except Exception as e:
            return {
                'league': league_acronym,
                'gender': gender,
                'season': season,
                'status': 'error',
                'error': str(e)
            }
    
    def process_excel_to_csv(self, excel_path, league_info, season):
        """Convert Excel to CSV with proper naming."""
        try:
            df = pd.read_excel(excel_path)
            
            league_acronym = league_info['acronym'].strip()
            gender = league_info['gender'].strip()
            csv_filename = f"{league_acronym}_{gender}_{season}.csv"
            csv_path = self.processed_dir / csv_filename
            
            # Save CSV with UTF-8
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Save metadata
            metadata = {
                'league_acronym': league_acronym,
                'league_name': league_info['name_of_the_league'].strip(),
                'gender': gender,
                'season': season,
                'sponsor': league_info.get('sponsor', '').strip(),
                'scraped_at': datetime.now().isoformat(),
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'column_names': list(df.columns)
            }
            
            metadata_filename = f"{league_acronym}_{gender}_{season}_metadata.json"
            metadata_path = self.processed_dir / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved: {csv_filename}")
            return csv_path
            
        except Exception as e:
            logger.error(f"❌ CSV conversion failed: {e}")
            return None
    
    def load_leagues(self):
        """Load league configuration."""
        try:
            leagues_file = self.base_dir / "leagues.csv"
            with open(leagues_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                leagues = [row for row in reader]
            return leagues
        except Exception as e:
            logger.error(f"❌ Error loading leagues: {e}")
            return []
    
    def scrape_all_data(self):
        """Main scraping function."""
        if not self.setup_chrome_driver():
            return False
        
        leagues = self.load_leagues()
        if not leagues:
            return False
        
        all_results = []
        
        try:
            logger.info("🚀 COMPREHENSIVE BASKETBALL DATA SCRAPING")
            
            # Discover seasons for each league
            league_seasons = {}
            for league in leagues:
                league_acronym = league['acronym'].strip()
                seasons = self.discover_seasons_for_league(league_acronym)
                league_seasons[league_acronym] = seasons
                time.sleep(1)
            
            # Calculate total
            total_ops = sum(len(seasons) for seasons in league_seasons.values())
            logger.info(f"📊 Total combinations to scrape: {total_ops}")
            
            current_op = 0
            
            # Scrape each combination
            for league in leagues:
                league_acronym = league['acronym'].strip()
                available_seasons = league_seasons.get(league_acronym, [])
                
                if not available_seasons:
                    continue
                
                for season in available_seasons:
                    current_op += 1
                    logger.info(f"[{current_op}/{total_ops}] {league_acronym} {season}")
                    
                    result = self.scrape_league_season(league, season)
                    all_results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"✅ SUCCESS")
                    else:
                        logger.error(f"❌ FAILED: {result.get('error', 'Unknown')}")
                    
                    time.sleep(2)  # Rate limiting
        
        except KeyboardInterrupt:
            logger.info("⏹️ Interrupted")
        finally:
            if self.driver:
                self.driver.quit()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.base_dir / f"scraping_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Summary
        successful = len([r for r in all_results if r['status'] == 'success'])
        total = len(all_results)
        
        print(f"\n{'='*50}")
        print(f"📊 SCRAPING COMPLETE")
        print(f"✅ Successful: {successful}/{total}")
        print(f"❌ Failed: {total - successful}/{total}")
        print(f"📁 Files in: {self.downloads_dir}")
        print(f"📄 CSV in: {self.processed_dir}")
        
        return successful > 0


def main():
    """Main execution."""
    print("🏀 Comprehensive TBF Basketball Data Scraper")
    print("=" * 50)
    
    scraper = ComprehensiveBasketballScraper()
    success = scraper.scrape_all_data()
    
    if success:
        print("🎉 Scraping completed!")
    else:
        print("⚠️ Scraping failed.")

if __name__ == "__main__":
    main() 