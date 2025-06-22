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
import random
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
            "2016-2017", "2015-2016", "2014-2015", "2013-2014",
            "2012-2013", "2011-2012", "2010-2011"
        ]
        
        # Progress tracking
        self.progress_file = self.base_dir / "scraping_progress.json"
        self.completed_combinations = set()
        self.load_progress()
        
    def setup_directories(self):
        """Create directories."""
        self.base_dir = Path("./stats")
        self.downloads_dir = self.base_dir / "downloads"
        self.processed_dir = self.base_dir / "processed"
        self.logs_dir = Path("./logs")
        
        for dir_path in [self.downloads_dir, self.processed_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_chrome_driver(self):
        """Setup Chrome driver with anti-detection measures."""
        chrome_options = Options()
        # Anti-detection measures
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # More human-like user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        prefs = {
            "download.default_directory": str(self.downloads_dir.absolute()),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        }
        chrome_options.add_experimental_option("prefs", prefs)
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to hide automation indicators
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            logger.info("✅ Chrome driver initialized with anti-detection")
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
    
    def load_progress(self):
        """Load progress from previous runs."""
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                    completed = progress_data.get('completed_combinations', [])
                    self.completed_combinations = set(tuple(item) for item in completed)
                    logger.info(f"📋 Loaded progress: {len(self.completed_combinations)} completed combinations")
            else:
                logger.info("📋 No previous progress found, starting fresh")
        except Exception as e:
            logger.warning(f"⚠️ Could not load progress: {e}")
            self.completed_combinations = set()
    
    def save_progress(self):
        """Save current progress."""
        try:
            progress_data = {
                'completed_combinations': [list(item) for item in self.completed_combinations],
                'last_updated': datetime.now().isoformat(),
                'total_completed': len(self.completed_combinations)
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️ Could not save progress: {e}")
    
    def is_combination_completed(self, league_acronym, gender, season):
        """Check if a combination has already been completed."""
        combination = (league_acronym, gender, season)
        return combination in self.completed_combinations
    
    def mark_combination_completed(self, league_acronym, gender, season):
        """Mark a combination as completed."""
        combination = (league_acronym, gender, season)
        self.completed_combinations.add(combination)
        self.save_progress()

    def get_seasons_for_league(self, league_acronym):
        """Return predefined seasons without checking URLs."""
        logger.info(f"📋 Using predefined seasons for {league_acronym}")
        logger.info(f"📊 {league_acronym}: {len(self.seasons_to_try)} seasons")
        return self.seasons_to_try
    
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
        """Scrape specific league and season with fresh browser instance."""
        league_acronym = league_info['acronym'].strip()
        gender = league_info['gender'].strip()
        url = f"{self.base_url}{league_acronym}-{season}/maclar"
        
        logger.info(f"🏀 Scraping {league_acronym} {gender} {season}")
        
        # Start fresh browser instance for this request
        if not self.setup_chrome_driver():
            return {
                'league': league_acronym,
                'gender': gender,
                'season': season,
                'status': 'driver_failed'
            }
        
        try:
            # Clear old downloads
            for old_file in self.downloads_dir.glob("*.xlsx"):
                try:
                    old_file.unlink()
                except:
                    pass
            
            # Navigate to page
            logger.info(f"🌐 Navigating to: {url}")
            self.driver.get(url)
            
            # Wait for page load
            time.sleep(8)
            
            # Wait for table to load
            wait = WebDriverWait(self.driver, 30)
            table_locator = (By.ID, "TableMaclar")
            wait.until(EC.presence_of_element_located(table_locator))
            logger.info("📊 Table loaded successfully")
            
            # Brief wait for DataTables to initialize
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
        finally:
            # Always close the browser instance after each request
            if self.driver:
                try:
                    self.driver.quit()
                    logger.info("🚪 Browser instance closed")
                except:
                    pass
                self.driver = None
    
    def process_excel_to_csv(self, excel_path, league_info, season):
        """Convert Excel to CSV with proper naming and header handling."""
        try:
            # Read Excel file - headers are in row 2 (index 1)
            df = pd.read_excel(excel_path, header=1)
            
            league_acronym = league_info['acronym'].strip()
            gender = league_info['gender'].strip()
            
            # Save Excel file to processed directory
            excel_filename = f"{league_acronym}_{gender}_{season}.xlsx"
            excel_processed_path = self.processed_dir / excel_filename
            excel_path.rename(excel_processed_path)
            
            # Save CSV with proper headers
            csv_filename = f"{league_acronym}_{gender}_{season}.csv"
            csv_path = self.processed_dir / csv_filename
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Save metadata with correct headers
            metadata = {
                'league_acronym': league_acronym,
                'league_name': league_info['name_of_the_league'].strip(),
                'gender': gender,
                'season': season,
                'sponsor': league_info.get('sponsor', '').strip(),
                'scraped_at': datetime.now().isoformat(),
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'column_names': list(df.columns),
                'excel_file': excel_filename,
                'csv_file': csv_filename
            }
            
            metadata_filename = f"{league_acronym}_{gender}_{season}_metadata.json"
            metadata_path = self.processed_dir / metadata_filename
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved Excel: {excel_filename}")
            logger.info(f"💾 Saved CSV: {csv_filename}")
            logger.info(f"📊 Data: {df.shape[0]} rows × {df.shape[1]} columns")
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
    
    def scrape_all_data(self, resume=True):
        """Main scraping function with resume capability."""
        leagues = self.load_leagues()
        if not leagues:
            logger.error("❌ No leagues loaded")
            return False
        
        all_results = []
        
        try:
            logger.info("🚀 COMPREHENSIVE BASKETBALL DATA SCRAPING")
            
            if resume and len(self.completed_combinations) > 0:
                logger.info(f"🔄 RESUMING: {len(self.completed_combinations)} combinations already completed")
            
            # Get predefined seasons for each league
            league_seasons = {}
            for league in leagues:
                league_acronym = league['acronym'].strip()
                seasons = self.get_seasons_for_league(league_acronym)
                league_seasons[league_acronym] = seasons
            
            # Calculate total and remaining
            total_ops = sum(len(seasons) for seasons in league_seasons.values())
            completed_count = len(self.completed_combinations)
            remaining_ops = total_ops - completed_count
            
            logger.info(f"📊 Total combinations: {total_ops}")
            logger.info(f"✅ Already completed: {completed_count}")
            logger.info(f"⏳ Remaining to scrape: {remaining_ops}")
            
            if remaining_ops == 0:
                logger.info("🎉 All combinations already completed!")
                return True
            
            current_op = 0
            skipped_count = 0
            
            # Scrape each combination
            for league in leagues:
                league_acronym = league['acronym'].strip()
                gender = league['gender'].strip()
                available_seasons = league_seasons.get(league_acronym, [])
                
                if not available_seasons:
                    continue
                
                for season in available_seasons:
                    current_op += 1
                    
                    # Check if already completed
                    if resume and self.is_combination_completed(league_acronym, gender, season):
                        skipped_count += 1
                        logger.info(f"[{current_op}/{total_ops}] ⏭️  SKIPPING {league_acronym} {gender} {season} (already completed)")
                        continue
                    
                    logger.info(f"[{current_op}/{total_ops}] 🏀 SCRAPING {league_acronym} {gender} {season}")
                    
                    result = self.scrape_league_season(league, season)
                    all_results.append(result)
                    
                    if result['status'] == 'success':
                        logger.info(f"✅ SUCCESS")
                        # Mark as completed
                        self.mark_combination_completed(league_acronym, gender, season)
                    else:
                        logger.error(f"❌ FAILED: {result.get('error', 'Unknown')}")
                    
                    # Simple delay between requests (browser restart provides main protection)
                    time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("⏹️ Interrupted")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.base_dir / f"scraping_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Summary
        successful = len([r for r in all_results if r['status'] == 'success'])
        total_attempted = len(all_results)
        total_completed = len(self.completed_combinations)
        
        print(f"\n{'='*50}")
        print(f"📊 SCRAPING SESSION COMPLETE")
        print(f"✅ This session successful: {successful}/{total_attempted}")
        print(f"❌ This session failed: {total_attempted - successful}/{total_attempted}")
        print(f"🎯 Total completed overall: {total_completed}")
        if hasattr(self, 'skipped_count'):
            print(f"⏭️  Skipped (already done): {skipped_count}")
        print(f"📁 Files in: {self.downloads_dir}")
        print(f"📄 CSV in: {self.processed_dir}")
        print(f"💾 Progress saved in: {self.progress_file}")
        
        return successful > 0
    
    def clear_progress(self):
        """Clear all progress - use to start completely fresh."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
                logger.info("🗑️  Progress file deleted")
            self.completed_combinations = set()
            logger.info("🔄 Ready for fresh start")
        except Exception as e:
            logger.error(f"❌ Could not clear progress: {e}")
    
    def show_progress_summary(self):
        """Show detailed progress summary."""
        print(f"\n{'='*50}")
        print(f"📊 PROGRESS SUMMARY")
        print(f"✅ Completed combinations: {len(self.completed_combinations)}")
        
        if self.completed_combinations:
            # Group by league
            by_league = {}
            for league, gender, season in self.completed_combinations:
                if league not in by_league:
                    by_league[league] = []
                by_league[league].append(f"{gender} {season}")
            
            for league, combinations in by_league.items():
                print(f"  🏀 {league}: {len(combinations)} combinations")
                for combo in sorted(combinations)[:3]:  # Show first 3
                    print(f"    - {combo}")
                if len(combinations) > 3:
                    print(f"    ... and {len(combinations) - 3} more")
        
        print(f"💾 Progress file: {self.progress_file}")
        print(f"{'='*50}")


def main():
    """Main execution with progress management options."""
    import sys
    
    print("🏀 Comprehensive TBF Basketball Data Scraper")
    print("=" * 50)
    
    scraper = ComprehensiveBasketballScraper()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            scraper.show_progress_summary()
            return
        elif command == "clear":
            scraper.clear_progress()
            print("🔄 Progress cleared. Ready for fresh start.")
            return
        elif command == "fresh":
            scraper.clear_progress()
            print("🔄 Starting fresh scraping...")
            success = scraper.scrape_all_data(resume=False)
        elif command == "resume":
            print("🔄 Resuming from previous progress...")
            success = scraper.scrape_all_data(resume=True)
        else:
            print("❓ Unknown command. Available commands:")
            print("  python script.py status  - Show progress summary")
            print("  python script.py clear   - Clear all progress")
            print("  python script.py fresh   - Start completely fresh")
            print("  python script.py resume  - Resume from previous progress")
            return
    else:
        # Default: resume if progress exists, otherwise start fresh
        if scraper.progress_file.exists():
            print("🔄 Found previous progress. Resuming...")
            success = scraper.scrape_all_data(resume=True)
        else:
            print("🚀 Starting fresh scraping...")
            success = scraper.scrape_all_data(resume=False)
    
    if success:
        print("🎉 Scraping session completed!")
        scraper.show_progress_summary()
    else:
        print("⚠️ Scraping session failed.")

if __name__ == "__main__":
    main() 