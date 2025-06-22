#!/usr/bin/env python3
"""
Compare Selenium vs BeautifulSoup Basketball Scrapers
Test both approaches and see which works better
"""

import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bs4_scraper():
    """Test BeautifulSoup scraper."""
    try:
        from basketball_scraper_bs4 import BasketballScraperBS4
        
        logger.info("🧪 Testing BeautifulSoup scraper...")
        start_time = time.time()
        
        scraper = BasketballScraperBS4()
        
        # Test just one league first
        if scraper.leagues:
            test_league = scraper.leagues[0]  # BSL
            logger.info(f"Testing with {test_league['acronym']}: {test_league['name']}")
            
            result = scraper.scrape_league(test_league)
            
            end_time = time.time()
            duration = end_time - start_time
            
            scraper.close()
            
            return {
                'method': 'BeautifulSoup',
                'success': result['success'],
                'duration': duration,
                'error': result.get('error'),
                'details': result
            }
        else:
            return {
                'method': 'BeautifulSoup',
                'success': False,
                'error': 'No leagues loaded'
            }
            
    except Exception as e:
        logger.error(f"❌ BS4 scraper test failed: {e}")
        return {
            'method': 'BeautifulSoup',
            'success': False,
            'error': str(e)
        }

def test_selenium_scraper():
    """Test Selenium scraper."""
    try:
        from basketball_data_scraper import BasketballDataScraper
        
        logger.info("🧪 Testing Selenium scraper...")
        start_time = time.time()
        
        with BasketballDataScraper() as scraper:
            # Test just one league first
            if scraper.leagues:
                test_league = scraper.leagues[0]  # BSL
                logger.info(f"Testing with {test_league['acronym']}: {test_league['name']}")
                
                result = scraper.scrape_league(test_league)
                
                end_time = time.time()
                duration = end_time - start_time
                
                return {
                    'method': 'Selenium',
                    'success': result['success'],
                    'duration': duration,
                    'error': result.get('error'),
                    'details': result
                }
            else:
                return {
                    'method': 'Selenium',
                    'success': False,
                    'error': 'No leagues loaded'
                }
                
    except Exception as e:
        logger.error(f"❌ Selenium scraper test failed: {e}")
        return {
            'method': 'Selenium',
            'success': False,
            'error': str(e)
        }

def main():
    """Compare both scrapers."""
    print("🏀 BASKETBALL SCRAPER COMPARISON")
    print("="*50)
    
    # Test BS4 first (lighter)
    bs4_result = test_bs4_scraper()
    
    print(f"\n📊 BeautifulSoup Results:")
    print(f"   ✅ Success: {bs4_result['success']}")
    if bs4_result['success']:
        print(f"   ⏱️  Duration: {bs4_result['duration']:.2f}s")
        print(f"   📁 Excel: {bs4_result['details'].get('excel_file', 'N/A')}")
        print(f"   📊 CSV: {bs4_result['details'].get('csv_file', 'N/A')}")
    else:
        print(f"   ❌ Error: {bs4_result.get('error', 'Unknown')}")
    
    # Test Selenium
    selenium_result = test_selenium_scraper()
    
    print(f"\n🌐 Selenium Results:")
    print(f"   ✅ Success: {selenium_result['success']}")
    if selenium_result['success']:
        print(f"   ⏱️  Duration: {selenium_result['duration']:.2f}s")
        print(f"   📁 Excel: {selenium_result['details'].get('excel_file', 'N/A')}")
        print(f"   📊 CSV: {selenium_result['details'].get('csv_file', 'N/A')}")
    else:
        print(f"   ❌ Error: {selenium_result.get('error', 'Unknown')}")
    
    # Comparison
    print(f"\n🏆 COMPARISON:")
    
    if bs4_result['success'] and selenium_result['success']:
        bs4_time = bs4_result['duration']
        selenium_time = selenium_result['duration']
        
        if bs4_time < selenium_time:
            faster = 'BeautifulSoup'
            difference = selenium_time - bs4_time
        else:
            faster = 'Selenium'
            difference = bs4_time - selenium_time
        
        print(f"   🚀 Faster: {faster} (by {difference:.2f}s)")
        print(f"   📈 Speed improvement: {(difference/max(bs4_time, selenium_time)*100):.1f}%")
    
    elif bs4_result['success']:
        print(f"   🎯 BeautifulSoup works, Selenium failed")
        print(f"   💡 Recommendation: Use BeautifulSoup scraper")
    
    elif selenium_result['success']:
        print(f"   🎯 Selenium works, BeautifulSoup failed")
        print(f"   💡 Recommendation: Use Selenium scraper")
    
    else:
        print(f"   ❌ Both scrapers failed")
        print(f"   💡 Check TBF website structure or network issues")
    
    print("="*50)
    
    # Provide recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if bs4_result['success']:
        print(f"   🚀 Use: python scripts/basketball_scraper_bs4.py")
        print(f"   ✅ Benefits: Faster, lighter, no browser needed")
    
    if selenium_result['success']:
        print(f"   🔄 Alternative: python scripts/basketball_data_scraper.py") 
        print(f"   ✅ Benefits: Handles JavaScript, more robust for complex pages")
    
    print(f"\n🔗 For full scraping:")
    print(f"   python scripts/scraper_launcher.py  # Uses current Selenium")
    print(f"   python scripts/basketball_scraper_bs4.py  # Direct BS4")

if __name__ == "__main__":
    main() 