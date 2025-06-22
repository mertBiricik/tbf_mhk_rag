#!/usr/bin/env python3
"""
Test script to validate Basketball Data Scraper setup
"""

import os
import sys
import csv
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dependencies():
    """Test if all required dependencies are available."""
    logger.info("🧪 Testing dependencies...")
    
    required_packages = {
        'selenium': 'selenium',
        'pandas': 'pandas', 
        'requests': 'requests',
        'openpyxl': 'openpyxl'
    }
    
    results = {}
    for name, package in required_packages.items():
        try:
            __import__(package)
            results[name] = "✅ Available"
            logger.info(f"   {name}: ✅")
        except ImportError:
            results[name] = "❌ Missing"
            logger.error(f"   {name}: ❌")
    
    return results

def test_league_data():
    """Test the league CSV file."""
    logger.info("📊 Testing league data...")
    
    leagues_file = Path("./stats/leagues.csv")
    
    if not leagues_file.exists():
        logger.error("❌ leagues.csv not found")
        return False
    
    try:
        with open(leagues_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            leagues = list(reader)
        
        logger.info(f"✅ Found {len(leagues)} leagues:")
        
        for league in leagues:
            acronym = league['acronym'].strip()
            name = league['name_of_the_league'].strip()
            link = league['link'].strip()
            
            logger.info(f"   🏀 {acronym.upper()}: {name}")
            logger.info(f"      📍 {link}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error reading leagues.csv: {e}")
        return False

def test_directory_structure():
    """Test directory creation."""
    logger.info("📁 Testing directory structure...")
    
    required_dirs = [
        "./stats",
        "./stats/downloads", 
        "./stats/processed",
        "./logs"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        
        if path.exists():
            logger.info(f"   ✅ {dir_path}")
        else:
            logger.error(f"   ❌ {dir_path}")
            return False
    
    return True

def test_browser_setup():
    """Test if browser setup works."""
    logger.info("🌐 Testing browser setup...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Try to create driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.quit()
        
        logger.info("   ✅ Chrome browser setup successful")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ Browser setup failed: {e}")
        logger.info("   💡 Make sure Chrome and ChromeDriver are installed")
        return False

def test_excel_processing():
    """Test Excel to CSV conversion."""
    logger.info("📊 Testing Excel processing...")
    
    try:
        import pandas as pd
        
        # Create a test Excel file
        test_data = {
            'Team A': ['Fenerbahçe', 'Galatasaray', 'Beşiktaş'],
            'Team B': ['Anadolu Efes', 'Bursaspor', 'Karşıyaka'],
            'Score A': [85, 78, 92],
            'Score B': [72, 80, 88]
        }
        
        df = pd.DataFrame(test_data)
        test_dir = Path("./stats/test")
        test_dir.mkdir(exist_ok=True)
        
        # Save as Excel
        test_excel = test_dir / "test_data.xlsx"
        df.to_excel(test_excel, index=False)
        
        # Convert to CSV
        test_csv = test_dir / "test_data.csv"
        df.to_csv(test_csv, index=False, encoding='utf-8')
        
        # Verify conversion
        df_read = pd.read_csv(test_csv, encoding='utf-8')
        
        if len(df_read) == len(df):
            logger.info("   ✅ Excel/CSV processing works")
            
            # Cleanup
            test_excel.unlink()
            test_csv.unlink()
            test_dir.rmdir()
            
            return True
        else:
            logger.error("   ❌ Data mismatch in conversion")
            return False
            
    except Exception as e:
        logger.error(f"   ❌ Excel processing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏀 BASKETBALL SCRAPER SETUP TEST")
    print("="*40)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("League Data", test_league_data),
        ("Directories", test_directory_structure),
        ("Browser Setup", test_browser_setup),
        ("Excel Processing", test_excel_processing)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*40)
    print("📋 TEST SUMMARY")
    print("="*40)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Ready to scrape!")
        print("\n🚀 Next steps:")
        print("   1. python scripts/basketball_data_scraper.py")
        print("   2. python scripts/scraper_launcher.py")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before scraping.")
        print("\n📦 If dependencies are missing:")
        print("   pip install selenium pandas requests openpyxl")
    
    print("="*40)

if __name__ == "__main__":
    main() 