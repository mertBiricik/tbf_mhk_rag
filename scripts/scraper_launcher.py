#!/usr/bin/env python3
"""
Basketball Data Scraper Launcher
Orchestrates the entire scraping and processing pipeline
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'selenium', 'pandas', 'requests', 'openpyxl'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"❌ Missing packages: {', '.join(missing)}")
        logger.info("📦 Install with: pip install selenium pandas requests openpyxl")
        return False
    
    logger.info("✅ All dependencies available")
    return True

def create_directories():
    """Create necessary directories."""
    dirs = [
        "./stats/downloads",
        "./stats/processed", 
        "./logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created directory: {dir_path}")

def run_scraper():
    """Run the basketball data scraper."""
    logger.info("🚀 Starting basketball data scraper...")
    
    try:
        # Import and run scraper
        from basketball_data_scraper import BasketballDataScraper
        
        with BasketballDataScraper() as scraper:
            results = scraper.scrape_all_leagues(delay_between_requests=3.0)
            
            logger.info(f"✅ Scraping completed: {results['successful']}/{results['total_leagues']} leagues")
            return results
            
    except Exception as e:
        logger.error(f"❌ Scraping failed: {e}")
        return None

def process_data():
    """Process scraped CSV files."""
    logger.info("📊 Processing scraped data...")
    
    try:
        # Simple data processing
        import pandas as pd
        from pathlib import Path
        
        processed_dir = Path("./stats/processed")
        csv_files = list(processed_dir.glob("*.csv"))
        csv_files = [f for f in csv_files if not f.name.endswith('_processed.csv')]
        
        logger.info(f"📂 Found {len(csv_files)} CSV files to process")
        
        processed_count = 0
        for csv_file in csv_files:
            try:
                # Read and analyze CSV
                df = pd.read_csv(csv_file, encoding='utf-8')
                
                # Create processed version
                processed_file = processed_dir / f"{csv_file.stem}_processed.csv"
                df.to_csv(processed_file, index=False, encoding='utf-8')
                
                logger.info(f"   ✅ Processed: {csv_file.name} ({len(df)} rows)")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"   ❌ Error processing {csv_file.name}: {e}")
        
        logger.info(f"📊 Data processing completed: {processed_count}/{len(csv_files)} files")
        return processed_count > 0
        
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        return False

def generate_summary():
    """Generate summary of all collected data."""
    logger.info("📋 Generating data summary...")
    
    try:
        import pandas as pd
        from pathlib import Path
        import json
        
        processed_dir = Path("./stats/processed")
        summary = {
            'timestamp': datetime.now().isoformat(),
            'files': [],
            'total_rows': 0
        }
        
        # Process each CSV file
        csv_files = list(processed_dir.glob("*_processed.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                
                file_info = {
                    'filename': csv_file.name,
                    'league': csv_file.stem.split('_')[0],
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()
                }
                
                summary['files'].append(file_info)
                summary['total_rows'] += len(df)
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
        
        # Save summary
        summary_file = processed_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📋 Summary saved: {summary_file.name}")
        logger.info(f"   📊 Total files: {len(summary['files'])}")
        logger.info(f"   📈 Total rows: {summary['total_rows']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Summary generation failed: {e}")
        return None

def main():
    """Main function to orchestrate the entire pipeline."""
    print("🏀 BASKETBALL DATA SCRAPER LAUNCHER")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run scraper
    scraping_results = run_scraper()
    if not scraping_results:
        logger.error("❌ Scraping failed, stopping pipeline")
        sys.exit(1)
    
    # Add delay between scraping and processing
    logger.info("⏳ Waiting 5 seconds before processing...")
    time.sleep(5)
    
    # Process data
    processing_success = process_data()
    if not processing_success:
        logger.warning("⚠️  Data processing had issues, but continuing...")
    
    # Generate summary
    summary = generate_summary()
    
    # Final report
    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETED!")
    print("="*50)
    
    if scraping_results:
        print(f"📥 Scraping: {scraping_results['successful']}/{scraping_results['total_leagues']} leagues")
    
    if summary:
        print(f"📊 Processing: {len(summary['files'])} files processed")
        print(f"📈 Data: {summary['total_rows']} total rows collected")
    
    print(f"📁 Files saved to:")
    print(f"   Excel: ./stats/downloads/")
    print(f"   CSV: ./stats/processed/")
    print("="*50)

if __name__ == "__main__":
    main() 