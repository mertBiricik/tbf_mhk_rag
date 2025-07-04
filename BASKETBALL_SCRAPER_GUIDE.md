# Basketball Data Scraper System

This guide explains the basketball data scraping system that downloads live match data from the Turkish Basketball Federation (TBF) website.

## Overview

The system consists of several components:

1. Web Scraper (`basketball_data_scraper.py`) - Downloads Excel files from TBF
2. Data Manager (`data_manager.py`) - Processes and cleans CSV data
3. Launcher (`scraper_launcher.py`) - Orchestrates the entire pipeline
4. Test Suite (`test_scraper_setup.py`) - Validates setup

## League Data Coverage

The system scrapes data from 9 Turkish basketball leagues:

| Acronym | League Name | Gender | Sponsor |
|---------|-------------|--------|---------|
| BSL | Türkiye Sigorta Basketbol Süper Ligi | Erkek | Türkiye Sigorta |
| KBSL | Kadınlar Basketbol Süper Ligi | Kadın | ING |
| TBL | Türkiye Basketbol Ligi | Erkek | - |
| TKBL | Türkiye Kadın Basketbol Ligi | Kadın | - |
| TB2L | Türkiye Basketbol 2. Ligi | Erkek | - |
| BGL | Basketbol Gençler Ligi | Erkek | - |
| BGLK | Basketbol Gençler Ligi | Kadın | - |
| EBBL | Erkekler Bölgesel Basketbol Ligi | Erkek | - |
| KBBL | Kadınlar Bölgesel Basketbol Ligi | Kadın | - |

## Installation

### 1. Install Dependencies

```bash
# Install new scraping dependencies
pip install selenium pandas requests openpyxl beautifulsoup4

# Or update all requirements
pip install -r requirements.txt
```

### 2. Install Chrome Browser & ChromeDriver

Ubuntu/WSL2:
```bash
# Install Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update
sudo apt install google-chrome-stable

# Install ChromeDriver
sudo apt install chromium-chromedriver
```

Windows:
- Download Chrome from https://www.google.com/chrome/
- Download ChromeDriver from https://chromedriver.chromium.org/
- Add ChromeDriver to your PATH

## Quick Start

### 1. Test Your Setup

```bash
python scripts/test_scraper_setup.py
```

This will validate:
- Dependencies installed
- League data available
- Directories created
- Browser setup working
- Excel processing functional

### 2. Run the Complete Pipeline

```bash
python scripts/scraper_launcher.py
```

This will:
1. Download Excel files from all 9 leagues
2. Rename files with timestamps and league acronyms
3. Convert Excel files to CSV format
4. Create unified dataset
5. Generate summary reports

### 3. Manual Scraping (Individual Control)

```bash
# Run just the scraper
python scripts/basketball_data_scraper.py

# Run just the data processing
python scripts/data_manager.py
```

## File Structure

After running the scraper, you'll have:

```
stats/
├── leagues.csv                 # League configuration
├── downloads/                  # Excel files from TBF
│   ├── bsl_20250122_143022.xlsx
│   ├── bsl_latest.xlsx         # Symlink to latest
│   ├── kbsl_20250122_143127.xlsx
│   └── ...
├── processed/                  # CSV files and analysis
│   ├── bsl_20250122_143022.csv
│   ├── bsl_processed.csv
│   ├── bsl_analysis.json
│   ├── unified_basketball_data.csv
│   └── summary_20250122_143500.json
└── logs/
    └── basketball_scraper.log
```

## Data Processing Features

### Automatic File Naming
- Problem: All TBF files download as "Türkiye Basketbol Federasyonu.xlsx"
- Solution: Automatic renaming to `{league}_{timestamp}.xlsx`
- Example: `bsl_20250122_143022.xlsx`

### Excel to CSV Conversion
- Handles Turkish character encoding (UTF-8)
- Creates metadata files with column analysis
- Identifies key columns (teams, dates, scores, referees)

### Data Cleaning
- Standardizes text formatting
- Removes extra whitespace
- Adds league metadata columns
- Handles missing values

## Integration with RAG System

### Vector Database Preparation

The processed CSV files are ready for vector database ingestion:

```python
# Example: Load basketball match data
import pandas as pd

# Load unified dataset
df = pd.read_csv('./stats/processed/unified_basketball_data.csv')

# Each row becomes a document for RAG
for _, row in df.iterrows():
    document_text = f"""
    League: {row['league_name']}
    Teams: {row['team_a']} vs {row['team_b']}
    Score: {row['score_a']}-{row['score_b']}
    Date: {row['match_date']}
    Venue: {row['venue']}
    Referee: {row['referee']}
    """
    # Add to vector database...
```

### Enhanced RAG Queries

With live basketball data, your RAG system can answer:

Turkish:
- "Fenerbahçe'nin son 5 maçtaki performansı nasıl?"
- "Bu hafta BSL'de hangi maçlar var?"
- "Galatasaray kaç maç kazandı bu sezon?"

English:
- "How did Fenerbahçe perform in their last 5 games?"
- "What BSL games are this week?"
- "How many games has Galatasaray won this season?"

## Configuration

### Scraping Settings

Edit the scraper configuration in `basketball_data_scraper.py`:

```python
# Timing settings
delay_between_requests = 3.0  # Seconds between league requests
download_timeout = 30         # Max wait time for download

# Directory settings  
download_dir = "./stats/downloads"
processed_dir = "./stats/processed"
```

### League Management

Add/remove leagues by editing `stats/leagues.csv`:

```csv
link,acronym,name of the league,gender,sponsor,season
```

## Error Handling

### Common Issues

1. ChromeDriver version mismatch
   - Update ChromeDriver to match Chrome version
   - Check compatibility matrix

2. Download timeouts
   - Increase timeout values in configuration
   - Check internet connection stability

3. File permission errors
   - Ensure write permissions to stats directory
   - Close Excel files that might be open

4. Turkish character encoding
   - System uses UTF-8 encoding by default
   - Check locale settings if issues persist

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check log files in `stats/logs/` for detailed error information.

## Performance Optimization

- Parallel league processing
- Intelligent retry mechanisms
- Efficient file I/O operations
- Memory-optimized data processing

## Data Quality

- Automatic data validation
- Duplicate detection and removal
- Missing value handling
- Data type consistency checks

The scraper system provides reliable, automated data collection for Turkish basketball leagues, integrating seamlessly with the RAG system for comprehensive basketball information retrieval. 