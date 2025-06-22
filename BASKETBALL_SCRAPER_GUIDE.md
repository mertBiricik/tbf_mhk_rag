# Basketball Data Scraper System 🏀

This guide explains how to use the new basketball data scraping system that downloads live match data from the Turkish Basketball Federation (TBF) website.

## Overview

The system consists of several components:

1. **Web Scraper** (`basketball_data_scraper.py`) - Downloads Excel files from TBF
2. **Data Manager** (`data_manager.py`) - Processes and cleans CSV data  
3. **Launcher** (`scraper_launcher.py`) - Orchestrates the entire pipeline
4. **Test Suite** (`test_scraper_setup.py`) - Validates setup

## League Data Coverage

The system scrapes data from **9 Turkish basketball leagues**:

| Acronym | League Name | Gender | Sponsor |
|---------|-------------|--------|---------|
| **BSL** | Türkiye Sigorta Basketbol Süper Ligi | Erkek | Türkiye Sigorta |
| **KBSL** | Kadınlar Basketbol Süper Ligi | Kadın | ING |
| **TBL** | Türkiye Basketbol Ligi | Erkek | - |
| **TKBL** | Türkiye Kadın Basketbol Ligi | Kadın | - |
| **TB2L** | Türkiye Basketbol 2. Ligi | Erkek | - |
| **BGL** | Basketbol Gençler Ligi | Erkek | - |
| **BGLK** | Basketbol Gençler Ligi | Kadın | - |
| **EBBL** | Erkekler Bölgesel Basketbol Ligi | Erkek | - |
| **KBBL** | Kadınlar Bölgesel Basketbol Ligi | Kadın | - |

## Installation

### 1. Install Dependencies

```bash
# Install new scraping dependencies
pip install selenium pandas requests openpyxl beautifulsoup4

# Or update all requirements
pip install -r requirements.txt
```

### 2. Install Chrome Browser & ChromeDriver

**Ubuntu/WSL2:**
```bash
# Install Chrome
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt update
sudo apt install google-chrome-stable

# Install ChromeDriver
sudo apt install chromium-chromedriver
```

**Windows:**
- Download Chrome from https://www.google.com/chrome/
- Download ChromeDriver from https://chromedriver.chromium.org/
- Add ChromeDriver to your PATH

## Quick Start

### 1. Test Your Setup

```bash
python scripts/test_scraper_setup.py
```

This will validate:
- ✅ Dependencies installed
- ✅ League data available  
- ✅ Directories created
- ✅ Browser setup working
- ✅ Excel processing functional

### 2. Run the Complete Pipeline

```bash
python scripts/scraper_launcher.py
```

This will:
1. 📥 Download Excel files from all 9 leagues
2. 📝 Rename files with timestamps and league acronyms
3. 📊 Convert Excel files to CSV format
4. 🔗 Create unified dataset
5. 📋 Generate summary reports

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
- **Problem**: All TBF files download as "Türkiye Basketbol Federasyonu.xlsx"
- **Solution**: Automatic renaming to `{league}_{timestamp}.xlsx`
- **Example**: `bsl_20250122_143022.xlsx`

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

**Turkish:**
- "Fenerbahçe'nin son 5 maçtaki performansı nasıl?"
- "Bu hafta BSL'de hangi maçlar var?"
- "Galatasaray kaç maç kazandı bu sezon?"

**English:**
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
https://www.tbf.org.tr/ligler/new-league/maclar,NEW,New League,Erkek,,2024-2025
```

## Automation & Scheduling

### Daily Updates

Create a cron job for daily data updates:

```bash
# Edit crontab
crontab -e

# Add daily scraping at 2 AM
0 2 * * * cd /path/to/tbf_mhk_rag_mcp && python scripts/scraper_launcher.py
```

### Real-time Integration

For real-time updates during games:

```python
# Schedule more frequent updates during game days
import schedule
import time

def scrape_live_data():
    """Scrape data during live games."""
    # Run scraper with shorter delays
    # Update vector database
    # Send notifications

# Schedule every 15 minutes during game hours
schedule.every(15).minutes.do(scrape_live_data)
```

## MCP Server Integration

This scraping system prepares data for your planned MCP server:

### Planned MCP Tools

```python
@server.tool("get_recent_games")
async def get_recent_games(team_name: str, count: int = 5) -> dict:
    """Get recent games for a team."""
    # Query processed CSV data
    # Return structured results

@server.tool("get_league_standings") 
async def get_league_standings(league: str) -> dict:
    """Get current league standings."""
    # Calculate standings from match data
    # Return formatted table

@server.tool("find_referee_stats")
async def find_referee_stats(referee_name: str) -> dict:
    """Get referee statistics."""
    # Analyze referee performance data
    # Return insights
```

## Troubleshooting

### Common Issues

**ChromeDriver not found:**
```bash
# Check if ChromeDriver is in PATH
which chromedriver

# Install if missing (Ubuntu)
sudo apt install chromium-chromedriver
```

**Download timeouts:**
- Increase `download_timeout` in scraper
- Check internet connection
- TBF website might be slow

**Encoding issues:**
- All files use UTF-8 encoding
- Turkish characters should display correctly
- Check your terminal encoding

**Missing Excel files:**
- TBF button selectors might have changed
- Run test script to debug
- Check scraper logs

### Debug Mode

Run scraper in debug mode:

```python
# In basketball_data_scraper.py
chrome_options.add_argument("--headless")  # Remove this line
# Browser will be visible for debugging
```

## Next Steps: Enhanced RAG + MCP

1. **Vector Database Integration**
   - Process CSV data into embeddings
   - Create basketball-specific collections
   - Index by teams, players, referees, dates

2. **MCP Server Development**
   - Build basketball statistics tools
   - Add real-time game analysis
   - Create match prediction capabilities

3. **Enhanced Queries**
   - Combine rules + live statistics
   - Cross-reference game events with rule violations
   - Provide contextual basketball intelligence

---

🏀 **You now have a complete basketball data pipeline!** The scraper handles the TBF website complexity, and the processed data is ready for your next-generation RAG system with live basketball intelligence.

## Scraper Comparison: Selenium vs BeautifulSoup

You now have **two scraping approaches**:

### **🌐 Selenium Scraper** (`basketball_data_scraper.py`)
- **Pros**: Handles JavaScript, clicks buttons, robust for complex sites
- **Cons**: Slower, requires Chrome/ChromeDriver, uses more resources
- **Best for**: Sites with dynamic content, button-triggered downloads

### **📊 BeautifulSoup Scraper** (`basketball_scraper_bs4.py`) 
- **Pros**: Much faster, lightweight, no browser needed
- **Cons**: Can't handle JavaScript, direct URL parsing only
- **Best for**: Direct download links, simple HTML parsing

### **Test Both Approaches**

```bash
# Compare both scrapers on one league
python scripts/compare_scrapers.py
```

This will test both methods and recommend which works better for TBF.

## Quick Start Options

### **Option 1: Test Setup First**
```bash
python scripts/test_scraper_setup.py
```

### **Option 2: Try BeautifulSoup (Recommended)**
```bash
python scripts/basketball_scraper_bs4.py
```

### **Option 3: Use Selenium (Fallback)**
```bash
python scripts/basketball_data_scraper.py
```

### **Option 4: Full Pipeline (Uses Selenium)**
```bash
python scripts/scraper_launcher.py
```

### **Option 5: Compare Both**
```bash
python scripts/compare_scrapers.py
```

---

🏀 **You now have a complete basketball data pipeline with multiple scraping strategies!** 

**Recommended workflow:**
1. `python scripts/test_scraper_setup.py` - Validate setup
2. `python scripts/compare_scrapers.py` - Test both approaches  
3. Use whichever scraper works better for TBF's current website structure 