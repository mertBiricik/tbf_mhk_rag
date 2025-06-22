#!/usr/bin/env python3
"""
TBF Page Analyzer - Downloads and analyzes TBF league pages
"""

import requests
from bs4 import BeautifulSoup
import json
import re
from pathlib import Path
import csv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_tbf_page():
    """Analyze TBF page structure for download mechanisms."""
    
    # Setup
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    analysis_dir = Path("./stats/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load first league (BSL) for analysis
    try:
        with open('./stats/leagues.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            league = next(reader)  # Get first league (BSL)
    except Exception as e:
        print(f"❌ Error loading leagues: {e}")
        return
    
    url = league['link'].strip()
    league_name = league['acronym'].strip()
    
    print(f"🔍 Analyzing {league_name}: {url}")
    
    # Download page
    try:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        # Save HTML
        html_file = analysis_dir / f"{league_name}_page.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"💾 Saved HTML: {html_file}")
        print(f"📄 Page size: {len(response.text):,} characters")
        
    except Exception as e:
        print(f"❌ Error downloading page: {e}")
        return
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Analysis results
    analysis = {
        'league': league_name,
        'url': url,
        'timestamp': datetime.now().isoformat(),
        'page_title': soup.title.string if soup.title else "No title",
        'findings': {
            'excel_mentions': [],
            'buttons': [],
            'links': [],
            'onclick_handlers': [],
            'javascript_snippets': [],
            'forms': []
        }
    }
    
    print(f"\n🔍 ANALYZING PAGE STRUCTURE")
    print("="*40)
    
    # 1. Find text mentioning "excel"
    excel_text = soup.find_all(string=re.compile(r'excel', re.IGNORECASE))
    for text in excel_text[:5]:  # First 5 mentions
        parent = text.parent if text.parent else None
        analysis['findings']['excel_mentions'].append({
            'text': str(text).strip()[:100],
            'parent_tag': parent.name if parent else None,
            'parent_attrs': dict(parent.attrs) if parent else None
        })
    
    print(f"📝 Excel mentions: {len(excel_text)}")
    
    # 2. Find all buttons
    buttons = soup.find_all(['button', 'input'])
    for button in buttons:
        btn_info = {
            'tag': button.name,
            'type': button.get('type'),
            'class': button.get('class'),
            'id': button.get('id'),
            'onclick': button.get('onclick'),
            'text': button.get_text(strip=True)[:50],
            'value': button.get('value')
        }
        analysis['findings']['buttons'].append(btn_info)
        
        # Print interesting buttons
        if any(keyword in str(btn_info).lower() 
               for keyword in ['excel', 'download', 'indir', 'export']):
            print(f"🔘 Button: {btn_info}")
    
    print(f"🔘 Total buttons: {len(buttons)}")
    
    # 3. Find links
    links = soup.find_all('a', href=True)
    excel_links = []
    for link in links:
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        if any(keyword in href.lower() or keyword in text.lower() 
               for keyword in ['excel', 'xls', 'download', 'indir', 'export']):
            link_info = {
                'href': href,
                'text': text[:50],
                'class': link.get('class'),
                'onclick': link.get('onclick')
            }
            excel_links.append(link_info)
            analysis['findings']['links'].append(link_info)
    
    print(f"🔗 Excel-related links: {len(excel_links)}")
    for link in excel_links[:3]:
        print(f"   🔗 '{link['text']}' -> {link['href']}")
    
    # 4. Find onclick handlers
    onclick_elements = soup.find_all(attrs={'onclick': True})
    for element in onclick_elements:
        onclick_info = {
            'tag': element.name,
            'onclick': element.get('onclick'),
            'class': element.get('class'),
            'id': element.get('id'),
            'text': element.get_text(strip=True)[:50]
        }
        analysis['findings']['onclick_handlers'].append(onclick_info)
        
        # Print interesting onclick handlers
        onclick_text = element.get('onclick', '').lower()
        if any(keyword in onclick_text for keyword in ['excel', 'download', 'export']):
            print(f"👆 Onclick: <{element.name}> {element.get('onclick')[:100]}...")
    
    print(f"👆 Onclick handlers: {len(onclick_elements)}")
    
    # 5. Find JavaScript that might handle downloads
    scripts = soup.find_all('script')
    js_count = 0
    for i, script in enumerate(scripts):
        if script.string:
            content = script.string.lower()
            if any(keyword in content for keyword in ['excel', 'download', 'export', 'indir']):
                js_count += 1
                
                # Save interesting JS
                js_file = analysis_dir / f"{league_name}_script_{i}.js"
                with open(js_file, 'w', encoding='utf-8') as f:
                    f.write(script.string)
                
                analysis['findings']['javascript_snippets'].append({
                    'script_index': i,
                    'file_saved': str(js_file),
                    'size': len(script.string)
                })
    
    print(f"📜 JavaScript files with download keywords: {js_count}")
    
    # 6. Find forms
    forms = soup.find_all('form')
    for form in forms:
        form_info = {
            'action': form.get('action'),
            'method': form.get('method'),
            'class': form.get('class'),
            'id': form.get('id')
        }
        analysis['findings']['forms'].append(form_info)
    
    print(f"📋 Forms: {len(forms)}")
    
    # Save analysis
    analysis_file = analysis_dir / f"{league_name}_analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Analysis saved: {analysis_file}")
    
    # Summary
    print(f"\n📊 SUMMARY FOR {league_name}")
    print("="*40)
    print(f"📄 Page title: {analysis['page_title']}")
    print(f"📝 Excel mentions: {len(analysis['findings']['excel_mentions'])}")
    print(f"🔘 Buttons: {len(analysis['findings']['buttons'])}")
    print(f"🔗 Excel links: {len(analysis['findings']['links'])}")
    print(f"👆 Onclick handlers: {len(analysis['findings']['onclick_handlers'])}")
    print(f"📜 JS files: {len(analysis['findings']['javascript_snippets'])}")
    print(f"📋 Forms: {len(analysis['findings']['forms'])}")
    
    print(f"\n📁 Files saved to: {analysis_dir}")
    print(f"   📄 {league_name}_page.html - Full page HTML")
    print(f"   📊 {league_name}_analysis.json - Structured analysis")
    print(f"   📜 {league_name}_script_*.js - JavaScript files")
    
    return analysis

if __name__ == "__main__":
    print("🔍 TBF PAGE STRUCTURE ANALYZER")
    print("="*40)
    analyze_tbf_page() 