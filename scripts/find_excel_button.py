#!/usr/bin/env python3
"""
Find Excel Button - Analyze TBF page to understand the Excel download mechanism
"""

import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path
import json

def analyze_page_for_excel_button():
    """Analyze the BSL page to understand the Excel download mechanism."""
    
    # Use the saved HTML from our previous analysis
    html_file = Path("./stats/analysis/bsl_page.html")
    
    if not html_file.exists():
        print("❌ HTML file not found. Run analyze_tbf_page.py first.")
        return
    
    print("🔍 Analyzing TBF page structure for Excel buttons...")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # 1. Find all buttons
    all_buttons = soup.find_all('button')
    print(f"\n📊 Found {len(all_buttons)} buttons on the page")
    
    excel_buttons = []
    
    # 2. Look for Excel-related buttons
    for i, button in enumerate(all_buttons):
        button_text = button.get_text(strip=True)
        button_classes = button.get('class', [])
        button_id = button.get('id', '')
        
        # Check if this button is Excel-related
        is_excel = any([
            'excel' in button_text.lower(),
            'aktar' in button_text.lower(),
            any('excel' in cls.lower() for cls in button_classes),
            'fa-file-excel-o' in str(button)
        ])
        
        if is_excel or 'Excel' in button_text:
            excel_buttons.append({
                'index': i,
                'text': button_text,
                'classes': button_classes,
                'id': button_id,
                'html': str(button)[:200] + '...' if len(str(button)) > 200 else str(button)
            })
    
    print(f"\n🎯 Found {len(excel_buttons)} Excel-related buttons:")
    for btn in excel_buttons:
        print(f"  Button {btn['index']}: '{btn['text']}'")
        print(f"    Classes: {btn['classes']}")
        print(f"    ID: {btn['id']}")
        print(f"    HTML: {btn['html']}")
        print()
    
    # 3. Look for DataTables configuration
    print("\n🔍 Searching for DataTables configuration...")
    
    # Find script tags containing DataTable
    datatable_scripts = soup.find_all('script', string=re.compile(r'DataTable', re.IGNORECASE))
    
    for script in datatable_scripts:
        script_content = script.string
        if script_content and 'excelHtml5' in script_content:
            print("✅ Found DataTables Excel configuration!")
            
            # Extract the relevant part
            lines = script_content.split('\n')
            excel_lines = []
            in_buttons_section = False
            
            for line in lines:
                if 'buttons:' in line or 'Buttons(' in line:
                    in_buttons_section = True
                
                if in_buttons_section:
                    excel_lines.append(line.strip())
                    if 'excelHtml5' in line or "Excel'e Aktar" in line:
                        excel_lines.extend([lines[lines.index(line) + j].strip() for j in range(1, 5) if lines.index(line) + j < len(lines)])
                
                if in_buttons_section and ('}]' in line or '});' in line):
                    break
            
            print("📄 Excel button configuration:")
            for line in excel_lines[-10:]:  # Show last 10 relevant lines
                if line.strip():
                    print(f"    {line}")
            break
    
    # 4. Look for the actual button in HTML
    print("\n🔍 Looking for actual button elements...")
    
    # Search for buttons with specific classes
    dt_buttons = soup.find_all(['button', 'a'], class_=re.compile(r'buttons-excel|dt-button'))
    if dt_buttons:
        print(f"✅ Found {len(dt_buttons)} DataTables buttons:")
        for btn in dt_buttons:
            print(f"  {btn}")
    
    # Search for buttons with Excel text
    excel_text_buttons = soup.find_all('button', string=re.compile(r'Excel', re.IGNORECASE))
    if excel_text_buttons:
        print(f"✅ Found {len(excel_text_buttons)} buttons with Excel text:")
        for btn in excel_text_buttons:
            print(f"  {btn}")
    
    # Search for buttons containing fa-file-excel-o icon
    excel_icon_buttons = soup.find_all('button', string=re.compile(r'fa-file-excel-o'))
    if excel_icon_buttons:
        print(f"✅ Found {len(excel_icon_buttons)} buttons with Excel icon:")
        for btn in excel_icon_buttons:
            print(f"  {btn}")
    
    # 5. Search for div containers that might hold the buttons
    dt_button_divs = soup.find_all('div', class_=re.compile(r'dt-buttons'))
    print(f"\n📦 Found {len(dt_button_divs)} DataTables button containers:")
    for div in dt_button_divs:
        print(f"  {div}")
    
    # 6. Generate Selenium selectors based on findings
    print("\n🎯 Recommended Selenium selectors:")
    
    selectors = [
        "//button[contains(@class, 'buttons-excel')]",
        "//button[contains(@class, 'dt-button') and contains(text(), 'Excel')]",
        "//div[contains(@class, 'dt-buttons')]//button[contains(text(), 'Excel')]",
        "//button[.//i[contains(@class, 'fa-file-excel-o')]]",
        "//button[contains(text(), 'Excel') and contains(text(), 'Aktar')]",
        "//a[contains(@class, 'buttons-excel')]",
    ]
    
    for i, selector in enumerate(selectors, 1):
        print(f"  Strategy {i}: {selector}")
    
    # 7. Summary
    print(f"\n📋 ANALYSIS SUMMARY:")
    print(f"  • Total buttons found: {len(all_buttons)}")
    print(f"  • Excel-related buttons: {len(excel_buttons)}")
    print(f"  • DataTables buttons found: {len(dt_buttons)}")
    print(f"  • Button containers found: {len(dt_button_divs)}")
    
    if excel_buttons:
        print(f"\n✅ MOST LIKELY EXCEL BUTTON:")
        main_button = excel_buttons[0]
        print(f"  Text: '{main_button['text']}'")
        print(f"  Classes: {main_button['classes']}")
        print(f"  Recommended selector: //button[contains(@class, '{main_button['classes'][0] if main_button['classes'] else 'dt-button'}')]")
    
    return excel_buttons

if __name__ == "__main__":
    analyze_page_for_excel_button() 