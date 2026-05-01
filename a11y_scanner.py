import os
from bs4 import BeautifulSoup
# Run 'pip install beautifulsoup4 colorama' before running

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore: RED = ''; GREEN = ''; YELLOW = ''; CYAN = ''
    class Style: RESET_ALL = ''; BRIGHT = ''

def scan_html_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    issues = []
    
    # 1. Detect Missing Alt Text
    for img in soup.find_all('img'):
        if not img.get('alt'):
            issues.append(f"{Fore.RED}[Error]{Style.RESET_ALL} <img> tag missing 'alt' attribute.")

    # 2. Detect Bad Buttons (Divs/Spans acting as buttons without tabindex)
    for elem in soup.find_all(['div', 'span']):
        if elem.get('role') == 'button' and not elem.get('tabindex'):
            issues.append(f"{Fore.RED}[Error]{Style.RESET_ALL} <{elem.name}> with role='button' is missing 'tabindex'. Keyboard users cannot reach it.")

    # 3. Detect Missing Form Labels
    for input_field in soup.find_all('input'):
        input_type = input_field.get('type', '')
        if input_type not in ['submit', 'button', 'hidden']:
            input_id = input_field.get('id')
            has_label = False
            if input_id:
                has_label = bool(soup.find('label', {'for': input_id}))
            if not has_label and not input_field.get('aria-label'):
                issues.append(f"{Fore.RED}[Error]{Style.RESET_ALL} <input type='{input_type}'> missing associated <label> or 'aria-label'.")

    if issues:
        print(f"\n{Fore.CYAN}Scanning:{Style.RESET_ALL} {filepath}")
        for issue in issues:
            print(f"  - {issue}")
            
def run_scanner(directory):
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}Starting Semantic Accessibility Scan in: {directory}{Style.RESET_ALL}")
    found_files = False
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.html'):
                found_files = True
                scan_html_file(os.path.join(root, file))
    
    if not found_files:
        print(f"{Fore.GREEN}No HTML files found to scan.{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Scan Complete.{Style.RESET_ALL}\n")

if __name__ == "__main__":
    import sys
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    run_scanner(target_dir)
