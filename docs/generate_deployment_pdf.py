"""
DRDO SIH 26054 Deployment Roadmap PDF Generator
=================================================
Converts docs/Simulation_to_Real_Deployment_Roadmap.md to a styled HTML document
and renders a pixel-perfect PDF using headless browser automation (Edge / Chrome).
"""

import sys
import os
import subprocess
import re
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
MD_FILE = DOCS_DIR / "Simulation_to_Real_Deployment_Roadmap.md"
HTML_FILE = DOCS_DIR / "Simulation_to_Real_Deployment_Roadmap.html"
PDF_FILE = DOCS_DIR / "Simulation_to_Real_Deployment_Roadmap.pdf"

BROWSER_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
]

def find_browser():
    for p in BROWSER_PATHS:
        if os.path.exists(p):
            return p
    return None

def md_to_html(md_content):
    """Converts markdown content to styled HTML with CSS page breaks."""
    # Convert headers
    lines = md_content.splitlines()
    html_lines = []
    in_table = False
    table_lines = []

    def flush_table(t_lines):
        if not t_lines:
            return ""
        res = ["<table class='doc-table'>"]
        header = True
        for line in t_lines:
            if "---" in line:
                continue
            cols = [c.strip() for c in line.strip('|').split('|')]
            tag = "th" if header else "td"
            row = "".join([f"<{tag}>{col}</{tag}>" for col in cols])
            res.append(f"<tr>{row}</tr>")
            header = False
        res.append("</table>")
        return "\n".join(res)

    for line in lines:
        # Check page division headers (## 1. Title Page, ## 2. Why..., etc.)
        if line.startswith("## ") and any(p in line for p in ["1. Title Page", "2. Why", "3. Current", "4. Real Deployment", "5. Gap Analysis", "6. Migration Roadmap", "7. Evaluator Talking Points", "8. Executive Summary"]):
            if in_table:
                html_lines.append(flush_table(table_lines))
                table_lines = []
                in_table = False
            html_lines.append('<div class="page-break"></div>')
            line_txt = line[3:].strip()
            html_lines.append(f'<h2 class="section-heading">{line_txt}</h2>')
            continue

        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            html_lines.append(flush_table(table_lines))
            table_lines = []
            in_table = False

        if line.startswith("# "):
            html_lines.append(f'<h1 class="main-title">{line[2:].strip()}</h1>')
        elif line.startswith("### "):
            html_lines.append(f'<h3 class="sub-heading">{line[4:].strip()}</h3>')
        elif line.startswith("#### "):
            html_lines.append(f'<h4 class="sub-sub-heading">{line[5:].strip()}</h4>')
        elif line.startswith("> "):
            callout = line[2:].strip()
            if callout.startswith("###"):
                callout = f"<strong>{callout[3:].strip()}</strong>"
            html_lines.append(f'<div class="callout-box">{callout}</div>')
        elif line.startswith("```"):
            if "```" in line and len(line) > 3:
                html_lines.append('<pre class="code-block"><code>')
            elif line.strip() == "```":
                html_lines.append('</code></pre>')
        elif line.startswith("- ") or line.startswith("* "):
            content = line[2:].strip()
            # bold lead in formatting
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            html_lines.append(f'<li class="bullet-item">{content}</li>')
        elif line.strip() == "---":
            html_lines.append('<hr class="divider"/>')
        elif line.strip():
            content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
            content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
            content = re.sub(r'`(.*?)`', r'<code class="inline-code">\1</code>', content)
            html_lines.append(f'<p class="body-text">{content}</p>')

    if in_table:
        html_lines.append(flush_table(table_lines))

    body_html = "\n".join(html_lines)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DRDO SIH 26054 Deployment Roadmap</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 18mm 15mm 18mm 15mm;
  }}
  @media print {{
    .page-break {{
      page-break-before: always;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    line-height: 1.5;
    font-size: 10.5pt;
    background: #ffffff;
  }}
  .main-title {{
    font-size: 20pt;
    color: #0284c7;
    margin-top: 0;
    margin-bottom: 8px;
    font-weight: 700;
  }}
  .section-heading {{
    font-size: 16pt;
    color: #0f172a;
    border-bottom: 2px solid #0284c7;
    padding-bottom: 4px;
    margin-top: 15px;
    margin-bottom: 12px;
  }}
  .sub-heading {{
    font-size: 12.5pt;
    color: #0369a1;
    margin-top: 12px;
    margin-bottom: 6px;
  }}
  .sub-sub-heading {{
    font-size: 11pt;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
  }}
  .body-text {{
    margin-top: 4px;
    margin-bottom: 6px;
    color: #1e293b;
  }}
  .bullet-item {{
    margin-bottom: 4px;
    color: #1e293b;
  }}
  .callout-box {{
    background: #f0f9ff;
    border-left: 4px solid #0284c7;
    padding: 10px 14px;
    margin: 10px 0;
    border-radius: 4px;
    font-size: 10pt;
    color: #0369a1;
  }}
  .code-block {{
    background: #0f172a;
    color: #38bdf8;
    padding: 10px 14px;
    border-radius: 6px;
    font-family: "Courier New", Courier, monospace;
    font-size: 8.5pt;
    line-height: 1.35;
    overflow-x: auto;
    white-space: pre;
    margin: 10px 0;
  }}
  .inline-code {{
    background: #f1f5f9;
    color: #0284c7;
    padding: 2px 5px;
    border-radius: 3px;
    font-family: monospace;
    font-size: 9.5pt;
  }}
  .doc-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9.5pt;
  }}
  .doc-table th {{
    background: #0f172a;
    color: #f8fafc;
    padding: 7px 10px;
    text-align: left;
    font-weight: 600;
  }}
  .doc-table td {{
    border-bottom: 1px solid #e2e8f0;
    padding: 6px 10px;
    color: #1e293b;
  }}
  .doc-table tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .divider {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 15px 0;
  }}
  strong {{
    color: #0f172a;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""
    return full_html

def generate_pdf():
    if not MD_FILE.exists():
        print(f"[ERROR] Markdown file not found at: {MD_FILE}")
        return False

    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_content = md_to_html(md_text)
    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Generated styled HTML at: {HTML_FILE}")

    browser_exe = find_browser()
    if not browser_exe:
        print("[WARNING] Neither Edge nor Chrome executable found for headless printing.")
        return False

    print(f"[INFO] Using browser executable: {browser_exe}")
    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={PDF_FILE}",
        str(HTML_FILE)
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if PDF_FILE.exists():
        print(f"[SUCCESS] Successfully generated PDF document at: {PDF_FILE}")
        return True
    else:
        print(f"[ERROR] Failed to generate PDF. Browser error output:\n{res.stderr}")
        return False

if __name__ == "__main__":
    generate_pdf()
