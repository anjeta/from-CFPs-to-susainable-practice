#!/usr/bin/env python3
"""
sustainability_cfp_paper_scanner.py
Prototype script to:
 - fetch CFP pages for conferences
 - fetch paper metadata (titles + abstracts) via DBLP / arXiv / Crossref (best-effort)
 - scan CFP text and paper metadata for provided sustainability keywords (lexicon + taxonomy)
 - output a CSV with per-conference CFP hits and per-conference paper hits

Usage:
    python sustainability_cfp_paper_scanner.py --conflist conferences.csv --out results.csv

Input conferences.csv should have columns:
    Conference, Acronym, CFP_URL, Year (optional), Notes (optional)

Output CSV columns:
    Conference, Acronym, Year, CFP_URL,
    CFP_keywords_found (semicolon-separated),
    CFP_matched_nodes (semicolon-separated taxonomy nodes inferred),
    Num_papers_scanned,
    Num_papers_with_matches,
    Paper_matches_summary (JSON string or semicolon-separated title|matched_keywords),
    Last_updated
"""

import re
import csv
import json
import time
import argparse
from urllib.parse import quote_plus
from collections import defaultdict
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import regex  # better Unicode handling (for CO₂)
import os
import uuid
import io
from pdfminer.high_level import extract_text

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ---------- Seed lexicon & small mapping to taxonomy nodes -----------
SEED_LEXICON = [
    "sustain*", "green", "carbon", "CO2", "CO₂", "carbon-aware", "carbon emission",
    "carbon footprint", "emission", "energy", "energy-eff", "energy-eff*", "low-power",
    "power-aware", "efficien*", "LCA", "life cycle", "life-cycle", "embodied carbon",
    "renewable", "net-zero", "offset", "environment*", "eco", "climate", "climate-aware",
    "carbon accounting", "carbon reporting", "environmental impact", "carbon-report",
    "energy-report"
]

# A simple mapping from matched lexicon tokens to taxonomy nodes (you can expand)
LEXICON_TO_NODES = {
    "energy": ["Environmental", "Techniques:Energy management & optimization"],
    "energy-eff": ["Environmental", "Techniques:Energy management & optimization"],
    "low-power": ["Techniques:Protocols & low-power networking", "Domain:Devices & Hardware"],
    "carbon": ["Environmental", "Techniques:Measurement, metrics & LCA"],
    "CO2": ["Environmental", "Techniques:Measurement, metrics & LCA"],
    "CO₂": ["Environmental", "Techniques:Measurement, metrics & LCA"],
    "carbon-aware": ["Techniques:Resource allocation & scheduling", "Data, AI & Services"],
    "LCA": ["Techniques:Measurement, metrics & LCA", "Lifecycle:Manufacture & Supply Chain"],
    "embodied carbon": ["Lifecycle:Manufacture & Supply Chain"],
    "renewable": ["Techniques:Renewable energy & grid interactions"],
    "sustain": ["Sustainability Theme:Environmental"],
    "green": ["Sustainability Theme:Environmental"],
    "eco": ["Sustainability Theme:Environmental"],
    "climate": ["Sustainability Theme:Environmental", "Sustainability Theme:Resilience & Adaptation"],
    "offset": ["Sustainability Theme:Governance, Policy & Standards"],
    # ... add more mappings as needed ...
}


def compile_regex_patterns(seed_list):
    patterns = []

    for token in seed_list:
        # Convert wildcard * to a real regex wildcard for continuation of letters
        # sustain* → sustain\w*
        # energy-eff* → energy[-_ ]?eff\w*
        base = token
        # Escape everything first
        esc = regex.escape(base)
        # Replace escaped wildcard `\*` with flexible continuation: \w*
        esc = esc.replace(r"\*", r"\w*")
        # Allow separators (-, _, space) between compound words
        # carbon[-_ ]?aware
        esc = esc.replace(r"\-", r"[-_ ]?")
        esc = esc.replace(r"\ ", r"[-_ ]?")
        # Build final pattern:
        # - (?i) → case-insensitive
        # - (?<!\w) & (?!\w) → safe word boundaries (work with hyphens)
        pattern_text = rf"(?i)(?<!\w){esc}(?!\w)"
        pat = regex.compile(pattern_text)
        patterns.append((token, pat))
    # Explicit additional CO₂ variant (though it is already handled above,
    # we include it for clarity)
    patterns.append(("CO₂", regex.compile(r"(?i)(?<!\w)CO₂(?!\w)")))
    return patterns


SEED_PATTERNS = compile_regex_patterns(SEED_LEXICON)

# ---------- helper functions ----------
def fetch_html(url):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

def fetch_pdf(url):
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with io.BytesIO(resp.content) as pdf_file:
            return extract_text(pdf_file)
    except Exception:
        return ""


def fetch_url_text(url, timeout=20, headers=None):
    """Fetch page text; return plain text (best-effort)."""
    headers = headers or {"User-Agent": "sustainability-cfp-scanner/1.0 (email@example.com)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        r.raise_for_status()
        # Attempt to extract main textual content
        soup = BeautifulSoup(r.text, "lxml")
        # remove scripts and styles
        for s in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            s.extract()
        text = soup.get_text(separator="\n")
        # collapse repeated whitespace
        text = regex.sub(r"\n\s+\n", "\n", text)
        return text.strip(), r.url
    except Exception as e:
        # Caller can try Selenium fallback if needed
        return None, None

def fetch_url_text_js(
    url,
    timeout=30,
    headless=True,
    debug_dir=None,
    wait_for_selector=None,   # optional CSS selector or (By, value) tuple to wait for
    ):
    """
    Fetch fully rendered page text using Selenium. Robust to:
      - hash fragment routing (#callforPapers)
      - cookie banners
      - iframes
      - accordion/collapse elements
      - lazy loading requiring scroll
    Returns: (text, final_url)
    If debug_dir is provided, saves page HTML and a PNG screenshot there on failure/success for inspection.
    """

    # --- cheap requests fallback first (keeps your original function) ---
    try:
        text_req, final_req = fetch_url_text(url)
        if text_req and len(text_req) > 2000:
            return text_req, final_req
    except Exception:
        text_req, final_req = None, None

    # --- prepare debug dir ---
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    # --- Selenium setup ---
    chrome_opts = Options()
    if headless:
        # new headless mode flag available in recent Chrome versions
        chrome_opts.add_argument("--headless=new")
    else:
        # allow opening a visible browser for debugging
        pass

    # Make Selenium look a bit more like a normal browser
    chrome_opts.add_argument("--disable-gpu")
    chrome_opts.add_argument("--no-sandbox")
    chrome_opts.add_argument("--disable-dev-shm-usage")
    chrome_opts.add_argument("--window-size=1400,1000")
    # set a realistic user agent (optional: customize)
    chrome_opts.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    # try to reduce headless detection surface
    chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_opts.add_experimental_option("useAutomationExtension", False)

    driver = None
    try:
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_opts)
        driver.set_page_load_timeout(timeout)
        driver.get(url)

        wait = WebDriverWait(driver, 20)

        # small helper: attempt to accept cookie banners if visible (common selectors)
        cookie_selectors = [
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree')]",
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'consent')]",
            "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]",
        ]
        for sel in cookie_selectors:
            try:
                els = driver.find_elements(By.XPATH, sel)
                for e in els:
                    if e.is_displayed():
                        try:
                            e.click()
                            time.sleep(0.3)
                        except:
                            pass
            except:
                pass

        # If URL contains a fragment (hash) ensure route handler runs:
        if "#" in url:
            # set location.hash explicitly (some SPAs react to changes)
            frag = url.split("#", 1)[1]
            if frag:
                try:
                    driver.execute_script("window.location.hash = arguments[0];", frag)
                    time.sleep(0.6)
                except:
                    pass

        # If caller gave a wait_for_selector, wait for that element to appear
        if wait_for_selector:
            try:
                if isinstance(wait_for_selector, tuple) and len(wait_for_selector) == 2:
                    wait.until(EC.presence_of_element_located(wait_for_selector))
                else:
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector)))
            except Exception:
                # continue — we'll try other heuristics
                pass

        # Try to detect and click anchors that target the hash fragment (href="#callforPapers")
        try:
            frags = []
            if "#" in url:
                frags.append(url.split("#",1)[1].strip())
            # click any link whose href ends with the fragment
            for f in frags:
                if not f:
                    continue
                xpath = f"//a[contains(@href, '#{f}')] | //*[contains(@data-target, '#{f}')]"
                try:
                    for el in driver.find_elements(By.XPATH, xpath):
                        try:
                            if el.is_displayed():
                                el.click()
                                time.sleep(0.3)
                        except:
                            pass
                except:
                    pass
        except:
            pass

        # If content is inside an iframe, try to find an iframe that includes our fragment id or likely content
        def try_switch_to_content_frame():
            try:
                frames = driver.find_elements(By.TAG_NAME, "iframe")
                for frame in frames:
                    try:
                        src = frame.get_attribute("src") or ""
                        # Heuristic: if src or name contains 'cfp' or fragment or is not empty, attempt switch
                        if "cfp" in src.lower() or "#" in src or src.strip() != "":
                            driver.switch_to.frame(frame)
                            return True
                    except:
                        continue
            except:
                pass
            return False

        switched = try_switch_to_content_frame()

        # Auto-click common expanders and accordions (buttons/links with typical words)
        expand_keywords = ["more", "expand", "details", "show", "read", "view", "abstract", "open"]
        # Find candidate elements (buttons, links, elements with role=button)
        candidates = driver.find_elements(By.XPATH, "//button | //a | //*[@role='button']")
        for el in candidates:
            try:
                label = (el.text or el.get_attribute("aria-label") or "").lower()
                if any(k in label for k in expand_keywords):
                    try:
                        if el.is_displayed():
                            driver.execute_script("arguments[0].scrollIntoView({behavior:'auto', block:'center'});", el)
                            time.sleep(0.12)
                            el.click()
                            time.sleep(0.2)
                    except:
                        pass
            except:
                pass

        # Scroll slowly to bottom to trigger lazy loading
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_attempts = 0
        while scroll_attempts < 6:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(0.8)
            new_h = driver.execute_script("return document.body.scrollHeight")
            if new_h == last_height:
                scroll_attempts += 1
            else:
                last_height = new_h
                scroll_attempts = 0

        # Wait a little for dynamic content to settle
        time.sleep(0.6)

        # If still in outer frame but we earlier switched to iframe, ensure we switch back for HTML extraction
        try:
            # get page source from current context (iframe or main)
            html = driver.page_source
        except:
            html = driver.page_source

        # If the content looks empty and we had switched frames, try the main document HTML as well
        if not html or len(html) < 200:
            try:
                driver.switch_to.default_content()
                html = driver.page_source
            except:
                pass

        # Save debug files if requested
        if debug_dir:
            uid = str(uuid.uuid4())[:8]
            html_path = os.path.join(debug_dir, f"page_{uid}.html")
            png_path = os.path.join(debug_dir, f"page_{uid}.png")
            try:
                with open(html_path, "w", encoding="utf-8") as fh:
                    fh.write(html)
            except Exception:
                pass
            try:
                driver.save_screenshot(png_path)
            except Exception:
                pass

        # Clean and extract text
        soup = BeautifulSoup(html, "lxml")
        for s in soup(["script", "style", "header", "footer", "nav", "noscript"]):
            s.extract()
        text = soup.get_text(separator="\n")
        text = regex.sub(r"\n\s+\n", "\n", text)
        final_url = driver.current_url if driver else url

        # return
        return text.strip(), final_url

    except Exception as e:
        # On error, optionally save what we have
        if driver and debug_dir:
            try:
                uid = str(uuid.uuid4())[:8]
                driver.save_screenshot(os.path.join(debug_dir, f"error_{uid}.png"))
                with open(os.path.join(debug_dir, f"error_{uid}.html"), "w", encoding="utf-8") as fh:
                    fh.write(driver.page_source)
            except:
                pass
        # fallback to earlier requests result if available
        if text_req:
            return text_req, final_req or url
        return None, url
    finally:
        try:
            if driver:
                driver.quit()
        except:
            pass


def find_seed_matches_in_text(text, patterns=SEED_PATTERNS):
    """Return set of matched tokens and mapping token -> count/contexts."""
    matches = defaultdict(int)
    contexts = defaultdict(list)
    if not text:
        return {}, {}
    text_lower = text.lower()  # don't keep original case for context slice
    for token, pat in patterns:
        for m in pat.finditer(text_lower):
            matches[token] += 1
            start, end = m.span()
            ctx = text_lower[max(0, start-80):min(len(text_lower), end+80)]
            contexts[token].append(ctx.replace("\n", " "))
    return dict(matches), dict(contexts)

def infer_taxonomy_nodes_from_tokens(tokens):
    """Map matched seed tokens to taxonomy nodes (deduplicated)."""
    nodes = set()
    for t in tokens:
        key = t.lower().replace("*", "")
        if key in LEXICON_TO_NODES:
            for n in LEXICON_TO_NODES[key]:
                nodes.add(n)
        else:
            # attempt substring matches
            for k in LEXICON_TO_NODES:
                if k in key:
                    for n in LEXICON_TO_NODES[k]:
                        nodes.add(n)
    return sorted(nodes)


# ---------- main orchestration ----------
def process_conference_row(row, patterns=SEED_PATTERNS, paper_limit=200):
    """
    row: dict with keys: Conference, Acronym, CFP_URL, Papers_URL, Year (optional)
    returns: dict of result for CSV
    """
    domain = row.get("Domain") or row.get("domain") or ""
    rank = row.get("Rank") or row.get("rank") or ""
    conf = row.get("Conference Title") or row.get("conference title") or ""
    acr = row.get("Acronym") or row.get("acronym") or ""
    cfp_url = row.get("CFP_URL") or row.get("CFP url") or row.get("CFP") or ""
    papers_url = row.get("Papers_URL") or row.get("Papers url") or row.get("Papers") or ""
    include = row.get("Include") or ""
    result = {
        "Domain": domain,
        "Rank": rank,
        "Conference Title": conf,
        "Acronym": acr,
        "CFP_URL": cfp_url,
        "Papers_URL": papers_url,
        "CFP_keyword_counts": "",
        "CFP_matched_nodes": "",
        "Papers_keyword_counts": "",
        "Papers_matched_nodes": "",
        # "Num_papers_scanned": 0,
        # "Num_papers_with_matches": 0,
        # "Paper_matches_summary": "",
        "Include": include,
        "Last_updated": datetime.utcnow().isoformat() + "Z"
    }

    # 1) Fetch CFP page
    # optional: pass a debug folder to inspect what Selenium saw
    if cfp_url.lower().endswith(".pdf"):
        text = fetch_pdf(cfp_url)
    else:
        text, final_url = fetch_url_text_js(cfp_url, headless=True, debug_dir="./selenium_debug")

    if text:
        matches, contexts = find_seed_matches_in_text(text, patterns)
        if matches:
            result["CFP_keyword_counts"] = json.dumps(matches)
            result["CFP_matched_nodes"] = ";".join(infer_taxonomy_nodes_from_tokens(matches.keys()))
        else:
            result["CFP_keywords_found"] = ""
    else:
        # Optionally: note that the page failed to fetch
        result["CFP_fetch_error"] = "Could not fetch CFP via requests; try Selenium fallback"

    # 2) Fetch accepted papers page
    urls = str(row["Papers_URL"]).split(",")
    
    aggregated_text = ""
    urls_clean = [u.strip() for u in urls if u.strip()]

    for url in urls_clean:
        if url.lower().endswith(".pdf"):
            text = fetch_pdf(url)
        else:
            head = requests.head(url, timeout=10)
            if "pdf" in head.headers.get("content-type", "").lower():
                text = fetch_pdf(url)
            else:
                text, final_url = fetch_url_text_js(url, headless=True, debug_dir="./selenium_debug")

        aggregated_text += "\n" + text
    
    # Scan paper titles for seed lexicon
    if aggregated_text:
        matches, contexts = find_seed_matches_in_text(aggregated_text, patterns)
        if matches:
            result["Papers_keyword_counts"] = json.dumps(matches)
            result["Papers_matched_nodes"] = ";".join(infer_taxonomy_nodes_from_tokens(matches.keys()))
        else:
            # result["Papers_keywords_found"] = ""
            result["Papers_fetch_error"] = "Could not fetch Accepted Papers via requests; try Selenium fallback"

    return result


def process_all_conferences(conference_csv_path, out_csv_path, max_papers_per_conf=200):
    df = pd.read_csv(conference_csv_path)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rowd = row.to_dict()
        try:
            res = process_conference_row(rowd)
        except Exception as e:
            res = {
                "Domain": rowd.get("Domain"),
                "Rank": rowd.get("Rank"),
                "Conference Title": rowd.get("Conference Title"),
                "Acronym": rowd.get("Acronym"),
                "Include": rowd.get("Include", ""),
                "CFP_URL": rowd.get("CFP_URL", ""),
                "Papers_URL": rowd.get("Papers_URL", ""),
                "CFP_keyword_counts": "",
                "CFP_matched_nodes": "",
                "Papers_keyword_counts": "",
                "Papers_matched_nodes": "",
                # "Num_papers_scanned": 0,
                # "Num_papers_with_matches": 0,
                # "Paper_matches_summary": f"ERROR: {str(e)}",
                "Last_updated": datetime.utcnow().isoformat() + "Z"
            }
        results.append(res)
        time.sleep(1.0)  # polite delay between conferences

    # Write results CSV
    keys = [
        "Domain", "Rank", "Conference Title", "Acronym", 
        "CFP_URL", "Papers_URL",
        "CFP_keyword_counts", "CFP_matched_nodes",
        "Papers_keyword_counts", "Papers_matched_nodes",
        # "Num_papers_scanned", "Num_papers_with_matches", "Paper_matches_summary",
        "Include", "Last_updated"
    ]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})
    print(f"Wrote results to {out_csv_path}")

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan CFPs and papers for sustainability keywords")
    parser.add_argument("--conflist", type=str, default="conferences_WEB.csv", help="CSV with Conference, Acronym, CFP_URL, Year")
    parser.add_argument("--out", type=str, default="cfp_paper_matches_WEB.csv", help="output results CSV")
    args = parser.parse_args()
    process_all_conferences(args.conflist, args.out)