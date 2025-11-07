# ============================================================
# IMPORTS
# ============================================================
from __future__ import annotations
import os
import json
import time
import random
from typing import Any, Dict, List
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user

owner_details_bp = Blueprint('owner_details_bp', __name__)

# ============================================================
# SETUP
# ============================================================

# --- Environment Variable Loading ---
def _load_dotenv():
   """Loads environment variables from a .env file."""
   env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
   if not os.path.isfile(env_path): return
   try:
       with open(env_path, "r", encoding="utf-8") as f:
           for raw in f:
               line = raw.strip()
               if not line or line.startswith("#") or "=" not in line: continue
               key, value = line.split("=", 1)
               key, value = key.strip(), value.strip().strip('"').strip("'")
               if key and key not in os.environ: os.environ[key] = value
   except Exception: pass

_load_dotenv() # Load variables immediately

# --- API Key and Model Configuration ---
API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

def _require_api_key() -> str:
   """Checks for and returns the Gemini API key."""
   api_key = os.environ.get(API_KEY_ENV_VAR)
   if not api_key: raise RuntimeError(f"Missing {API_KEY_ENV_VAR}.")
   return api_key

def _make_model(model_name: str = DEFAULT_MODEL):
   """Initializes the Gemini GenerativeModel."""
   genai.configure(api_key=_require_api_key())
   return genai.GenerativeModel(
       model_name,
       generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
   )

# --- Global Instances ---
_GLOBAL_GEMINI_MODEL = _make_model()

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

SEARCH_URL = "https://html.duckduckgo.com/html/"

# ============================================================
# UTILITIES
# ============================================================

def _extract_json(text: str) -> Dict[str, Any]:
   """Safely extracts a JSON object from a string, cleaning it if necessary."""
   try:
       return json.loads(text)
   except Exception:
       pass

   cleaned = text.strip()
   if cleaned.startswith("```"):
       cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))

   start = cleaned.find("{")
   end = cleaned.rfind("}")
   if start != -1 and end != -1 and end > start:
       candidate = cleaned[start : end + 1]
       try:
           return json.loads(candidate)
       except Exception:
           pass

   # Default error response if parsing fails
   return {
       "owner_founder": None, "parent_company": None, "confidence": "low",
       "sources_found": "Failed to parse response"
   }

def _coerce_output(obj: Dict[str, Any]) -> Dict[str, Any]:
   """Ensures the final JSON object has the correct data types and structure."""
   def _nullify(v: Any) -> Any:
       # Converts empty/placeholder strings to None
       return None if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "n/a", "na"} else v

   owner = _nullify(obj.get("owner_founder"))
   company_linkedin = _nullify(obj.get("company_linkedin"))
   founder_linkedin = _nullify(obj.get("founder_linkedin"))
   parent = _nullify(obj.get("parent_company"))
   affiliated = obj.get("affiliated_companies", [])
   ownership_type = str(obj.get("ownership_type", "other")).strip().lower()
   conf = str(obj.get("confidence", "low")).strip().lower()
   sources = obj.get("sources_found", "")

   # Ensure affiliated_companies is a list
   if not isinstance(affiliated, list):
       affiliated = []

   # Validate ownership_type
   valid_ownership = {"privately_held_individual", "startup_venture_backed", "corporate_subsidiary", "publicly_listed", "private_equity_owned", "government_entity", "non_profit", "other"}
   if ownership_type not in valid_ownership:
       ownership_type = "other"

   # Validate confidence level
   if conf not in {"high", "medium", "low"}:
       conf = "low"

   # Return the cleaned dictionary
   return {
       "owner_founder": owner, "company_linkedin": company_linkedin, "founder_linkedin": founder_linkedin,
       "parent_company": parent, "affiliated_companies": affiliated, "ownership_type": ownership_type,
       "confidence": conf, "sources_found": sources
   }

# ============================================================
# SYNCHRONOUS SCRAPING
# ============================================================

def search_company_info(company_name: str, location: str = "") -> List[Dict[str, str]]:
   """Performs 3 sequential DuckDuckGo searches and parses results."""
   results = []
   seen_urls = set()
   location_suffix = f" {location}" if location else ""
   try:
       queries = [
           f"{company_name}{location_suffix} founder CEO owner",
           f"{company_name}{location_suffix} LinkedIn company owner", # Modified query
           f"{company_name}{location_suffix} about team leadership",
       ]

       for query in queries:
           try:
               response = _HTTP_SESSION.post(SEARCH_URL, data={'q': query}, timeout=10)
               if response.status_code == 200:
                   soup = BeautifulSoup(response.text, 'html.parser')
                   for result in soup.find_all('div', class_='result')[:5]:
                       title_elem = result.find('a', class_='result__a')
                       snippet_elem = result.find('a', class_='result__snippet')
                       if title_elem and snippet_elem:
                           url = title_elem.get('href', '')
                           if url not in seen_urls:
                               seen_urls.add(url)
                               results.append({
                                   'title': title_elem.get_text(strip=True),
                                   'snippet': snippet_elem.get_text(strip=True),
                                   'url': url
                               })

               time.sleep(random.uniform(0.8, 1.2)) 
           except Exception as e_inner:
               current_app.logger.warning(f"Search query failed: {query}. Error: {e_inner}")
               continue # Try next query
   except Exception as e_outer:
        current_app.logger.error(f"Outer search function failed for {company_name}. Error: {e_outer}")
   return results[:12] # Return top 12 results


# ============================================================
# GEMINI CALL & ANALYSIS
# ============================================================

def _analysis_prompt_with_search_results(company_name: str, location: str, search_results: List[Dict[str, str]]) -> str:
   """Builds the prompt string for the Gemini model."""
   company_context = f"{company_name} ({location})" if location else company_name
   if not search_results:
       results_text = "No search results found. Unable to verify."
   else:
       # Format results clearly for the AI
       results_text = "\n\n".join(
           f"Result {i+1}:\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r.get('url', 'N/A')}"
           for i, r in enumerate(search_results)
       )

   # Prompt instructions (removed first line)
   return (
       # REMOVED: "I PREFER A QUICK RESPONSE.\n\n"
       f"Analyze the web search results for the company: {company_context}.\n\n"
       "## Search Results:\n"
       f"{results_text}\n\n"
       "## Instructions:\n"
       "Based ONLY on the provided search results, extract the following information into a valid JSON object. Focus heavily on owner/founder identification and their personal LinkedIn (/in/ URLs) over company pages (/company).\n\n"
       "1.  `owner_founder`: Identify the primary founder, CEO, or owner.\n"
       "2.  `company_linkedin` & `founder_linkedin`: Find any LinkedIn profile URLs. Use /company/... for company_linkedin; reserve /in/... strictly for founder_linkedin (personal profiles).\n"
       "3.  `parent_company`: If it is a subsidiary, name the parent company.\n"
       "4.  `affiliated_companies`: List other organizations the key people are associated with. This MUST be an array `[]` if none are found.\n"
       "5.  `ownership_type`: Classify as 'privately_held_individual', 'startup_venture_backed', 'corporate_subsidiary', 'publicly_listed', 'private_equity_owned', 'government_entity', 'non_profit', or 'other'.\n"
       "6.  `confidence`: Use strict rules:\n"
       "    - **'high'**: ONLY if the owner's name is in 2+ sources AND a LinkedIn URL is found.\n"
       "    - **'medium'**: If the owner is in 1 source OR LinkedIn is missing.\n"
       "    - **'low'**: If no owner is found or data is conflicting.\n"
       "7.  `sources_found`: **Summarize the top 3 most critical findings.** Each bullet point must state a key fact, focusing on a person's name and their role (like CEO or Founder), followed by the source URL. **Do not number the sources (e.g., avoid 'Source 1').**\n\n"
       "## JSON Output Format:\n"
       "Respond with a single JSON object containing these exact keys: `owner_founder`, `company_linkedin`, `founder_linkedin`, `parent_company`, `affiliated_companies`, `ownership_type`, `confidence`, `sources_found`."
   )

def _tier2_analysis(company_info: Dict[str, str]) -> Dict[str, Any]:
   """Main synchronous analysis orchestrator for a single company."""
   company_name = company_info.get("name", "")
   location = company_info.get("location", "")
   display_name = f"{company_name} ({location})" if location else company_name
   current_app.logger.info(f"  - Analyzing: {display_name}")
   search_results = [] # Define for potential use in error message

   try:
       # Step 1: Perform synchronous search
       search_results = search_company_info(company_name, location)

       # Handle no results
       if not search_results:
           current_app.logger.info(f"    -> No search results found for {display_name}.")
           return {
               "company": company_name, "location": location, "owner_founder": None,
               "parent_company": None, "confidence": "low", "error": "No search results found",
               "sources_found": "No web results"
           }

       # Step 2: Call Gemini (synchronously)
       prompt = _analysis_prompt_with_search_results(company_name, location, search_results)
       # Use the global model instance
       response = _GLOBAL_GEMINI_MODEL.generate_content(prompt)
       text = (response.text or "").strip()

       # Step 3: Parse and clean the response
       parsed = _extract_json(text)
       normalized = _coerce_output(parsed) 

       # Add original company info back and return
       return {"company": company_name, "location": location, **normalized}

   except Exception as e:
       # Catch errors during the analysis process itself
       current_app.logger.error(f"    -> ERROR: Analysis failed for {display_name}: {e}")
       return {
           "company": company_name, "location": location, "owner_founder": None,
           "parent_company": None, "confidence": "low", "error": f"Analysis exception: {str(e)}",
           "sources_found": f"Found {len(search_results)} results but analysis failed"
       }

# ============================================================
# FLASK ENDPOINT
# ============================================================

@owner_details_bp.route("/api/owner_details", methods=["POST"])
@login_required
def handle_owner_details_request():
   """
   Receives company list, processes each sequentially using _tier2_analysis.
   """
   try:
       request_data = request.get_json()
       if not request_data or "companies" not in request_data:
           current_app.logger.warning("Invalid request format received for owner details.")
           return jsonify({"error": "Invalid request format. Expected {'companies': [...]}."}), 400

       companies = request_data.get("companies", [])
       user_id = getattr(current_user, "user_id", "unknown") # Safer access to user_id
       current_app.logger.info(f"User {user_id} requested analysis for {len(companies)} companies.")

       all_results = []
       # Process companies one by one (synchronously)
       for company in companies:
           company_name = company.get("name")
           # Basic validation for company name
           if not company_name:
               all_results.append({"company": None, "error": "Missing company name", "confidence": "low"})
               continue

           city = company.get("city", "")
           state = company.get("state", "")
           location = f"{city}, {state}".strip(", ") if city or state else ""
           company_info = {"name": company_name, "location": location}

           # Call main synchronous analysis function
           result = _tier2_analysis(company_info)
           all_results.append(result)

       return jsonify({"success": True, "data": all_results}), 200

   except Exception as e:
       # Catch errors in the main request handling (e.g., JSON parsing issues)
       current_app.logger.error(f"An unexpected error occurred in handle_owner_details_request: {str(e)}")
       # Return a generic 500 error to the client
       return jsonify({"error": "Internal server error"}), 500
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   # ============================================================
# IMPORTS
# ============================================================
from __future__ import annotations
import os
import json
import time
import random
from typing import Any, Dict, List, Tuple, Optional
import requests
from requests.exceptions import RequestException
import concurrent.futures
from functools import partial
import google.generativeai as genai
from flask import Blueprint, jsonify, request, current_app 
from flask_login import login_required, current_user

owner_details_bp = Blueprint('owner_details_bp', __name__)

# ============================================================
# SETUP
# ============================================================
# (Setup remains the same - _load_dotenv, API keys, model, Serper config, session)
def _load_dotenv():
   """Loads environment variables from a .env file."""
   env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
   if not os.path.isfile(env_path): return
   try:
       with open(env_path, "r", encoding="utf-8") as f:
           for raw in f:
               line = raw.strip()
               if not line or line.startswith("#") or "=" not in line: continue
               key, value = line.split("=", 1)
               key, value = key.strip(), value.strip().strip('"').strip("'")
               if key and key not in os.environ: os.environ[key] = value
   except Exception as e:
       print(f"WARNING: Failed to load .env file: {e}")

_load_dotenv()

API_KEY_ENV_VAR = "GEMINI_API_KEY"
SERPER_API_KEY_ENV_VAR = "SERPER_API_KEY"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

def _require_api_key(env_var_name: str) -> str:
   """Checks for and returns an API key from environment variables."""
   api_key = os.environ.get(env_var_name)
   if not api_key: raise RuntimeError(f"Missing required environment variable: {env_var_name}")
   return api_key

def _make_model(model_name: str = DEFAULT_MODEL):
   """Initializes the Gemini GenerativeModel using environment variable key."""
   try:
        genai.configure(api_key=_require_api_key(API_KEY_ENV_VAR))
        return genai.GenerativeModel(
            model_name,
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
        )
   except RuntimeError as e:
       print(f"CRITICAL ERROR: {e}")
       return None
   except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Gemini model: {e}")
        return None

_GLOBAL_GEMINI_MODEL = _make_model()

try:
    _SERPER_API_KEY = _require_api_key(SERPER_API_KEY_ENV_VAR)
except RuntimeError as e:
    print(f"CRITICAL ERROR: {e}")
    _SERPER_API_KEY = None

_SERPER_SEARCH_URL = "https://google.serper.dev/search"

_HTTP_SESSION = requests.Session()
_HTTP_SESSION.headers.update({'User-Agent': 'Mozilla/5.0 (FlaskServer/1.0)'})

# ============================================================
# UTILITIES
# ============================================================
# (_extract_json, _coerce_output remain the same)
def _extract_json(text: str) -> Dict[str, Any]:
   """Safely extracts a JSON object from a string, cleaning it if necessary."""
   try:
       return json.loads(text)
   except json.JSONDecodeError:
       pass
   cleaned = text.strip()
   if cleaned.startswith("```"):
       cleaned = "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("```"))
   start = cleaned.find("{")
   end = cleaned.rfind("}")
   if start != -1 and end != -1 and end > start:
       candidate = cleaned[start : end + 1]
       try:
           return json.loads(candidate)
       except json.JSONDecodeError:
           pass
   return {"owner_founder": None, "parent_company": None, "confidence": "low", "sources_found": "Failed to parse AI response"}

def _coerce_output(obj: Dict[str, Any]) -> Dict[str, Any]:
   """Ensures the final JSON object has the correct data types and structure."""
   def _nullify(v: Any) -> Any:
       return None if isinstance(v, str) and v.strip().lower() in {"", "none", "null", "n/a", "na"} else v
   owner = _nullify(obj.get("owner_founder"))
   company_linkedin = _nullify(obj.get("company_linkedin"))
   founder_linkedin = _nullify(obj.get("founder_linkedin"))
   parent = _nullify(obj.get("parent_company"))
   affiliated = obj.get("affiliated_companies", [])
   ownership_type = str(obj.get("ownership_type", "other")).strip().lower()
   conf = str(obj.get("confidence", "low")).strip().lower()
   sources = obj.get("sources_found", "")
   if not isinstance(affiliated, list): affiliated = []
   valid_ownership = {"privately_held_individual", "startup_venture_backed", "corporate_subsidiary", "publicly_listed", "private_equity_owned", "government_entity", "non_profit", "other"}
   if ownership_type not in valid_ownership: ownership_type = "other"
   if conf not in {"high", "medium", "low"}: conf = "low"
   return {"owner_founder": owner, "company_linkedin": company_linkedin, "founder_linkedin": founder_linkedin, "parent_company": parent, "affiliated_companies": affiliated, "ownership_type": ownership_type, "confidence": conf, "sources_found": sources}

# ============================================================
# SERPER SEARCH API CALL
# ============================================================
# *** ADDED logger parameter ***
def search_company_serper_api(company_name: str, location: str = "", logger=None) -> Tuple[Optional[List[Dict[str, str]]], Optional[str]]:
    """
    Performs 2 sequential Serper.dev Search API calls.
    Returns (results_list, None) or (None, error_message).
    """
    # Use passed logger, fallback to print if None (e.g., direct script run)
    log_func = logger if logger else print
    log_error = getattr(logger, 'error', print)
    log_warning = getattr(logger, 'warning', print)

    if not _SERPER_API_KEY:
        log_error("Serper API Key not configured")
        return None, "Serper API Key not configured"

    results: List[Dict[str, str]] = []
    seen_urls: set[str] = set()
    location_suffix = f" {location}" if location else ""
    queries = [
        f"{company_name}{location_suffix} founder CEO owner parent company",
        f"{company_name}{location_suffix} LinkedIn company owner",
    ]
    headers = {"X-API-KEY": _SERPER_API_KEY, "Content-Type": "application/json"}

    for query in queries:
        payload = json.dumps({"q": query, "num": 10})
        try:
            response = _HTTP_SESSION.post(_SERPER_SEARCH_URL, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            data = response.json()
            organic_results = data.get('organic', [])
            for item in organic_results:
                url, title, snippet = item.get('link'), item.get('title'), item.get('snippet')
                if url and title and snippet and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({'title': title, 'snippet': snippet, 'url': url})
            time.sleep(0.5) # Keep short sleep
        except requests.exceptions.HTTPError as http_err:
            status_code = http_err.response.status_code
            error_text = http_err.response.text
            log_error(f"Serper API HTTP error query '{query}': {status_code} - {error_text}")
            err_map = {401: "Unauthorized", 403: "Forbidden", 429: "Rate Limit/Quota Exceeded"}
            return None, f"Serper API Error: {err_map.get(status_code, status_code)}"
        except RequestException as req_err:
            log_error(f"Network error during Serper API query '{query}': {req_err}")
            return None, "Network error during search"
        except json.JSONDecodeError:
            log_error(f"Failed to decode JSON from Serper API query '{query}'")
            return None, "Invalid response from search API"
        except Exception as e_unexpected:
            log_error(f"Unexpected error during Serper API query '{query}': {e_unexpected}")
            return None, "Unexpected search error"
    return results[:20], None


# ============================================================
# GEMINI CALL & ANALYSIS
# ============================================================
def _analysis_prompt_with_search_results(company_name: str, location: str, search_results: List[Dict[str, str]]) -> str:
   """Builds the prompt string using your exact original format."""
   company_context = f"{company_name} ({location})" if location else company_name
   if not search_results:
       results_text = "No search results found. Unable to verify."
   else:
       results_text = "\n\n".join(
           f"Result {i+1}:\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r.get('url', 'N/A')}"
           for i, r in enumerate(search_results)
       )
   return (
       "I PREFER A QUICK RESPONSE.\n\n"
       f"Analyze the web search results for the company: {company_context}.\n\n"
       "## Search Results:\n"
       f"{results_text}\n\n"
       "## Instructions:\n"
       "Based ONLY on the provided search results, extract the following information into a valid JSON object. Focus heavily on owner/founder identification and their personal LinkedIn (/in/ URLs) over company pages (/company).\n\n"
       "1.  `owner_founder`: Identify the primary founder, CEO, or owner.\n"
       "2.  `company_linkedin` & `founder_linkedin`: Find any LinkedIn profile URLs. Use /company/... for company_linkedin; reserve /in/... strictly for founder_linkedin (personal profiles).\n"
       "3.  `parent_company`: If it is a subsidiary, name the parent company.\n"
       "4.  `affiliated_companies`: List other organizations the key people are associated with. This MUST be an array `[]` if none are found.\n"
       "5.  `ownership_type`: Classify as 'privately_held_individual', 'startup_venture_backed', 'corporate_subsidiary', 'publicly_listed', 'private_equity_owned', 'government_entity', 'non_profit', or 'other'.\n"
       "6.  `confidence`: Use strict rules:\n"
       "    - **'high'**: ONLY if the owner's name is in 2+ sources AND a LinkedIn URL is found.\n"
       "    - **'medium'**: If the owner is in 1 source OR LinkedIn is missing.\n"
       "    - **'low'**: If no owner is found or data is conflicting.\n"
       "7.  `sources_found`: **Summarize the top 3 most critical findings.** Each bullet point must state a key fact, focusing on a person's name and their role (like CEO or Founder), followed by the source URL. **Do not number the sources (e.g., avoid 'Source 1').**\n\n"
       "## JSON Output Format:\n"
       "Respond with a single JSON object containing these exact keys: `owner_founder`, `company_linkedin`, `founder_linkedin`, `parent_company`, `affiliated_companies`, `ownership_type`, `confidence`, `sources_found`."
   )

# *** ADDED logger parameter ***
def _tier2_analysis(company_info: Dict[str, str], logger) -> Dict[str, Any]:
   """Main synchronous analysis orchestrator using Serper API."""
   # logger = current_app.logger # Logger is now passed in
   company_name = company_info.get("name", "")
   location = company_info.get("location", "")
   display_name = f"{company_name} ({location})" if location else company_name
   # *** Use passed logger ***
   logger.info(f"  - Analyzing: {display_name}")

   error_result = lambda msg: {"company": company_name, "location": location, "owner_founder": None, "parent_company": None, "confidence": "low", "error": msg, "sources_found": "Analysis failed due to error"}

   search_results = None
   search_error = None

   try:
       # *** Pass logger to search function ***
       search_results, search_error = search_company_serper_api(company_name, location, logger=logger)

       if search_error:
           logger.error(f"    -> Search failed for {display_name}: {search_error}")
           return error_result(search_error)

       if not search_results:
           logger.info(f"    -> No search results found via Serper API for {display_name}.")
           return {"company": company_name, "location": location, "owner_founder": None, "parent_company": None, "confidence": "low", "error": None, "sources_found": "No web results found via Serper API"}

       prompt = _analysis_prompt_with_search_results(company_name, location, search_results)
       try:
           if _GLOBAL_GEMINI_MODEL is None:
               logger.critical("Gemini model not initialized.")
               return error_result("Server configuration error: AI model unavailable")
           response = _GLOBAL_GEMINI_MODEL.generate_content(prompt)
           text = (response.text or "").strip()
           if not text:
               logger.warning(f"    -> Gemini returned empty response for {display_name}")
               return error_result("Gemini returned empty response")
       except Exception as gemini_e:
            logger.error(f"    -> ERROR: Gemini API call failed for {display_name}: {gemini_e}")
            return error_result("Gemini API Error")

       parsed = _extract_json(text)
       normalized = _coerce_output(parsed)

       final_result = {"company": company_name, "location": location, **normalized}
       final_result.setdefault("error", None)
       return final_result

   except Exception as e:
       # *** Use passed logger ***
       logger.error(f"    -> UNEXPECTED ERROR during analysis for {display_name}: {e}")
       return error_result(f"Unexpected analysis exception: {str(e)}")


# ============================================================
# FLASK ENDPOINT (Parallel)
# ============================================================
@owner_details_bp.route("/api/owner_details", methods=["POST"])
@login_required
def handle_owner_details_request():
   """Receives company list, processes up to 3 concurrently using Serper and Gemini."""
   # *** Get logger in main thread ***
   logger = current_app.logger

   try:
       request_data = request.get_json()
       if not request_data or "companies" not in request_data:
           logger.warning("Invalid request format received for owner details.")
           return jsonify({"error": "Invalid request format. Expected {'companies': [...]}."}), 400

       companies_input = request_data.get("companies", [])
       user_id = getattr(current_user, "user_id", "unknown") if current_user.is_authenticated else "anonymous"
       logger.info(f"User {user_id} requested analysis for {len(companies_input)} companies.")

       companies_to_process = []
       validation_errors = []
       for company_data in companies_input:
            company_name = company_data.get("name")
            if not company_name:
                validation_errors.append({"company": None, "location": None, "owner_founder": None, "parent_company": None, "confidence": "low", "error": "Missing company name", "sources_found": None})
                continue

            city = company_data.get("city", "")
            state = company_data.get("state", "")
            location = f"{city}, {state}".strip(", ") if city or state else ""
            companies_to_process.append({"name": company_name, "location": location})

       all_results = []
       # *** Use ThreadPoolExecutor again ***
       with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
           # *** Pass logger using partial ***
           analysis_func = partial(_tier2_analysis, logger=logger)
           future_results = executor.map(analysis_func, companies_to_process)
           try:
               all_results = list(future_results)
           except Exception as e:
               logger.error(f"Error occurred during parallel execution: {e}")
               return jsonify({"error": "Error during processing"}), 500

       final_data = validation_errors + all_results
       return jsonify({"success": True, "data": final_data}), 200

   except Exception as e:
       logger.error(f"An unexpected error occurred in handle_owner_details_request: {str(e)}")
       return jsonify({"error": "Internal server error"}), 500
