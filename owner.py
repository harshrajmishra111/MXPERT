from __future__ import annotations
import os
import json
import time
from typing import Any, Dict, List
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from flask import Blueprint, jsonify, request, current_app
from flask_login import login_required, current_user

# Blueprint for ownership-related endpoints, as per project structure.
owner_details_bp = Blueprint('owner_details_bp', __name__)

# ======================================================================
# FLASK API ROUTE INTEGRATION
# ======================================================================

@owner_details_bp.route("/api/owner_details", methods=['POST'])
@login_required
def handle_owner_details_request():
   """
   Endpoint to receive company data, process it through the ownership
   analysis pipeline, and return the enriched results.
   """
   try:
       request_data = request.get_json()
       if not request_data or "companies" not in request_data:
           current_app.logger.warning("Invalid request format received for owner details.")
           return jsonify({"error": "Invalid request format. Expected {'companies': [...]}."}), 400

       companies = request_data.get("companies", [])
       current_app.logger.info(f"User {current_user.user_id} requested analysis for {len(companies)} companies.")

       all_results = []
       for company in companies:
          
           company_name = company.get("name")
           city = company.get("city", "")
           state = company.get("state", "")
           location = f"{city}, {state}".strip(", ")
           company_info = {"name": company_name, "location": location}
          
           # Call main analysis function
           result = _tier2_analysis(company_info)
           all_results.append(result)
      
       return jsonify({"success": True, "data": all_results}), 200

   except Exception as e:
       current_app.logger.error(f"An unexpected error occurred in handle_owner_details_request: {str(e)}")
       return jsonify({"error": "Internal server error"}), 500

# ======================================================================
# ANALYSIS LOGIC
# ======================================================================

def _load_dotenv():
  
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

_load_dotenv()
API_KEY_ENV_VAR = "GEMINI_API_KEY"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"

def _require_api_key() -> str:
   api_key = os.environ.get(API_KEY_ENV_VAR)
   if not api_key: raise RuntimeError(f"Missing {API_KEY_ENV_VAR}.")
   return api_key

def _make_model(model_name: str = DEFAULT_MODEL):
   genai.configure(api_key=_require_api_key())
   return genai.GenerativeModel(
       model_name,
       generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
   )

def search_company_info(company_name: str, location: str = "") -> List[Dict[str, str]]:
   results = []
   seen_urls = set()
   location_suffix = f" {location}" if location else ""
   try:
       search_url = "https://html.duckduckgo.com/html/"
       queries = [
           f"{company_name}{location_suffix} founder CEO owner",
           f"{company_name}{location_suffix} LinkedIn company",
           f"{company_name}{location_suffix} about team leadership",
           f'"{company_name}" {location_suffix} owner LinkedIn',
       ]
       headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
       for query in queries:
           try:
               response = requests.post(search_url, data={'q': query}, headers=headers, timeout=10)
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
               time.sleep(1)
           except Exception: continue
   except Exception: pass
   return results[:12]

def _find_owner_linkedin(owner_name: str, company_name: str, location: str = "") -> str:
    if not owner_name:
        return None
    location_suffix = f" {location}" if location else ""
    try:
        search_url = "https://html.duckduckgo.com/html/"
        query = f'"{owner_name}" "{company_name}"{location_suffix} site:www.linkedin.com/in'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.post(search_url, data={'q': query}, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for result in soup.find_all('div', class_='result')[:3]:
                title_elem = result.find('a', class_='result__a')
                if title_elem:
                    url = title_elem.get('href', '')
                    if 'linkedin.com/in/' in url:
                        if url.startswith('/'):
                            url = 'https://www.linkedin.com' + url
                        return url
        time.sleep(1)
    except Exception:
        pass
    return None

def _analysis_prompt_with_search_results(company_name: str, location: str, search_results: List[Dict[str, str]]) -> str:
   company_context = f"{company_name} ({location})" if location else company_name
   if not search_results: results_text = "No search results found. Unable to verify."
   else: results_text = "\n\n".join([f"Result {i+1}:\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r.get('url', 'N/A')}" for i, r in enumerate(search_results)])
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

def _extract_json(text: str) -> Dict[str, Any]:
   """extracts a JSON object from a string, cleaning it if necessary."""
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
  
   return {
       "owner_founder": None, "parent_company": None, "confidence": "low",
       "sources_found": "Failed to parse response"
   }

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

   if not isinstance(affiliated, list):
       affiliated = []
  
   valid_ownership = {"privately_held_individual", "startup_venture_backed", "corporate_subsidiary", "publicly_listed", "private_equity_owned", "government_entity", "non_profit", "other"}
   if ownership_type not in valid_ownership:
       ownership_type = "other"
      
   if conf not in {"high", "medium", "low"}:
       conf = "low"

   return {
       "owner_founder": owner, "company_linkedin": company_linkedin, "founder_linkedin": founder_linkedin,
       "parent_company": parent, "affiliated_companies": affiliated, "ownership_type": ownership_type,
       "confidence": conf, "sources_found": sources
   }


def _tier2_analysis(company_info: Dict[str, str]) -> Dict[str, Any]:
   """Main analysis orchestrator for a single company."""
   company_name = company_info.get("name", "")
   location = company_info.get("location", "")
   display_name = f"{company_name} ({location})" if location else company_name
   current_app.logger.info(f"  - Analyzing: {display_name}")
  
   search_results = search_company_info(company_name, location)
  
   if not search_results:
       current_app.logger.info(f"    -> No search results found for {display_name}.")
       return {
           "company": company_name, "location": location, "owner_founder": None,
           "parent_company": None, "confidence": "low", "error": "No search results found",
           "sources_found": "No web results"
       }
  
   try:
       model = _make_model()
       prompt = _analysis_prompt_with_search_results(company_name, location, search_results)
       response = model.generate_content(prompt)
       text = (response.text or "").strip()
       parsed = _extract_json(text)
       normalized = _coerce_output(parsed)
       
       # Additional step: If owner found but no founder LinkedIn, search specifically
       owner = normalized.get("owner_founder")
       founder_linkedin = normalized.get("founder_linkedin")
       if owner and not founder_linkedin:
           current_app.logger.info(f"    -> Enhancing with targeted owner LinkedIn search for {owner}")
           targeted_linkedin = _find_owner_linkedin(str(owner), company_name, location)
           if targeted_linkedin:
               normalized["founder_linkedin"] = targeted_linkedin
               if normalized["confidence"] in ["low", "medium"]:
                   normalized["confidence"] = "medium"
               current_app.logger.info(f"    -> Found targeted LinkedIn: {targeted_linkedin}")
       
       return {"company": company_name, "location": location, **normalized}
      
   except Exception as e:
       current_app.logger.error(f"    -> ERROR: Gemini analysis failed for {display_name}: {e}")
       return {
           "company": company_name, "location": location, "owner_founder": None,
           "parent_company": None, "confidence": "low", "error": str(e),
           "sources_found": f"Found {len(search_results)} results but analysis failed"
       }