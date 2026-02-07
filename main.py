from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import os
from dotenv import load_dotenv
import serpapi
import numpy as np
from scipy.signal import find_peaks
import datetime
from groq import Groq
from groq.types.chat import ( ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam )
from fastapi.staticfiles import StaticFiles
import re
from urllib.parse import quote

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")  # Not used in this example, but kept for potential expansion
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def verify_song_artist(song, artist):
    """
    Verify if the song and artist combination exists using SerpAPI Google Search.
    Returns a dict with 'valid' boolean and 'message' if invalid.
    ONLY validates if the artist is the PRIMARY artist for the song.
    """
    if not song or not artist:
        return {"valid": False, "message": "Please provide both song name and artist."}
    
    # Search with both song and artist together in quotes
    search_query = f'"{song}" "{artist}" site:spotify.com OR site:music.apple.com OR site:genius.com'
    
    try:
        url = "https://serpapi.com/search"
        params = {
            "q": search_query,
            "api_key": SERPAPI_KEY,
            "num": 10
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        organic_results = data.get("organic_results", [])
        
        # If no results from music platforms, try broader search
        if not organic_results or len(organic_results) < 2:
            search_query = f'"{song}" by "{artist}"'
            params["q"] = search_query
            response = requests.get(url, params=params)
            data = response.json()
            organic_results = data.get("organic_results", [])
        
        if not organic_results or len(organic_results) < 2:
            return {
                "valid": False,
                "message": f"Song '{song}' by {artist} not found. Please check your spelling."
            }
        
        artist_lower = artist.lower().strip()
        song_lower = song.lower().strip()
        
        # Normalize by removing special characters
        artist_normalized = re.sub(r'[^\w\s]', '', artist_lower)
        song_normalized = re.sub(r'[^\w\s]', '', song_lower)
        
        # Count how many results show this artist as the PRIMARY artist
        # Look for patterns like "Song - Artist", "Artist - Song", "Song by Artist"
        primary_artist_matches = 0
        
        for result in organic_results[:10]:
            title = result.get("title", "").lower()
            link = result.get("link", "").lower()
            
            # Normalize title
            title_normalized = re.sub(r'[^\w\s]', '', title)
            
            # Skip if either song or artist not mentioned at all
            if not ((song_lower in title or song_normalized in title_normalized) and
                    (artist_lower in title or artist_normalized in title_normalized)):
                continue
            
            # Check for patterns that indicate PRIMARY artist
            # Pattern 1: "Song - Artist" or "Song – Artist" (most common)
            pattern1 = rf'{re.escape(song_lower)}\s*[-–—]\s*{re.escape(artist_lower)}'
            pattern1_norm = rf'{re.escape(song_normalized)}\s*{re.escape(artist_normalized)}'
            
            # Pattern 2: "Artist - Song"
            pattern2 = rf'{re.escape(artist_lower)}\s*[-–—]\s*{re.escape(song_lower)}'
            pattern2_norm = rf'{re.escape(artist_normalized)}\s*{re.escape(song_normalized)}'
            
            # Pattern 3: "Song by Artist"
            pattern3 = rf'{re.escape(song_lower)}\s+by\s+{re.escape(artist_lower)}'
            pattern3_norm = rf'{re.escape(song_normalized)}\s+by\s+{re.escape(artist_normalized)}'
            
            # Check if any pattern matches
            if (re.search(pattern1, title) or re.search(pattern1_norm, title_normalized) or
                re.search(pattern2, title) or re.search(pattern2_norm, title_normalized) or
                re.search(pattern3, title) or re.search(pattern3_norm, title_normalized)):
                primary_artist_matches += 1
        
        # STRICT: Require at least 2 results showing this artist as PRIMARY artist
        if primary_artist_matches >= 2:
            return {
                "valid": True,
                "matched_song": song,
                "matched_artist": artist
            }
        else:
            return {
                "valid": False,
                "message": f"Cannot verify '{song}' by {artist}. The artist may be incorrect - please check the actual artist for this song."
            }
        
    except Exception as e:
        print(f"SerpAPI verification error: {e}")
        return {
            "valid": False,
            "message": "Verification service unavailable. Please try again."
        }


def get_trends_data(query):
    params = {
        "engine": "google_trends",
        "q": query,
        "data_type": "TIMESERIES",
        "date": "today 12-m",  # Last 12 months
        "api_key": SERPAPI_KEY
    }
    search = serpapi.GoogleSearch(params)
    results = search.get_dict()
    if "interest_over_time" in results and "timeline_data" in results["interest_over_time"]:
        # Parse values to floats, handling '<1' as 0
        for item in results["interest_over_time"]["timeline_data"]:
            raw_value = item['values'][0]['value']
            if isinstance(raw_value, str) and '<' in raw_value:
                item['values'][0]['value'] = 0.0  # Or 0.5 if you prefer an average low value
            else:
                item['values'][0]['value'] = float(raw_value)
        return results["interest_over_time"]["timeline_data"]
    return []


def detect_spikes(timeline_data):
    if not timeline_data:
        return []
    values = np.array([item['values'][0]['value'] for item in timeline_data], dtype=float)  # Force float dtype
    if len(values) == 0 or np.all(values == 0):  # Avoid issues with all-zero data
        return []
    # Find peaks with a minimum prominence (adjust as needed for sensitivity)
    peaks, _ = find_peaks(values, prominence=values.std() * 2)
    spike_dates = [timeline_data[i]['date'] for i in peaks]
    return spike_dates


def get_headlines(query, start_date=None, end_date=None):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "tbm": "nws",  # news search
        "api_key": SERPAPI_KEY
    }
    if start_date and end_date:
        tbs = f"cdr:1,cd_min:{start_date.month}/{start_date.day}/{start_date.year},cd_max:{end_date.month}/{end_date.day}/{end_date.year}"
        params["tbs"] = tbs
    response = requests.get(url, params=params)
    data = response.json()
    headlines = []
    if "news_results" in data:
        for item in data["news_results"][:5]:
            headlines.append(item["title"])
    return headlines


def summarize(headlines, song, artist, date=None):
    api_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    text = "\n".join(headlines)
    date_str = f" around {date}" if date else ""
    prompt = f"""
These are recent news headlines about the song "{song}" by {artist}{date_str}:
{text}
Explain in 3 short bullet points why this song spiked in popularity{date_str}.
"""
    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": prompt}
    )
    result = response.json()
    # Hugging Face returns a list when successful
    if isinstance(result, list):
        return result[0].get("generated_text", "No summary available.")
    else:
        return "Model is loading, refresh once."


def groq_summarize(headlines, song, artist, date=None):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    text = "\n".join(headlines)
    date_str = f" around {date}" if date else ""

    prompt = f"""
These are recent news headlines about the song "{song}" by {artist}{date_str}:
{text}
Explain in 3 short bullet points why this song spiked in popularity{date_str}.
"""

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
    }

    response = requests.post(api_url, headers=headers, json=payload)
    result = response.json()
    print(response)
    print()
    print(result)

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return "No summary available."


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, song: str = Form(...), artist: str = Form(...)):
    # Verify song and artist first
    verification = verify_song_artist(song.strip(), artist.strip())
    
    if not verification["valid"]:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": verification.get("message", "Invalid song/artist combination.")
        })
    
    # Use matched/verified names if available
    verified_song = verification.get("matched_song", song)
    verified_artist = verification.get("matched_artist", artist)
    
    query = f"{verified_song} {verified_artist}"
    timeline_data = get_trends_data(query)
    dates = [item['date'] for item in timeline_data]
    values = [item['values'][0]['value'] for item in timeline_data]
    spike_dates = detect_spikes(timeline_data)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "song": verified_song,
        "artist": verified_artist,
        "dates": dates,
        "values": values,
        "spike_dates": spike_dates
    })


@app.get("/explain", response_class=HTMLResponse)
def explain(request: Request, song: str, artist: str, date: str):
    # Recompute trends for consistency
    query = f"{song} {artist}"
    timeline_data = get_trends_data(query)
    dates = [item['date'] for item in timeline_data]
    values = [item['values'][0]['value'] for item in timeline_data]
    spike_dates = detect_spikes(timeline_data)

    # Parse date range and extract start date for datetime
    try:
        parsed_date = parse_date_range(date)
        start_date = parsed_date["start"] - datetime.timedelta(days=3)
        end_date = parsed_date["end"] + datetime.timedelta(days=3)
        month = parsed_date["month"]
        year = parsed_date["year"]
    except ValueError:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "song": song,
            "artist": artist,
            "dates": dates,
            "values": values,
            "spike_dates": spike_dates,
            "error": "Invalid date format."
        })

    news_query = f"{song} {artist} music"
    headlines = get_headlines(news_query, start_date, end_date)
    explanation = groq_summarize(headlines, song, artist, date)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "song": song,
        "artist": artist,
        "dates": dates,
        "values": values,
        "spike_dates": spike_dates,
        "selected_date": date,  # Keep the full range for display
        "explanation": explanation,
        "headlines": headlines  # Optional: Keep if you want to show raw headlines too
    })


def parse_date_range(date: str):
    parts = date.split(',')
    if len(parts) != 2:
        raise ValueError("Invalid date format: missing comma separator.")
    left = parts[0].replace('\u2009', '').strip()  # Remove thin space, e.g., "Aug 24–30"
    year = parts[1].strip()  # "2025"
    month_day_range = left.split(" ", 1)
    if len(month_day_range) != 2:
        raise ValueError("Invalid date format: missing space after month.")
    month = month_day_range[0]
    day_range = month_day_range[1]
    day_parts = day_range.replace('–', '-').split('-')  # Handle en-dash
    if len(day_parts) != 2:
        raise ValueError("Invalid date format: missing day range separator.")
    start_day = day_parts[0]
    end_day = day_parts[1]
    start_str = f"{month} {start_day}, {year}"
    end_str = f"{month} {end_day}, {year}"
    start = datetime.datetime.strptime(start_str, "%b %d, %Y")
    end = datetime.datetime.strptime(end_str, "%b %d, %Y")
    return {
        "start": start,
        "end": end,
        "year": int(year),
        "month": month
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)