from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI
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
from urllib.parse import quote

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")  # Not used in this example, but kept for potential expansion
app = FastAPI()
templates = Jinja2Templates(directory="templates")


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


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, song: str = Form(...), artist: str = Form(...)):
    query = f"{song} {artist}"
    timeline_data = get_trends_data(query)
    dates = [item['date'] for item in timeline_data]
    values = [item['values'][0]['value'] for item in timeline_data]
    spike_dates = detect_spikes(timeline_data)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "song": song,
        "artist": artist,
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

    # Parse date and set range +/- 3 days
    try:
        selected_date = datetime.datetime.strptime(date, "%b %d, %Y")  # e.g., "Jan 1, 2023"
        start_date = selected_date - datetime.timedelta(days=3)
        end_date = selected_date + datetime.timedelta(days=3)
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

    # Simple explanation without AI: Just list headlines as potential reasons
    explanation = "Possible reasons for the spike based on news headlines:\n" + "\n".join(
        [f"- {h}" for h in headlines]) if headlines else "No relevant headlines found for this period."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "song": song,
        "artist": artist,
        "dates": dates,
        "values": values,
        "spike_dates": spike_dates,
        "selected_date": date,
        "explanation": explanation,
        "headlines": headlines
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)