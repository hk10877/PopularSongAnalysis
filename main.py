from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
templates = Jinja2Templates(directory="templates")


def get_headlines(query):
    url = "https://serpapi.com/search"
    params = {
        "q": query,
        "tbm": "nws",  # news search
        "api_key": SERPAPI_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    headlines = []
    if "news_results" in data:
        for item in data["news_results"][:5]:
            headlines.append(item["title"])

    return headlines


def summarize(headlines, song, artist):
    text = "\n".join(headlines)

    prompt = f"""
These are recent news headlines about the song "{song}" by {artist}:

{text}

Explain in 3 short bullet points why this song is currently popular.
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, song: str = Form(...), artist: str = Form(...)):
    query = f"{song} {artist} trending"
    headlines = get_headlines(query)
    summary = summarize(headlines, song, artist)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "summary": summary,
        "headlines": headlines
    })
