# Hype Search 🎵📈

### AI-Powered Song Trend Analyzer

**Hype Search** is an AI‑powered tool that explains why songs suddenly spike in popularity. It pulls Google Trends data through SerpAPI, detects statistically significant “hype spikes,” and then uses Groq’s Llama 3 to summarize news from that exact time period. The result is a clear, human‑readable explanation that connects data patterns to real‑world events, all delivered through a fast FastAPI backend and a clean, theme‑toggle UI.

---

## 🚀 Features

* **Search any song + artist:** Automatic verification using SerpAPI.
* **Interactive Google Trends graph:** Visual representation of interest over the last 12 months.
* **Spike detection:** Automated peak identification using signal processing.
* **News headline aggregation:** Contextual headlines pulled specifically around spike dates.
* **AI-generated explanations:** LLM analysis provides 3 bullet-point reasons for popularity surges.
* **Modern UI:** Built with TailwindCSS and Chart.js.
* **Theme switcher:** Support for a Default theme and a custom **UT Burnt Orange** theme.

---

## 🛠 Tech Stack

### **Backend** * **FastAPI** & **Python** * **SerpAPI** (Google Trends + News)

* **NumPy** & **SciPy** (Peak detection)
* **Groq API** (LLM explanations)
* **Jinja2 Templates** ### **Frontend** * **HTML / TailwindCSS** * **Chart.js** (Data visualization)
* **Lucide Icons** * **JavaScript** (Fetch API & interactivity)

---

## ⚙️ How It Works

1. **Input:** User enters a Song and Artist.
2. **Verification:** The app verifies the combination via SerpAPI search.
3. **Data Fetching:** Google Trends data is retrieved for the past 12 months.
4. **Analysis:** `SciPy.find_peaks` detects significant interest spikes.
5. **Contextualization:** When a user clicks a spike, news headlines are pulled for that specific date range.
6. **Explanation:** AI analyzes the headlines to generate the "Why" behind the trend.

---

## 📂 Project Structure

```text
PopularSongAnalysis/
│
├── main.py                # FastAPI backend logic
├── templates/
│   └── index.html         # Frontend UI
├── static/                # Static files (CSS/JS/images)
├── .env                   # API keys (not committed)
└── README.md              # Project documentation

```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add your keys:

```env
SERPAPI_KEY=your_serpapi_key
GROQ_API_KEY=your_groq_key
LASTFM_API_KEY=optional_key

```

---

## 🛠 Installation & Run

### 1. Clone Repository

```bash
git clone [https://github.com/your-username/hype-graph.git](https://github.com/hk10877/PopularSongAnalysis.git)
cd PopularSongAnalysis

```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate  
# Mac/Linux
source venv/bin/activate  

```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn requests python-dotenv serpapi numpy scipy groq jinja2

```

### 4. Run Server

```bash
uvicorn main:app --reload

```

Open your browser at: `http://127.0.0.1:8000`

---

## 📡 API Endpoints

| Route | Method | Description |
| --- | --- | --- |
| `/` | `GET` | Home page |
| `/analyze` | `POST` | Analyze song trends |
| `/explain` | `GET` | AI explanation for spike date |

---

## 🔮 Future Improvements

* **Spotify API integration** for direct playback and playlist data.
* **Comparison Mode:** Compare trends between multiple songs.
* **Sentiment Analysis:** Social media integration to gauge public mood.
* **User Accounts:** Save search history and exportable PDF reports.

---

## 👥 Contributors

* **Harshita Kumari** 
* **Zara Ike** 
* **Benita Benjamin** 

## 📄 License

This project is licensed under the **MIT License**.

---

**Hype Search** helps answer the question: *“Why did this song suddenly blow up?”* 🎶📊
