# Lotto Dashboard – All-in-One (Profiler + Fetch + Verify)

This Streamlit app combines:
- Frequency, Quick Picks, Simulation, Data Profiler (rolling window & gap analysis)
- **Data Manager** (append/dedup into `data/*_master.csv`)
- **Fetch & Verify** tab to fetch last ~12 months for:
  - Saturday Lotto (TattsLotto)
  - Monday & Wednesday (via Weekday Windfall archives; filtered by weekday)
- Built-in **verifier** and optional **diff vs official CSV**

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Internet is required for the Fetch tab on your machine (the app uses `requests` + `BeautifulSoup`).
- CSV schema: DrawDate,N1..N6,S1,S2 (DrawDate in YYYY-MM-DD)
- Dedup key: DrawDate (latest wins)
