
import streamlit as st
import pandas as pd
import numpy as np
import io, re, sys, os, requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict

st.set_page_config(page_title="Lotto Dashboard – All-in-One", layout="wide")
st.title("🎲 Lotto Dashboard – Profiler, Fetch & Verify (All-in-One)")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

GAME_FILES = {
    "Saturday": os.path.join(DATA_DIR, "saturday_master.csv"),
    "Monday":   os.path.join(DATA_DIR, "monday_master.csv"),
    "Wednesday":os.path.join(DATA_DIR, "wednesday_master.csv"),
}

def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["DrawDate","N1","N2","N3","N4","N5","N6","S1","S2"]
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols].copy()
    df["DrawDate"] = pd.to_datetime(df["DrawDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    for c in ["N1","N2","N3","N4","N5","N6","S1","S2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_master(game: str) -> pd.DataFrame:
    path = GAME_FILES[game]
    if os.path.exists(path):
        try:
            return normalize_schema(pd.read_csv(path))
        except Exception:
            return pd.DataFrame(columns=["DrawDate","N1","N2","N3","N4","N5","N6","S1","S2"])
    return pd.DataFrame(columns=["DrawDate","N1","N2","N3","N4","N5","N6","S1","S2"])

def save_master(game: str, df: pd.DataFrame):
    path = GAME_FILES[game]
    normalize_schema(df).to_csv(path, index=False)

def merge_into_master(game: str, incoming: pd.DataFrame) -> pd.DataFrame:
    master = load_master(game)
    incoming = normalize_schema(incoming)
    merged = pd.concat([master, incoming], ignore_index=True)
    merged = merged.dropna(subset=["DrawDate"]).sort_values("DrawDate")
    merged = merged.drop_duplicates(subset=["DrawDate"], keep="last")
    return merged

def verify_df(df: pd.DataFrame, k_main=6, k_supp=2, minn=1, maxn=45):
    problems = []
    try:
        dd = pd.to_datetime(df["DrawDate"], errors="raise")
    except Exception as e:
        problems.append(f"DrawDate parse error: {e}")
        return problems
    if dd.duplicated().any():
        dups = dd[dd.duplicated()].dt.strftime("%Y-%m-%d").tolist()
        problems.append(f"Duplicate DrawDate rows found: {dups}")
    mains_cols = [f"N{i}" for i in range(1, k_main+1)]
    supp_cols  = [f"S{i}" for i in range(1, k_supp+1)]
    for idx, row in df.iterrows():
        try:
            mains = [int(row[c]) for c in mains_cols]
        except Exception:
            problems.append(f"Row {idx}: Non-integer main value(s)")
            continue
        if any((m < minn or m > maxn) for m in mains):
            problems.append(f"Row {idx}: Main out of range {minn}-{maxn}: {mains}")
        if len(set(mains)) != len(mains):
            problems.append(f"Row {idx}: Duplicate mains in row: {mains}")
        supps = []
        for c in supp_cols:
            v = row.get(c, pd.NA)
            if pd.notna(v):
                try:
                    v = int(v)
                except Exception:
                    problems.append(f"Row {idx}: Non-integer supplementary in {c}: {v}")
                    continue
                if v < minn or v > maxn:
                    problems.append(f"Row {idx}: Supplementary out of range {minn}-{maxn} in {c}: {v}")
                supps.append(v)
        if any(s in mains for s in supps):
            problems.append(f"Row {idx}: Supplementary overlaps with mains: mains={mains}, supps={supps}")
        if len(set(supps)) != len(supps):
            problems.append(f"Row {idx}: Duplicate supplementaries in row: {supps}")
    return problems

def cadence_check(df: pd.DataFrame, expected_weekday: int):
    dd = pd.to_datetime(df["DrawDate"], errors="coerce").sort_values().tolist()
    missing = []
    for i in range(1, len(dd)):
        delta = (dd[i] - dd[i-1]).days
        if delta not in (7, 14):
            missing.append((dd[i-1].strftime("%Y-%m-%d"), dd[i].strftime("%Y-%m-%d"), delta))
    weekday_bad = pd.to_datetime(df["DrawDate"], errors="coerce").dt.weekday
    bad_days = df[weekday_bad != expected_weekday]["DrawDate"].tolist()
    return missing, bad_days

def diff_against_official(user_df: pd.DataFrame, official_df: pd.DataFrame, k_main=6, k_supp=2):
    u = normalize_schema(user_df).copy()
    o = normalize_schema(official_df).copy()
    u["DrawDate"] = pd.to_datetime(u["DrawDate"], errors="coerce")
    o["DrawDate"] = pd.to_datetime(o["DrawDate"], errors="coerce")
    merged = u.merge(o, on="DrawDate", how="outer", suffixes=("_user", "_off"))
    diffs = []
    for _, r in merged.iterrows():
        d = r["DrawDate"]
        user_missing = any(pd.isna(r.get(f"N{i}_user")) for i in range(1, k_main+1))
        off_missing  = any(pd.isna(r.get(f"N{i}_off"))  for i in range(1, k_main+1))
        if user_missing or off_missing:
            diffs.append({"DrawDate": d, "type": "missing_in_one_file"})
            continue
        user_mains = [int(r.get(f"N{i}_user")) for i in range(1, k_main+1)]
        off_mains  = [int(r.get(f"N{i}_off"))  for i in range(1, k_main+1)]
        user_supps = [int(r.get(f"S{i}_user")) for i in range(1, k_supp+1) if pd.notna(r.get(f"S{i}_user"))]
        off_supps  = [int(r.get(f"S{i}_off"))  for i in range(1, k_supp+1) if pd.notna(r.get(f"S{i}_off"))]
        if sorted(user_mains) != sorted(off_mains) or sorted(user_supps) != sorted(off_supps):
            diffs.append({"DrawDate": d, "type": "numbers_mismatch",
                          "user_mains": user_mains, "off_mains": off_mains,
                          "user_supps": user_supps, "off_supps": off_supps})
    return pd.DataFrame(diffs)

HEADERS = {"User-Agent": "Mozilla/5.0"}

def _parse_archive(url: str, date_prefix_regex: str):
    html = requests.get(url, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for block in soup.select("div.results, li.results, div.result, li.result, table tr"):
        text = " ".join(block.get_text(" ", strip=True).split())
        m = re.search(date_prefix_regex + r"\s+(\d{1,2}\s+\w+\s+\d{4})", text, re.IGNORECASE)
        if not m:
            m2 = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", text)
            if not m2:
                continue
            date_str = m2.group(1)
        else:
            date_str = m.group(2)
        dt = None
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                break
            except Exception:
                pass
        if dt is None:
            continue
        nums = [int(x) for x in re.findall(r"\b([1-9]|[1-3]\d|4[0-5])\b", text)]
        if len(nums) < 6:
            balls = [b.get_text(strip=True) for b in block.select(".ball, .ballSmall")]
            nums = [int(x) for x in balls if x.isdigit() and 1 <= int(x) <= 45]
        if len(nums) < 6:
            continue
        mains = nums[:6]
        supps = nums[6:8] if len(nums) >= 8 else [None, None]
        rows.append({"DrawDate": dt.date().isoformat(),
                     "N1": mains[0], "N2": mains[1], "N3": mains[2],
                     "N4": mains[3], "N5": mains[4], "N6": mains[5],
                     "S1": supps[0], "S2": supps[1]})
    return rows

def fetch_saturday_last12m():
    urls = [
        "https://australia.national-lottery.com/saturday-lotto/results-archive-2025",
        "https://australia.national-lottery.com/saturday-lotto/results-archive-2024",
    ]
    all_rows = []
    for u in urls:
        try:
            all_rows.extend(_parse_archive(u, r"(Saturday|Sat)"))
        except Exception as e:
            st.warning(f"Failed to parse {u}: {e}")
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["DrawDate"])
    df = df[pd.to_datetime(df["DrawDate"]) >= (pd.Timestamp.today() - pd.Timedelta(days=366))]
    df = df.sort_values("DrawDate")
    return df

def fetch_weekday_windfall_last12m(filter_weekday: int):
    urls = [
        "https://australia.national-lottery.com/weekday-windfall/results-archive-2025",
        "https://australia.national-lottery.com/weekday-windfall/results-archive-2024",
    ]
    all_rows = []
    for u in urls:
        try:
            all_rows.extend(_parse_archive(u, r"(Monday|Tuesday|Wednesday|Thursday|Friday|Mon|Tue|Wed|Thu|Fri)"))
        except Exception as e:
            st.warning(f"Failed to parse {u}: {e}")
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["DrawDate"])
    df["DrawDate_dt"] = pd.to_datetime(df["DrawDate"], errors="coerce")
    df["Weekday"] = df["DrawDate_dt"].dt.weekday
    df = df[df["Weekday"] == filter_weekday]
    df = df.drop(columns=["DrawDate_dt","Weekday"], errors="ignore")
    df = df[pd.to_datetime(df["DrawDate"]) >= (pd.Timestamp.today() - pd.Timedelta(days=366))]
    df = df.sort_values("DrawDate")
    return df

st.sidebar.header("Configuration")
number_min = st.sidebar.number_input("Smallest ball number", 1, 1_000, 1)
number_max = st.sidebar.number_input("Largest ball number", number_min+1, 1_000, 45)
balls_per_draw = st.sidebar.number_input("Main numbers per draw", 3, 15, 6)
supplementaries = st.sidebar.number_input("Supplementary balls per draw (optional)", 0, 5, 2)

st.sidebar.markdown("---")
st.sidebar.header("Active dataset")
use_master = st.sidebar.checkbox("Use saved master dataset", value=True)
game_choice = st.sidebar.selectbox("Game (for master dataset)", ["Saturday","Monday","Wednesday"], index=0)

uploaded = st.sidebar.file_uploader("Upload ad-hoc CSV (optional)", type=["csv"], help="Used only if not using master dataset.")
sample_btn = st.sidebar.button("Load sample dataset")

def load_sample_df(n_draws=250, n_min=1, n_max=45, k_main=6, k_supp=2, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_draws):
        main = rng.choice(np.arange(n_min, n_max+1), size=k_main, replace=False)
        supp = rng.choice(np.setdiff1d(np.arange(n_min, n_max+1), main), size=k_supp, replace=False) if k_supp>0 else []
        rows.append({
            "DrawDate": (datetime(2024,1,1) + pd.Timedelta(days=i*3)).date().isoformat(),
            **{f"N{j+1}": int(main[j]) for j in range(k_main)},
            **({f"S{j+1}": int(supp[j]) for j in range(k_supp)} if k_supp>0 else {}),
        })
    return pd.DataFrame(rows)

if use_master:
    draws_df = load_master(game_choice)
else:
    if uploaded is not None:
        draws_df = normalize_schema(pd.read_csv(uploaded))
    elif sample_btn:
        draws_df = load_sample_df(n_draws=250, n_min=number_min, n_max=number_max, k_main=balls_per_draw, k_supp=supplementaries)
    else:
        draws_df = pd.DataFrame()

tab_data, tab_fetch, tab_freq, tab_picks, tab_sim, tab_profiler = st.tabs([
    "Data Manager", "Fetch & Verify", "Frequency", "Quick Picks", "Simulation", "Data Profiler"
])

with tab_data:
    st.header("🗂️ Data Manager (Masters & Append)")
    st.write("Upload CSVs to **append & deduplicate** into master datasets.")

    dm_game = st.selectbox("Select game to update", ["Saturday","Monday","Wednesday"], key="dm_game")
    dm_file = st.file_uploader("Upload CSV to append", type=["csv"], key="dm_upload")
    if st.button("Append to Master"):
        if dm_file is None:
            st.error("Please upload a CSV first.")
        else:
            try:
                inc = normalize_schema(pd.read_csv(dm_file))
                merged = merge_into_master(dm_game, inc)
                for c in ["N1","N2","N3","N4","N5","N6","S1","S2"]:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").clip(lower=number_min, upper=number_max)
                save_master(dm_game, merged)
                st.success(f"Updated **{dm_game}** master: {len(merged)} rows total.")
                st.dataframe(merged.tail(10), use_container_width=True)
                buf = io.StringIO(); merged.to_csv(buf, index=False)
                st.download_button("⬇️ Download updated master", data=buf.getvalue(), file_name=f"{dm_game.lower()}_master.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Failed to append: {e}")

    st.subheader("Existing masters")
    for g in ["Saturday","Monday","Wednesday"]:
        path = GAME_FILES[g]
        if os.path.exists(path):
            dfm = load_master(g)
            st.write(f"**{g}** — {len(dfm)} rows")
            st.dataframe(dfm.tail(5), use_container_width=True)
        else:
            st.write(f"**{g}** — *(no file yet)*")

with tab_fetch:
    st.header("🌐 Fetch & Verify (Built-in)")
    st.caption("Fetch last ~12 months from the National Lottery archives, verify, and append into master files.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Saturday")
        if st.button("Fetch Saturday last 12 months"):
            sat = fetch_saturday_last12m()
            st.session_state["sat_fetched"] = sat
            st.success(f"Fetched {len(sat)} Saturday draws.")
            st.dataframe(sat.tail(10), use_container_width=True)
        if "sat_fetched" in st.session_state:
            sat = st.session_state["sat_fetched"]
            probs = verify_df(sat, k_main=6, k_supp=2, minn=number_min, maxn=number_max)
            if probs:
                st.warning(f"Verifier found {len(probs)} issues. First 10:")
                for p in probs[:10]:
                    st.text("- " + p)
            else:
                st.success("Verifier: OK")
            if st.button("Append Saturday to master"):
                merged = merge_into_master("Saturday", sat)
                for c in ["N1","N2","N3","N4","N5","N6","S1","S2"]:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").clip(lower=number_min, upper=number_max)
                save_master("Saturday", merged)
                st.success(f"Saturday master updated: {len(merged)} rows.")

    with col2:
        st.subheader("Monday")
        if st.button("Fetch Monday last 12 months"):
            mon = fetch_weekday_windfall_last12m(filter_weekday=0)
            st.session_state["mon_fetched"] = mon
            st.success(f"Fetched {len(mon)} Monday draws.")
            st.dataframe(mon.tail(10), use_container_width=True)
        if "mon_fetched" in st.session_state:
            mon = st.session_state["mon_fetched"]
            probs = verify_df(mon, k_main=6, k_supp=2, minn=number_min, maxn=number_max)
            if probs:
                st.warning(f"Verifier found {len(probs)} issues. First 10:")
                for p in probs[:10]:
                    st.text("- " + p)
            else:
                st.success("Verifier: OK")
            if st.button("Append Monday to master"):
                merged = merge_into_master("Monday", mon)
                for c in ["N1","N2","N3","N4","N5","N6","S1","S2"]:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").clip(lower=number_min, upper=number_max)
                save_master("Monday", merged)
                st.success(f"Monday master updated: {len(merged)} rows.")

    with col3:
        st.subheader("Wednesday")
        if st.button("Fetch Wednesday last 12 months"):
            wed = fetch_weekday_windfall_last12m(filter_weekday=2)
            st.session_state["wed_fetched"] = wed
            st.success(f"Fetched {len(wed)} Wednesday draws.")
            st.dataframe(wed.tail(10), use_container_width=True)
        if "wed_fetched" in st.session_state:
            wed = st.session_state["wed_fetched"]
            probs = verify_df(wed, k_main=6, k_supp=2, minn=number_min, maxn=number_max)
            if probs:
                st.warning(f"Verifier found {len(probs)} issues. First 10:")
                for p in probs[:10]:
                    st.text("- " + p)
            else:
                st.success("Verifier: OK")
            if st.button("Append Wednesday to master"):
                merged = merge_into_master("Wednesday", wed)
                for c in ["N1","N2","N3","N4","N5","N6","S1","S2"]:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").clip(lower=number_min, upper=number_max)
                save_master("Wednesday", merged)
                st.success(f"Wednesday master updated: {len(merged)} rows.")

    st.markdown("---")
    st.subheader("Compare vs Official (optional)")
    st.caption("Upload an official CSV to diff against your fetched/merged data.")
    target_game = st.selectbox("Game to compare", ["Saturday","Monday","Wednesday"], key="cmp_game")
    official_csv = st.file_uploader("Upload official CSV", type=["csv"], key="cmp_upload")
    if st.button("Run diff"):
        if official_csv is None:
            st.error("Upload an official CSV first.")
        else:
            off = normalize_schema(pd.read_csv(official_csv))
            user = load_master(target_game)
            if user.empty:
                st.error("Your master is empty. Append fetched data first.")
            else:
                diffs = diff_against_official(user, off, k_main=6, k_supp=2)
                if diffs.empty:
                    st.success("No differences found.")
                else:
                    st.warning(f"Found {len(diffs)} differences. Showing table below.")
                    st.dataframe(diffs, use_container_width=True)

with tab_freq:
    st.header("📊 Frequency Analysis")
    if draws_df.empty:
        st.info("Select **Use saved master dataset** or upload a CSV / load sample in the sidebar.")
    else:
        st.dataframe(draws_df.head(20), use_container_width=True)

        def compute_frequencies(df):
            counts = Counter()
            for _, row in df.iterrows():
                for i in range(balls_per_draw):
                    counts[int(row[f'N{i+1}'])] += 1
            freq_df = pd.DataFrame({"Number": list(range(number_min, number_max+1))})
            freq_df["Count"] = freq_df["Number"].map(lambda x: counts.get(x, 0))
            return freq_df

        freq_df = compute_frequencies(draws_df)

        fig, ax = plt.subplots()
        ax.bar(freq_df["Number"].astype(str), freq_df["Count"])
        ax.set_title("Frequency of Main Numbers")
        ax.set_xlabel("Number")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        hot_n = st.number_input("How many HOT numbers to consider?", 1, number_max-number_min+1, min(10, number_max-number_min+1))
        cold_n = st.number_input("How many COLD numbers to consider?", 1, number_max-number_min+1, min(10, number_max-number_min+1))
        hot_list = freq_df.sort_values("Count", ascending=False).head(hot_n)["Number"].tolist()
        cold_list = freq_df.sort_values("Count", ascending=True).head(cold_n)["Number"].tolist()
        st.write("🔥 **Hot**:", hot_list)
        st.write("❄️ **Cold**:", cold_list)

with tab_picks:
    st.header("🎟️ Quick Picks")
    strategy = st.selectbox("Strategy", ["Hot", "Cold", "Balanced", "Random"])
    n_picks = st.number_input("How many games to generate?", 1, 200, 6)
    rng = np.random.default_rng(42)

    def pick_from_pool(pool, k):
        pool = sorted(list(set([x for x in pool if number_min <= x <= number_max])))
        if len(pool) < k:
            extra = [x for x in range(number_min, number_max+1) if x not in pool]
            rng.shuffle(extra)
            pool = pool + extra[:max(0,k-len(pool))]
        return sorted(rng.choice(pool, size=k, replace=False).tolist())

    picks = []
    if not draws_df.empty:
        tmp_freq = Counter()
        for _, row in draws_df.iterrows():
            for i in range(balls_per_draw):
                tmp_freq[int(row[f'N{i+1}'])] += 1
        freq_df2 = pd.DataFrame({"Number": list(range(number_min, number_max+1))})
        freq_df2["Count"] = freq_df2["Number"].map(lambda x: tmp_freq.get(x, 0))
        hot_list = freq_df2.sort_values("Count", ascending=False).head(10)["Number"].tolist()
        cold_list = freq_df2.sort_values("Count", ascending=True).head(10)["Number"].tolist()

        if strategy == "Hot":
            pool = hot_list
        elif strategy == "Cold":
            pool = cold_list
        elif strategy == "Balanced":
            half = balls_per_draw // 2
            pool = hot_list[:max(half,1)] + cold_list[:max(balls_per_draw-half,1)]
        else:
            pool = list(range(number_min, number_max+1))

        for _ in range(n_picks):
            picks.append(pick_from_pool(pool, balls_per_draw))
    else:
        for _ in range(n_picks):
            picks.append(sorted(rng.choice(np.arange(number_min, number_max+1), size=balls_per_draw, replace=False).tolist()))

    picks_df = pd.DataFrame({"Pick #": range(1, len(picks)+1), "Numbers": picks})
    st.dataframe(picks_df, use_container_width=True)
    csv_buf = io.StringIO(); picks_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download picks as CSV", data=csv_buf.getvalue(), file_name="lotto_picks.csv", mime="text/csv")

with tab_profiler:
    st.header("📑 Data Profiler")
    if draws_df.empty:
        st.info("Load data to profile using the sidebar.")
    else:
        st.subheader("Rolling Window")
        window = st.number_input("Window size (most recent draws)", 10, max(10, len(draws_df)), min(50, len(draws_df)) if len(draws_df) else 50)
        dfw = draws_df.tail(window) if len(draws_df) >= window else draws_df.copy()

        c1, c2 = st.columns(2)
        c1.metric("Total draws (window)", f"{len(dfw)}")
        c2.metric("Number range", f"{number_min}–{number_max}")

        freq_counts = Counter()
        for _, row in dfw.iterrows():
            for i in range(balls_per_draw):
                freq_counts[int(row[f'N{i+1}'])] += 1
        freq_df = pd.DataFrame({"Number": list(range(number_min, number_max+1))})
        freq_df["Count"] = freq_df["Number"].map(lambda x: freq_counts.get(x, 0))

        st.subheader("Top/Bottom in Window")
        topN = st.number_input("Show top/bottom N", 5, 50, 15)
        topN_df = freq_df.sort_values("Count", ascending=False).head(topN)
        bottomN_df = freq_df.sort_values("Count", ascending=True).head(topN)

        figA, axA = plt.subplots(); axA.bar(topN_df["Number"].astype(str), topN_df["Count"])
        axA.set_title(f"Top {topN} Numbers (Window)"); axA.set_xlabel("Number"); axA.set_ylabel("Count"); st.pyplot(figA)

        figB, axB = plt.subplots(); axB.bar(bottomN_df["Number"].astype(str), bottomN_df["Count"])
        axB.set_title(f"Bottom {topN} Numbers (Window)"); axB.set_xlabel("Number"); axB.set_ylabel("Count"); st.pyplot(figB)

        st.subheader("Position heatmap (Window)")
        pos_mat = np.zeros((balls_per_draw, number_max - number_min + 1), dtype=int)
        for _, row in dfw.iterrows():
            for i in range(balls_per_draw):
                num = int(row[f"N{i+1}"])
                if number_min <= num <= number_max:
                    pos_mat[i, num - number_min] += 1
        figC, axC = plt.subplots(); im = axC.imshow(pos_mat, aspect="auto", origin="lower")
        axC.set_title("Frequency by position (window)"); axC.set_xlabel("Number"); axC.set_ylabel("N position (0=top)")
        st.pyplot(figC)

        st.subheader("Gap Analysis (full dataset)")
        appearances = defaultdict(list)
        for idx, row in draws_df.reset_index(drop=True).iterrows():
            for i in range(balls_per_draw):
                n = int(row[f"N{i+1}"])
                appearances[n].append(idx)

        records = []
        total_draws = len(draws_df)
        for n in range(number_min, number_max+1):
            idxs = appearances.get(n, [])
            if not idxs:
                records.append({"Number": n, "Hits": 0, "MaxGap": None, "CurrentGap": total_draws, "AvgGap": None})
                continue
            gaps = [j - i for i, j in zip(idxs[:-1], idxs[1:])] if len(idxs) > 1 else []
            max_gap = max(gaps) if gaps else (total_draws - idxs[-1] - 1)
            current_gap = total_draws - idxs[-1] - 1
            avg_gap = (sum(gaps) / len(gaps)) if gaps else None
            records.append({"Number": n, "Hits": len(idxs), "MaxGap": max_gap, "CurrentGap": current_gap, "AvgGap": avg_gap})

        gap_df = pd.DataFrame(records)
        st.dataframe(gap_df.sort_values(["MaxGap","CurrentGap"], ascending=[False, False]).reset_index(drop=True), use_container_width=True)

        K = st.number_input("Show top K numbers by MaxGap", 5, 50, 15)
        top_gap = gap_df.sort_values("MaxGap", ascending=False).head(K)
        figG, axG = plt.subplots(); axG.bar(top_gap["Number"].astype(str), top_gap["MaxGap"].fillna(0))
        axG.set_title(f"Top {K} Numbers by Max Gap"); axG.set_xlabel("Number"); axG.set_ylabel("Max Gap (draws)")
        st.pyplot(figG)

st.caption("Tip: Use the **Fetch & Verify** tab to pull the latest results, verify, and append directly to masters, then switch to **Use saved master dataset** in the sidebar for analysis.")
