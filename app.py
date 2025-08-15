
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from datetime import datetime

st.set_page_config(page_title="Lotto Calculation Dashboard", layout="wide")
st.title("🎲 Lotto Calculation Dashboard")

st.sidebar.header("Configuration")
number_min = st.sidebar.number_input("Smallest ball number", 1, 1_000, 1)
number_max = st.sidebar.number_input("Largest ball number", number_min+1, 1_000, 45)
balls_per_draw = st.sidebar.number_input("Main numbers per draw", 3, 15, 6)
supplementaries = st.sidebar.number_input("Supplementary balls per draw (optional)", 0, 5, 2)

st.sidebar.markdown("---")
st.sidebar.header("Data input")
uploaded = st.sidebar.file_uploader("Upload historical draws (CSV)", type=["csv"])
sample_btn = st.sidebar.button("Load sample dataset")

def load_sample_df(n_draws=300, n_min=1, n_max=45, k_main=6, k_supp=2, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_draws):
        main = rng.choice(np.arange(n_min, n_max+1), size=k_main, replace=False)
        supp = rng.choice(np.setdiff1d(np.arange(n_min, n_max+1), main), size=k_supp, replace=False) if k_supp>0 else []
        rows.append({
            "DrawDate": datetime(2024,1,1) + pd.Timedelta(days=i*3),
            **{f"N{j+1}": int(main[j]) for j in range(k_main)},
            **({f"S{j+1}": int(supp[j]) for j in range(k_supp)} if k_supp>0 else {}),
        })
    return pd.DataFrame(rows)

def parse_draws(df, k_main, k_supp):
    cols = list(df.columns)
    main_cols = [c for c in cols if str(c).upper().startswith("N")]
    supp_cols = [c for c in cols if str(c).upper().startswith("S")]
    def is_int_col(series):
        try:
            pd.to_numeric(series, errors="coerce")
            return True
        except Exception:
            return False
    if len(main_cols) < k_main:
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) or is_int_col(df[c])]
        main_cols = num_cols[:k_main]
        supp_cols = num_cols[k_main:k_main+k_supp] if k_supp>0 else []
    def coerce_int(series):
        s = pd.to_numeric(series, errors="coerce").astype("Int64")
        return s
    out = pd.DataFrame()
    for i, c in enumerate(main_cols[:k_main]):
        out[f"N{i+1}"] = coerce_int(df[c])
    for i, c in enumerate(supp_cols[:k_supp]):
        out[f"S{i+1}"] = coerce_int(df[c])
    if "DrawDate" in df.columns:
        out["DrawDate"] = pd.to_datetime(df["DrawDate"], errors="coerce")
    out = out.dropna(subset=[f"N{i+1}" for i in range(k_main)])
    for c in [c for c in out.columns if c.startswith("N") or c.startswith("S")]:
        out[c] = out[c].astype(int).clip(lower=number_min, upper=number_max)
    return out

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
    draws_df = parse_draws(raw_df, balls_per_draw, supplementaries)
elif sample_btn:
    raw_df = load_sample_df(n_draws=300, n_min=number_min, n_max=number_max, k_main=balls_per_draw, k_supp=supplementaries)
    draws_df = parse_draws(raw_df, balls_per_draw, supplementaries)
else:
    draws_df = pd.DataFrame()

tab_freq, tab_picks, tab_sim, tab_profiler = st.tabs(["Frequency", "Quick Picks", "Simulation", "Data Profiler"])

with tab_freq:
    st.header("📊 Frequency Analysis")
    st.markdown("#### Loaded Draws")
    if draws_df.empty:
        st.info("Upload a CSV of past draws or click **Load sample dataset** in the sidebar to proceed.")
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

    if not draws_df.empty:
        freq_df = compute_frequencies(draws_df)
        st.dataframe(freq_df, use_container_width=True)

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
        try:
            _ = hot_list; _ = cold_list
        except NameError:
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
    csv_buf = io.StringIO()
    picks_df.to_csv(csv_buf, index=False)
    st.download_button("⬇️ Download picks as CSV", data=csv_buf.getvalue(), file_name="lotto_picks.csv", mime="text/csv")

with tab_sim:
    st.header("🧪 Simulation (strategy vs random)")
    n_draws_sim = st.number_input("Number of simulated future draws", 100, 1_000_000, 10000, step=100)
    compare_btn = st.button("Run Simulation")

    rng = np.random.default_rng(123)

    def simulate_draw(n_min, n_max, k_main, k_supp):
        main = rng.choice(np.arange(n_min, n_max+1), size=k_main, replace=False).tolist()
        supp = []
        if k_supp > 0:
            remaining = [x for x in range(n_min, n_max+1) if x not in main]
            supp = rng.choice(np.array(remaining), size=k_supp, replace=False).tolist()
        return main, supp

    def count_matches(ticket, main, supp):
        m_main = len(set(ticket) & set(main))
        m_supp = len(set(ticket) & set(supp)) if len(supp) else 0
        return m_main, m_supp

    def pick_ticket(strategy):
        if draws_df.empty or strategy == "Random":
            return sorted(rng.choice(np.arange(number_min, number_max+1), size=balls_per_draw, replace=False).tolist())
        tmp_freq = Counter()
        for _, row in draws_df.iterrows():
            for i in range(balls_per_draw):
                tmp_freq[int(row[f'N{i+1}'])] += 1
        freq_df2 = pd.DataFrame({"Number": list(range(number_min, number_max+1))})
        freq_df2["Count"] = freq_df2["Number"].map(lambda x: tmp_freq.get(x, 0))
        hot_list2 = freq_df2.sort_values("Count", ascending=False).head(10)["Number"].tolist()
        cold_list2 = freq_df2.sort_values("Count", ascending=True).head(10)["Number"].tolist()

        if strategy == "Hot":
            pool = hot_list2
        elif strategy == "Cold":
            pool = cold_list2
        else:  # Balanced
            half = balls_per_draw // 2
            pool = hot_list2[:max(half,1)] + cold_list2[:max(balls_per_draw-half,1)]
        pool = sorted(list(set(pool)))
        if len(pool) < balls_per_draw:
            extra = [x for x in range(number_min, number_max+1) if x not in pool]
            rng.shuffle(extra)
            pool += extra[:balls_per_draw-len(pool)]
        return sorted(rng.choice(pool, size=balls_per_draw, replace=False).tolist())

    if compare_btn:
        strategies = ["Random", "Hot", "Cold", "Balanced"]
        tickets = {s: pick_ticket(s) for s in strategies}

        results = {s: {} for s in strategies}
        for _ in range(n_draws_sim):
            main, supp = simulate_draw(number_min, number_max, balls_per_draw, supplementaries)
            for s in strategies:
                m_main, m_supp = count_matches(tickets[s], main, supp)
                results[s][(m_main, m_supp)] = results[s].get((m_main, m_supp), 0) + 1

        rows = []
        for s in strategies:
            total = sum(results[s].values()) or 1
            for m in range(0, balls_per_draw+1):
                count_m = sum(v for (mm, ms), v in results[s].items() if mm == m)
                rows.append({"Strategy": s, "Main matches": m, "Trials": count_m, "Probability": count_m/total})
        res_df = pd.DataFrame(rows)
        st.subheader("Match distribution (probability of exact main matches)")
        st.dataframe(res_df.pivot(index="Main matches", columns="Strategy", values="Probability").fillna(0).sort_index(ascending=False), use_container_width=True)

        st.subheader("Distribution for Random (example)")
        fig2, ax2 = plt.subplots()
        subset = res_df[res_df["Strategy"]=="Random"].sort_values("Main matches")
        ax2.bar(subset["Main matches"].astype(str), subset["Probability"])
        ax2.set_title("Probability by # of main matches (Random)")
        ax2.set_xlabel("Main matches")
        ax2.set_ylabel("Probability")
        st.pyplot(fig2)

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

        figA, axA = plt.subplots()
        axA.bar(topN_df["Number"].astype(str), topN_df["Count"])
        axA.set_title(f"Top {topN} Numbers (Window)")
        axA.set_xlabel("Number"); axA.set_ylabel("Count")
        st.pyplot(figA)

        figB, axB = plt.subplots()
        axB.bar(bottomN_df["Number"].astype(str), bottomN_df["Count"])
        axB.set_title(f"Bottom {topN} Numbers (Window)")
        axB.set_xlabel("Number"); axB.set_ylabel("Count")
        st.pyplot(figB)

        st.subheader("Position heatmap (Window)")
        pos_mat = np.zeros((balls_per_draw, number_max - number_min + 1), dtype=int)
        for _, row in dfw.iterrows():
            for i in range(balls_per_draw):
                num = int(row[f"N{i+1}"])
                if number_min <= num <= number_max:
                    pos_mat[i, num - number_min] += 1
        figC, axC = plt.subplots()
        im = axC.imshow(pos_mat, aspect="auto", origin="lower")
        axC.set_title("Frequency by position (window)")
        axC.set_xlabel("Number"); axC.set_ylabel("N position (0=top)")
        st.pyplot(figC)

        st.subheader("Gap Analysis (full dataset)")
        from collections import defaultdict
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
        figG, axG = plt.subplots()
        axG.bar(top_gap["Number"].astype(str), top_gap["MaxGap"].fillna(0))
        axG.set_title(f"Top {K} Numbers by Max Gap")
        axG.set_xlabel("Number"); axG.set_ylabel("Max Gap (draws)")
        st.pyplot(figG)

st.caption("Data Profiler adds rolling-window insights and gap analysis to verify patterns and spot 'overdue' numbers.")
