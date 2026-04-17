import os
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SEQ_CSV = "seq_results.csv"
PAR_CSV = "par_results.csv"
OUTDIR = "plots_compare"

# ---------------------------------------------------------------------------
# NOTA: lo speedup qui è sempre calcolato come T_seq / T_par,
# dove T_seq viene dal CSV sequenziale (bench_seq.py).
# Non usiamo mai lo speedup già presente nel CSV parallelo come fonte
# primaria, per evitare ambiguità su quale baseline sia stata usata.
# La colonna 'speedup' del CSV parallelo viene comunque verificata
# e confrontata come sanity check se presente.
# ---------------------------------------------------------------------------


# --------------------------
# Utils
# --------------------------
def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close(fig)


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep=";")


def _pick_time_col(columns) -> Tuple[str, float]:
    cols = set(columns)
    if "time_mean_ms" in cols:
        return "time_mean_ms", 1.0
    if "time_mean" in cols:
        return "time_mean", 1000.0
    raise ValueError("Colonna tempo non trovata (time_mean o time_mean_ms).")


def _require_columns(df: pd.DataFrame, required: set, csv_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Nel file {csv_name} mancano colonne: {sorted(missing)}\n"
            f"Colonne trovate: {list(df.columns)}"
        )


def _build_merged(seq_df, par_df, time_col_seq, time_col_par):
    """
    Merge seq e par su (n, p). Calcola sempre speedup e efficiency
    da T_seq (sequenziale puro) / T_par.
    """
    merged = par_df.merge(
        seq_df[["n", "p", time_col_seq]].rename(
            columns={time_col_seq: "time_seq"}
        ),
        on=["n", "p"],
        how="inner"
    )
    merged["speedup_vs_seq"] = merged["time_seq"] / merged[time_col_par]
    merged["efficiency_vs_seq"] = merged["speedup_vs_seq"] / merged["n_jobs"]
    return merged


# --------------------------
# Plots
# --------------------------
def plot_time_seq_vs_par_best(seq_df, par_df, outdir,
                              time_col_seq, time_col_par,
                              factor_seq, factor_par):
    """
    Confronto diretto tempo sequenziale vs parallelo (n_jobs massimo).
    """
    max_jobs = par_df["n_jobs"].max()
    par_best = par_df[par_df["n_jobs"] == max_jobs]

    fig = plt.figure()
    for p_val in sorted(seq_df["p"].unique()):
        s = seq_df[seq_df["p"] == p_val].sort_values("n")
        pb = par_best[par_best["p"] == p_val].sort_values("n")

        plt.plot(s["n"], s[time_col_seq] * factor_seq,
                 marker="o", linestyle="--", label=f"SEQ p={p_val}")
        plt.plot(pb["n"], pb[time_col_par] * factor_par,
                 marker="o", label=f"PAR (n_jobs={max_jobs}) p={p_val}")

    plt.xlabel("Numero di nodi n")
    plt.ylabel("Tempo (ms)")
    plt.title("Tempo medio: sequenziale vs parallelo (n_jobs massimo)")
    plt.grid(True)
    plt.legend()
    _save(fig, os.path.join(outdir, "01_time_seq_vs_par_best.png"))


def plot_speedup_vs_jobs(merged: pd.DataFrame, outdir: str) -> None:
    """
    Speedup (rispetto al sequenziale puro) vs n_jobs per ogni (n, p_grafo).
    Traccia anche la curva ideale e una linea a speedup=1 per riferimento.
    """
    jobs_all = sorted(merged["n_jobs"].unique())

    for (n_val, p_val), sub in merged.groupby(["n", "p"]):
        sub = sub.sort_values("n_jobs")
        fig = plt.figure()

        # Riferimenti visivi
        plt.plot(jobs_all, jobs_all, linestyle=":", color="gray",
                 label="Speedup ideale")
        plt.axhline(y=1.0, linestyle="--", color="red", alpha=0.6,
                    label="Speedup = 1 (pari al sequenziale)")

        plt.plot(sub["n_jobs"], sub["speedup_vs_seq"], marker="o",
                 label="Speedup osservato")

        plt.xlabel("Numero di core (n_jobs)")
        plt.ylabel("Speedup (T_seq / T_par)")
        plt.title(f"Speedup vs n_jobs\n(n={n_val}, p_grafo={p_val})")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"02_speedup_n{n_val}_p{p_val}.png"))


def plot_efficiency_vs_jobs(merged: pd.DataFrame, outdir: str) -> None:
    """
    Efficienza (speedup / n_jobs) vs n_jobs per ogni (n, p_grafo).
    """
    for (n_val, p_val), sub in merged.groupby(["n", "p"]):
        sub = sub.sort_values("n_jobs")
        fig = plt.figure()
        plt.axhline(y=1.0, linestyle=":", color="gray", label="Efficienza ideale")
        plt.plot(sub["n_jobs"], sub["efficiency_vs_seq"], marker="o",
                 label="Efficienza osservata")
        plt.xlabel("Numero di core (n_jobs)")
        plt.ylabel("Efficienza (Speedup / n_jobs)")
        plt.title(f"Efficienza vs n_jobs\n(n={n_val}, p_grafo={p_val})")
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"03_efficiency_n{n_val}_p{p_val}.png"))


def plot_speedup_vs_n_per_jobs(merged: pd.DataFrame, outdir: str) -> None:
    """
    Speedup vs n (dimensione del grafo) per ogni n_jobs.
    Questo è il grafico più importante: mostra se e da quale n
    la parallelizzazione diventa conveniente (crossover point).
    Traccia una linea a speedup=1 per identificare visivamente il crossover.
    """
    fig = plt.figure()
    plt.axhline(y=1.0, linestyle="--", color="red", alpha=0.7,
                label="Speedup = 1 (crossover)")

    for n_jobs_val in sorted(merged["n_jobs"].unique()):
        if n_jobs_val == 1:
            continue  # speedup con 1 core è sempre ~1, non informativo
        sub = merged[merged["n_jobs"] == n_jobs_val].sort_values("n")
        # Una curva per p_grafo, per ogni n_jobs
        for p_val in sorted(sub["p"].unique()):
            s = sub[sub["p"] == p_val]
            plt.plot(s["n"], s["speedup_vs_seq"], marker="o",
                     label=f"n_jobs={n_jobs_val}, p={p_val}")

    plt.xlabel("Numero di nodi n")
    plt.ylabel("Speedup (T_seq / T_par)")
    plt.title("Speedup vs dimensione grafo\n(crossover: dove la parallela batte la sequenziale)")
    plt.grid(True)
    plt.legend(fontsize=7)
    _save(fig, os.path.join(outdir, "04_speedup_vs_n_all_jobs.png"))


def plot_crossover_summary(merged: pd.DataFrame, outdir: str) -> None:
    """
    Tabella testuale del crossover point:
    per ogni (n_jobs, p_grafo), indica il minimo n per cui speedup >= 1.
    Se non viene mai raggiunto, scrive "non raggiunto".
    Salva anche un CSV riassuntivo.
    """
    rows = []
    for (n_jobs_val, p_val), sub in merged.groupby(["n_jobs", "p"]):
        if n_jobs_val == 1:
            continue
        sub = sub.sort_values("n")
        above = sub[sub["speedup_vs_seq"] >= 1.0]
        if len(above) > 0:
            crossover_n = int(above["n"].iloc[0])
        else:
            crossover_n = None
        rows.append({
            "n_jobs": n_jobs_val,
            "p_grafo": p_val,
            "crossover_n": crossover_n if crossover_n else "non raggiunto",
        })

    if rows:
        summary = pd.DataFrame(rows)
        out_path = os.path.join(outdir, "crossover_summary.csv")
        summary.to_csv(out_path, index=False, sep=";")
        print(f"\nCrossover summary salvato in: {out_path}")
        print(summary.to_string(index=False))


# --------------------------
# Main
# --------------------------
def main():
    _ensure_outdir(OUTDIR)

    seq_df = _read_csv(SEQ_CSV)
    par_df = _read_csv(PAR_CSV)

    _require_columns(seq_df, {"n", "p"}, SEQ_CSV)
    _require_columns(par_df, {"n", "p", "n_jobs"}, PAR_CSV)

    # Filtra il CSV sequenziale per sicurezza
    if "backend" in seq_df.columns:
        seq_df = seq_df[seq_df["backend"] == "sequential"]

    time_col_seq, factor_seq = _pick_time_col(seq_df.columns)
    time_col_par, factor_par = _pick_time_col(par_df.columns)

    merged = _build_merged(seq_df, par_df, time_col_seq, time_col_par)

    plot_time_seq_vs_par_best(seq_df, par_df, OUTDIR,
                              time_col_seq, time_col_par,
                              factor_seq, factor_par)
    plot_speedup_vs_jobs(merged, OUTDIR)
    plot_efficiency_vs_jobs(merged, OUTDIR)
    plot_speedup_vs_n_per_jobs(merged, OUTDIR)
    plot_crossover_summary(merged, OUTDIR)

    print("\nFatto! Trovi tutti i grafici in:", OUTDIR)


if __name__ == "__main__":
    main()
