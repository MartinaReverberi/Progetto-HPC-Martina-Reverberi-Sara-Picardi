# trasformare i numeri del benchmark in grafici leggibili 
import os
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_CSV = "seq_results.csv"
ROUNDS_CSV = "seq_rounds_vs_n.csv"
OUTDIR = "plots_seq"


# --------------------------
# Utils
# --------------------------
def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved: {filename}")


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    # tu scrivi con delimiter=";" quindi leggiamo con sep=";"
    return pd.read_csv(path, sep=";")


def _pick_time_columns(columns) -> Tuple[str, str, float, str]:
    """
    Ritorna (time_mean_col, time_std_col, factor, ylabel)
    - Se nel CSV ci sono *_ms usa quelli (factor=1)
    - Altrimenti usa i secondi e converte in ms (factor=1000)
    """
    cols = set(columns)

    if "time_mean_ms" in cols and "time_std_ms" in cols:
        return "time_mean_ms", "time_std_ms", 1.0, "Tempo (ms)"

    if "time_mean" in cols and "time_std" in cols:
        # convertiamo in ms per grafici più leggibili
        return "time_mean", "time_std", 1000.0, "Tempo (ms)"

    raise ValueError(
        "Nel file risultati mancano colonne tempo.\n"
        f"Colonne trovate: {sorted(cols)}\n"
        "Mi aspettavo (time_mean_ms,time_std_ms) oppure (time_mean,time_std)."
    )


def _require_columns(df: pd.DataFrame, required: set, csv_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Nel file {csv_name} mancano colonne: {sorted(missing)}\n"
            f"Colonne trovate: {list(df.columns)}"
        )


# --------------------------
# Plots
# --------------------------
def plot_time_mean_vs_n(df: pd.DataFrame, outdir: str,
                        time_mean_col: str, time_factor: float, time_ylabel: str) -> None:
    # 1) Tempo medio vs n (una curva per ogni p)
    fig = plt.figure()
    for p_value in sorted(df["p"].unique()):
        sub = df[df["p"] == p_value].sort_values("n")
        y = sub[time_mean_col] * time_factor
        plt.plot(sub["n"], y, marker="o", label=f"p={p_value}")
    plt.xlabel("Numero di nodi n")
    plt.ylabel(time_ylabel)
    plt.title("Tempo medio vs n (per diversi p)")
    plt.grid(True)
    plt.legend()
    _save(fig, os.path.join(outdir, "01_time_mean_vs_n.png"))
    plt.close(fig)


def plot_time_std_vs_n(df: pd.DataFrame, outdir: str,
                       time_std_col: str, time_factor: float, time_ylabel: str) -> None:
    # 2) Deviazione standard del tempo vs n (una curva per ogni p)
    fig = plt.figure()
    for p_value in sorted(df["p"].unique()):
        sub = df[df["p"] == p_value].sort_values("n")
        y = sub[time_std_col] * time_factor
        plt.plot(sub["n"], y, marker="o", label=f"p={p_value}")
    plt.xlabel("Numero di nodi n")
    plt.ylabel(f"Deviazione standard {time_ylabel.lower()}")
    plt.title("Deviazione standard del tempo vs n (per diversi p)")
    plt.grid(True)
    plt.legend()
    _save(fig, os.path.join(outdir, "02_time_std_vs_n.png"))
    plt.close(fig)


def plot_mis_mean_vs_p(df: pd.DataFrame, outdir: str) -> None:
    # 3) MIS medio vs p (una curva per ogni n)
    fig = plt.figure()
    for n_value in sorted(df["n"].unique()):
        sub = df[df["n"] == n_value].sort_values("p")
        plt.plot(sub["p"], sub["mis_mean"], marker="o", label=f"n={n_value}")
    plt.xlabel("Probabilità p (densità Erdős–Rényi)")
    plt.ylabel("MIS size medio")
    plt.title("MIS size medio vs p (per diversi n)")
    plt.grid(True)
    plt.legend()
    _save(fig, os.path.join(outdir, "03_mis_mean_vs_p.png"))
    plt.close(fig)


def plot_rounds_mean_vs_n_from_results(df: pd.DataFrame, outdir: str) -> None:
    # 4a) Rounds medi vs n (da seq_results.csv) -> una curva per ogni p
    fig = plt.figure()
    for p_value in sorted(df["p"].unique()):
        sub = df[df["p"] == p_value].sort_values("n")
        plt.plot(sub["n"], sub["rounds_mean"], marker="o", label=f"p={p_value}")
    plt.xlabel("Numero di nodi n")
    plt.ylabel("Rounds medi")
    plt.title("Rounds medi vs n (per diversi p)")
    plt.grid(True)
    plt.legend()
    _save(fig, os.path.join(outdir, "04a_rounds_mean_vs_n_per_p.png"))
    plt.close(fig)


def plot_rounds_mean_vs_n_fixed_p(rounds_df: pd.DataFrame, outdir: str) -> None:
    # 4b) Rounds medi vs n (da seq_rounds_vs_n.csv) -> p fisso
    rounds_df = rounds_df.sort_values("n")
    p_val = rounds_df["p"].iloc[0] if "p" in rounds_df.columns and len(rounds_df) > 0 else None

    fig = plt.figure()
    plt.plot(rounds_df["n"], rounds_df["rounds_mean"], marker="o")
    plt.xlabel("Numero di nodi n")
    plt.ylabel("Rounds medi")
    if p_val is None:
        plt.title("Rounds medi vs n (p fisso)")
    else:
        plt.title(f"Rounds medi vs n (p = {p_val})")
    plt.grid(True)
    _save(fig, os.path.join(outdir, "04b_rounds_mean_vs_n_fixed_p.png"))
    plt.close(fig)


# --------------------------
# Main
# --------------------------
def main():
    _ensure_outdir(OUTDIR)

    df = _read_csv(RESULTS_CSV)

    # colonne minime
    _require_columns(df, {"n", "p", "mis_mean", "rounds_mean"}, RESULTS_CSV)

    # tempo: supporta sia secondi (time_mean/time_std) sia ms (time_mean_ms/time_std_ms)
    time_mean_col, time_std_col, time_factor, time_ylabel = _pick_time_columns(df.columns)

    plot_time_mean_vs_n(df, OUTDIR, time_mean_col, time_factor, time_ylabel)
    plot_time_std_vs_n(df, OUTDIR, time_std_col, time_factor, time_ylabel)
    plot_mis_mean_vs_p(df, OUTDIR)
    plot_rounds_mean_vs_n_from_results(df, OUTDIR)

    # opzionale: se hai anche seq_rounds_vs_n.csv, facciamo il grafico dedicato (p fisso)
    if os.path.exists(ROUNDS_CSV):
        rounds_df = _read_csv(ROUNDS_CSV)
        if {"n", "rounds_mean"}.issubset(set(rounds_df.columns)):
            plot_rounds_mean_vs_n_fixed_p(rounds_df, OUTDIR)
        else:
            print(f"WARNING: {ROUNDS_CSV} non ha colonne sufficienti per il grafico (serve n e rounds_mean).")
    else:
        print(f"NOTE: {ROUNDS_CSV} non trovato, salto il grafico rounds vs n (p fisso).")

    print("\nFatto! Trovi tutti i grafici in:", OUTDIR)
    print("Se i tempi sono molto piccoli, qui li sto convertendo in millisecondi quando servono.")


if __name__ == "__main__":
    main()