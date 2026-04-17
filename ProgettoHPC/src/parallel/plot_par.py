import os
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PAR_CSV = "par_results.csv"
OUTDIR = "plots_par"


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


def _pick_time_columns(columns) -> Tuple[str, str, float, str]:
    cols = set(columns)
    if "time_mean_ms" in cols and "time_std_ms" in cols:
        return "time_mean_ms", "time_std_ms", 1.0, "Tempo (ms)"
    if "time_mean" in cols and "time_std" in cols:
        return "time_mean", "time_std", 1000.0, "Tempo (ms)"
    raise ValueError(
        "Nel file risultati mancano colonne tempo.\n"
        f"Colonne trovate: {sorted(columns)}\n"
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
def plot_time_mean_vs_n_per_jobs(df: pd.DataFrame, outdir: str,
                                 time_mean_col: str, time_factor: float, time_ylabel: str) -> None:
    """Tempo medio vs n: una figura per n_jobs, curve per p."""
    for n_jobs in sorted(df["n_jobs"].unique()):
        sub_jobs = df[df["n_jobs"] == n_jobs]
        fig = plt.figure()
        for p_val in sorted(sub_jobs["p"].unique()):
            sub = sub_jobs[sub_jobs["p"] == p_val].sort_values("n")
            y = sub[time_mean_col] * time_factor
            plt.plot(sub["n"], y, marker="o", label=f"p={p_val}")
        plt.xlabel("Numero di nodi n")
        plt.ylabel(time_ylabel)
        plt.title(f"Tempo medio vs n (parallelo) - n_jobs={n_jobs}")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"01_time_mean_vs_n_jobs_{n_jobs}.png"))


def plot_time_mean_vs_jobs_per_n(df: pd.DataFrame, outdir: str,
                                 time_mean_col: str, time_factor: float, time_ylabel: str) -> None:
    """Tempo medio vs n_jobs: una figura per n, curve per p."""
    for n_val in sorted(df["n"].unique()):
        sub_n = df[df["n"] == n_val].sort_values("n_jobs")
        fig = plt.figure()
        for p_val in sorted(sub_n["p"].unique()):
            sub = sub_n[sub_n["p"] == p_val].sort_values("n_jobs")
            y = sub[time_mean_col] * time_factor
            plt.plot(sub["n_jobs"], y, marker="o", label=f"p={p_val}")
        plt.xlabel("Numero di core (n_jobs)")
        plt.ylabel(time_ylabel)
        plt.title(f"Tempo medio vs n_jobs (parallelo) - n={n_val}")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"02_time_mean_vs_jobs_n_{n_val}.png"))


def plot_rounds_mean_vs_n_per_jobs(df: pd.DataFrame, outdir: str) -> None:
    """Rounds medi vs n: una figura per n_jobs, curve per p."""
    for n_jobs in sorted(df["n_jobs"].unique()):
        sub_jobs = df[df["n_jobs"] == n_jobs]
        fig = plt.figure()
        for p_val in sorted(sub_jobs["p"].unique()):
            sub = sub_jobs[sub_jobs["p"] == p_val].sort_values("n")
            plt.plot(sub["n"], sub["rounds_mean"], marker="o", label=f"p={p_val}")
        plt.xlabel("Numero di nodi n")
        plt.ylabel("Rounds medi")
        plt.title(f"Rounds medi vs n (parallelo) - n_jobs={n_jobs}")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"03_rounds_mean_vs_n_jobs_{n_jobs}.png"))


def plot_mis_mean_vs_p_per_jobs(df: pd.DataFrame, outdir: str) -> None:
    """MIS mean vs p: una figura per n_jobs, curve per n."""
    for n_jobs in sorted(df["n_jobs"].unique()):
        sub_jobs = df[df["n_jobs"] == n_jobs]
        fig = plt.figure()
        for n_val in sorted(sub_jobs["n"].unique()):
            sub = sub_jobs[sub_jobs["n"] == n_val].sort_values("p")
            plt.plot(sub["p"], sub["mis_mean"], marker="o", label=f"n={n_val}")
        plt.xlabel("Probabilità p (densità Erdős–Rényi)")
        plt.ylabel("MIS size medio")
        plt.title(f"MIS size medio vs p (parallelo) - n_jobs={n_jobs}")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"04_mis_mean_vs_p_jobs_{n_jobs}.png"))


def plot_speedup_vs_jobs(df: pd.DataFrame, outdir: str) -> None:
    """
    Speedup vs n_jobs per ogni (n, p).

    Richiede che il CSV contenga la colonna 'speedup' calcolata rispetto
    al sequenziale puro (prodotta da bench_par.py aggiornato).
    Traccia anche la curva di speedup ideale (lineare) come riferimento.
    """
    if "speedup" not in df.columns:
        print("WARNING: colonna 'speedup' non trovata, salto plot_speedup_vs_jobs.")
        return

    for n_val in sorted(df["n"].unique()):
        sub_n = df[df["n"] == n_val]
        fig = plt.figure()
        jobs_range = sorted(df["n_jobs"].unique())

        # Linea ideale (speedup lineare)
        plt.plot(jobs_range, jobs_range, linestyle="--", color="gray",
                 label="Speedup ideale")

        for p_val in sorted(sub_n["p"].unique()):
            sub = sub_n[sub_n["p"] == p_val].sort_values("n_jobs")
            plt.plot(sub["n_jobs"], sub["speedup"], marker="o", label=f"p={p_val}")

        plt.xlabel("Numero di core (n_jobs)")
        plt.ylabel("Speedup (rispetto a sequenziale puro)")
        plt.title(f"Speedup vs n_jobs - n={n_val}")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"05_speedup_vs_jobs_n{n_val}.png"))


def plot_amdahl_fit(df: pd.DataFrame, outdir: str) -> None:
    """
    Stima empirica della frazione parallelizzabile 'f' della Legge di Amdahl:

        S(p) = 1 / ((1 - f) + f/p)

    Per ogni coppia (n, p_grafo) con dati di speedup su più n_jobs,
    fittiamo f tramite regressione non lineare (curva fitting).
    Produciamo:
    1. Un grafico speedup osservato vs curva Amdahl fittata.
    2. Una tabella riassuntiva dei valori di f stimati.

    Questo permette di rispondere alla domanda: "quanto dell'algoritmo
    è effettivamente parallelizzabile, empiricamente?"
    """
    if "speedup" not in df.columns:
        print("WARNING: colonna 'speedup' non trovata, salto plot_amdahl_fit.")
        return

    try:
        from scipy.optimize import curve_fit
    except ImportError:
        print("WARNING: scipy non disponibile. "
              "Installa scipy per il fit di Amdahl (pip install scipy).")
        return

    def amdahl(p_arr, f):
        """S(p) = 1 / ((1-f) + f/p), con f ∈ [0, 1]."""
        return 1.0 / ((1.0 - f) + f / p_arr)

    summary_rows = []

    for (n_val, p_val), sub in df.groupby(["n", "p"]):
        sub = sub.sort_values("n_jobs")
        jobs = sub["n_jobs"].values.astype(float)
        speedups = sub["speedup"].values.astype(float)

        if len(jobs) < 3:
            # Troppo pochi punti per fittare in modo affidabile
            continue

        try:
            popt, _ = curve_fit(amdahl, jobs, speedups,
                                p0=[0.5], bounds=(0.0, 1.0))
            f_est = popt[0]
        except RuntimeError:
            f_est = float("nan")

        summary_rows.append({"n": n_val, "p_grafo": p_val, "f_est": f_est})

        # Grafico: dati osservati + curva fittata
        fig = plt.figure()
        jobs_fine = np.linspace(jobs.min(), jobs.max(), 200)
        plt.plot(jobs, speedups, "o", label="Speedup osservato")
        if not np.isnan(f_est):
            plt.plot(jobs_fine, amdahl(jobs_fine, f_est), "--",
                     label=f"Fit Amdahl (f={f_est:.3f})")
        plt.plot(jobs_fine, jobs_fine, ":", color="gray", label="Speedup ideale")
        plt.xlabel("Numero di core (n_jobs)")
        plt.ylabel("Speedup")
        plt.title(f"Fit Legge di Amdahl - n={n_val}, p_grafo={p_val}")
        plt.grid(True)
        plt.legend()
        _save(fig, os.path.join(outdir, f"06_amdahl_fit_n{n_val}_p{p_val}.png"))

    # Tabella riassuntiva f stimata
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        out_path = os.path.join(outdir, "amdahl_f_estimates.csv")
        summary_df.to_csv(out_path, index=False, sep=";")
        print(f"\nStime f (Amdahl) salvate in: {out_path}")
        print(summary_df.to_string(index=False))


def plot_internal_time_breakdown(df: pd.DataFrame, outdir: str,
                                 time_factor: float) -> None:
    """
    Breakdown dei tempi interni (prio / select / update) per un n_jobs fisso.
    Utile per capire quale parte dell'algoritmo domina il tempo totale
    e per giustificare la stima della frazione sequenziale per Amdahl.
    """
    required = {"t_prio_mean", "t_select_mean", "t_update_mean"}
    if not required.issubset(set(df.columns)):
        print("WARNING: colonne timing interni non trovate, salto breakdown.")
        return

    # Usa n_jobs=1 (parallelo con 1 worker) per il breakdown → più confrontabile
    # Se non esiste, usa il primo n_jobs disponibile
    available_jobs = sorted(df["n_jobs"].unique())
    ref_jobs = 1 if 1 in available_jobs else available_jobs[0]
    sub_ref = df[df["n_jobs"] == ref_jobs]

    for p_val in sorted(sub_ref["p"].unique()):
        sub = sub_ref[sub_ref["p"] == p_val].sort_values("n")
        n_vals = sub["n"].values

        t_prio = sub["t_prio_mean"].values * time_factor
        t_sel = sub["t_select_mean"].values * time_factor
        t_upd = sub["t_update_mean"].values * time_factor

        x = np.arange(len(n_vals))
        width = 0.5

        fig, ax = plt.subplots()
        ax.bar(x, t_prio, width, label="Priorità")
        ax.bar(x, t_sel, width, bottom=t_prio, label="Selezione (parallel)")
        ax.bar(x, t_upd, width, bottom=t_prio + t_sel, label="Aggiornamento (seq)")

        ax.set_xticks(x)
        ax.set_xticklabels(n_vals)
        ax.set_xlabel("Numero di nodi n")
        ax.set_ylabel("Tempo (ms)")
        ax.set_title(f"Breakdown tempi interni - p={p_val}, n_jobs={ref_jobs}")
        ax.legend()
        ax.grid(True, axis="y")
        _save(fig, os.path.join(outdir, f"07_time_breakdown_p{p_val}_jobs{ref_jobs}.png"))


# --------------------------
# Main
# --------------------------
def main():
    _ensure_outdir(OUTDIR)

    df = _read_csv(PAR_CSV)

    _require_columns(df, {"n", "p", "n_jobs", "mis_mean", "rounds_mean"}, PAR_CSV)
    time_mean_col, time_std_col, time_factor, time_ylabel = _pick_time_columns(df.columns)

    plot_time_mean_vs_n_per_jobs(df, OUTDIR, time_mean_col, time_factor, time_ylabel)
    plot_time_mean_vs_jobs_per_n(df, OUTDIR, time_mean_col, time_factor, time_ylabel)
    plot_rounds_mean_vs_n_per_jobs(df, OUTDIR)
    plot_mis_mean_vs_p_per_jobs(df, OUTDIR)
    plot_speedup_vs_jobs(df, OUTDIR)
    plot_amdahl_fit(df, OUTDIR)
    plot_internal_time_breakdown(df, OUTDIR, time_factor)

    print("\nFatto! Trovi tutti i grafici in:", OUTDIR)


if __name__ == "__main__":
    main()
