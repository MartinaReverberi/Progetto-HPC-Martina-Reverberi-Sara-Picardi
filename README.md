# Progetto HPC – Maximal Independent Set con algoritmo di Luby
Questo progetto implementa l’algoritmo di **Luby** per il calcolo del **Maximal Independent Set (MIS)** di un grafo, utilizzando la libreria **Joblib** per la parallelizzazione.

## Obiettivo
L’obiettivo è confrontare le prestazioni tra:
- versione sequenziale
- versione parallela con Joblib
analizzando tempi di esecuzione e comportamento al variare dei parametri.

## Struttura del progetto
```text
ProgettoHPC/
│
├── src/
│   ├── sequential/     # implementazione sequenziale
│   ├── parallel/       # implementazione parallela con Joblib
│   └── common/         # funzioni condivise
│
├── test cluster/
│   ├── *.csv           # risultati degli esperimenti
│   ├── plots_seq/      # grafici versione sequenziale
│   ├── plots_par/      # grafici versione parallela
│   └── plots_compare/  # confronto tra le due versioni
│
└── RELAZIONE TECNICA HPC.pdf
