/**Progetto HPC – Maximal Independent Set con algoritmo di Luby**/
Questo progetto implementa l’algoritmo di Luby per il calcolo del Maximal Independent Set (MIS) di un grafo, utilizzando la libreria Joblib per la parallelizzazione.

/**Obiettivo**/
L’obiettivo è confrontare le prestazioni tra:
- versione sequenziale
- versione parallela (Joblib)
analizzando tempi di esecuzione e comportamento al variare dei parametri.

/**Struttura del progetto**/
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

/**Tecnologie utilizzate**/
- Python
- Joblib (parallel computing)
- NumPy / Pandas (analisi dati)
- Matplotlib (grafici)

/**Esperimenti**/
Sono stati eseguiti test variando:
- numero di nodi del grafo
- numero di processi (versione parallela)

I risultati includono:
- tempo medio (time_mean)
- deviazione standard (time_std)
- numero di iterazioni dell’algoritmo

/**Risultati**/
I grafici mostrano il confronto tra implementazione sequenziale e parallela, evidenziando lo speedup ottenuto.

/**Relazione**/
Per una descrizione dettagliata dell’algoritmo, delle scelte implementative e dei risultati, consultare il file RELAZIONE TECNICA HPC.pdf
