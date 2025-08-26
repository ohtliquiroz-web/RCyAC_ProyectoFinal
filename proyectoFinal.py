import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
import networkx as nx

# ================== CONFIGURACIÓN ==================
CSV_PATH = "Data/Estadísticas_operativas_seleccionadas_Mayor.csv"
index_col_name = 0  # si la primera columna contiene el nombre del producto
n_estimators = 500 # Número de arboles para el Random Forest
random_state = 50 # Semilla
n_boot = 30  # número de bootstraps temporales
top_percent = 85  # percentil de corte para aristas (mantener las más fuertes)

# ================== CARGA DE DATOS ==================
df = pd.read_csv(CSV_PATH,encoding="latin-1", index_col=index_col_name)

# Normaliza cada fila (producto) con z-score
df_z = df.T.apply(lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-9)).T

series_names = df_z.index.tolist()
n = df_z.shape[0]

# ================== MATRIZ DE INFLUENCIAS (RF + BOOTSTRAP) ==================
W = pd.DataFrame(0.0, index=series_names, columns=series_names)  # fuente -> objetivo

rng = np.random.RandomState(random_state)
for target in series_names:
    y = df_z.loc[target, :].values
    # predictores: todas las demás series
    X = df_z.drop(index=target).T.values
    feat_names = df_z.drop(index=target).index.tolist()

    imp_accum = np.zeros(len(feat_names), dtype=float)

    for b in range(n_boot):
        # bootstrap por columnas (meses)
        idx = resample(np.arange(len(y)), replace=True, random_state=rng.randint(0, 1_000_000))
        Xb = X[idx, :]
        yb = y[idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=rng.randint(0, 1_000_000),
            n_jobs=-1,
            max_features="sqrt"
        )
        rf.fit(Xb, yb)

        # importancia por permutación
        pi = permutation_importance(rf, Xb, yb, n_repeats=5,
                                    random_state=rng.randint(0, 1_000_000), n_jobs=-1)
        imp_accum += np.maximum(0, pi.importances_mean)

    imp_mean = imp_accum / n_boot
    if imp_mean.max() > 0:
        imp_mean = imp_mean / imp_mean.max()

    for k, src in enumerate(feat_names):
        W.loc[src, target] = imp_mean[k]

W.to_csv('matrizInfluencias.csv', index=True)

# ================== CONSTRUCCIÓN DE LA RED ==================
thr = np.percentile(W.values[W.values > 0], top_percent) if (W.values > 0).sum() > 0 else 1.0
A = (W >= thr).astype(int)  # adyacencia binaria por percentil
G = nx.from_pandas_adjacency(pd.DataFrame(A, index=series_names, columns=series_names),
                             create_using=nx.DiGraph)

# Convertir a no dirigido para small-world clásico
G_und = G.to_undirected()
G_und.remove_nodes_from(list(nx.isolates(G_und)))  # elimina nodos aislados


# ================== FUNCIÓN SMALL-WORLD ==================
def small_world(Gu):
    if Gu.number_of_nodes() < 3 or Gu.number_of_edges() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    # componente gigante
    components = list(nx.connected_components(Gu))
    GC = Gu.subgraph(max(components, key=len)).copy()

    C = nx.transitivity(GC)  # clustering global
    L = nx.average_shortest_path_length(GC)

    # Grafo aleatorio ER con mismo n y p
    n_nodes = GC.number_of_nodes()
    m_edges = GC.number_of_edges()
    p = (2 * m_edges) / (n_nodes * (n_nodes - 1))
    ER = nx.gnp_random_graph(n_nodes, p, seed=123)
    Cr = nx.transitivity(ER)
    Lr = nx.average_shortest_path_length(ER)

    S = (C / Cr) / (L / Lr) if (Cr > 0 and Lr > 0) else np.nan
    return C, L, Cr, Lr, S


# ================== RESULTADOS ==================
C, L, Cr, Lr, S = small_world(G_und)
print(f"Nodos: {G_und.number_of_nodes()}, Aristas: {G_und.number_of_edges()}")
print(f"C={C:.3f}, L={L:.3f}, C_rand={Cr:.3f}, L_rand={Lr:.3f}, S={S:.3f}")

if S > 1:
    print("La red muestra propiedades de mundo pequeño.")
else:
    print("No se detectan propiedades claras de mundo pequeño.")
