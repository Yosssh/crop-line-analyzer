import numpy as np
from scipy.spatial import cKDTree
from pyproj import Transformer

def lat_long_to_utm(latitud, longitud, zona_utm=29):
    """Convierte coordenadas de latitud y longitud a UTM."""

    epsg_utm = 32600 + zona_utm  # Calcula el código EPSG de la zona UTM
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}")
    este, norte = transformer.transform(latitud, longitud)
    return este, norte

def assign_row_ids(df, x_col='X', y_col='Y'):
    """
    1) Estima la dirección de avance (PCA 2D)
    2) Proyecta a coordenada cross-track
    3) Binning 1D robusto para asignar id de línea
    """
    P = df[[x_col, y_col]].to_numpy(dtype=float)
    P -= P.mean(axis=0, keepdims=True)
    # PCA por SVD
    U, S, Vt = np.linalg.svd(P, full_matrices=False)
    v_long = Vt[0]            # dirección a lo largo de la marcha
    v_cross = Vt[1]           # perpendicular (para separar líneas)

    cross = P @ v_cross       # coordenada perpendicular
    # ancho de banda: mediana de diferencias entre cross ordenado
    cross_sorted = np.sort(cross)
    diffs = np.diff(cross_sorted)
    bw = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 1.0
    if not np.isfinite(bw) or bw <= 0:
        bw = 1.0
    row_id = np.round(cross / bw).astype(int)

    df = df.copy()
    df['_vlong_x'] = v_long[0]
    df['_vlong_y'] = v_long[1]
    df['_vcross_x'] = v_cross[0]
    df['_vcross_y'] = v_cross[1]
    df['_cross'] = cross
    df['row_id'] = row_id
    return df

def translate_along_track(df, dt, x_col='X', y_col='Y', speed_col='Velocidad (km/h)', time_col=None):
    """
    Ecs. (1)-(2) del artículo: desplaza (x_i, y_i) a lo largo del vector local
    formado por el segmento (i-1 -> i) una distancia v_i * dt.
    Velocidad en m/s. Para i=0, usa dirección del punto siguiente.
    """
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    v_kmh = df[speed_col].to_numpy(dtype=float)
    v = v_kmh / 3.6  # m/s

    # Asegurar orden temporal
    if time_col is not None:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
        # Reindexar arrays conforme al orden temporal
        x, y, v = df_sorted[x_col].to_numpy(), df_sorted[y_col].to_numpy(), (df_sorted[speed_col].to_numpy()/3.6)
    else:
        df_sorted = df.reset_index(drop=True)

    # vector dirección local: i-1 -> i
    vxi = x - np.roll(x, 1)
    vyi = y - np.roll(y, 1)
    vxi[0] = np.roll(x, -1)[0] - x[0]
    vyi[0] = np.roll(y, -1)[0] - y[0]
    seg_len = np.hypot(vxi, vyi)
    seg_len[seg_len == 0] = 1.0  # evita división por cero

    # unidad hacia atrás (xi-1 -> xi) como en la Ec. (1)
    ex = (np.roll(x, 1) - x) / seg_len
    ey = (np.roll(y, 1) - y) / seg_len
    ex[0] = (x[0] - np.roll(x, -1)[0]) / seg_len[0]
    ey[0] = (y[0] - np.roll(y, -1)[0]) / seg_len[0]

    shift = v * dt  # distancia = v_i * Δt
    x_corr = x + ex * shift
    y_corr = y + ey * shift

    df_out = df_sorted.copy()
    df_out['X_corr'] = x_corr
    df_out['Y_corr'] = y_corr
    return df_out

def mean_log_abs_nn_diff(df_corr, z_col, search_side_m=4.0):
    """
    Calcula el promedio de ln|Δz| usando el vecino más cercano
    SOLO en líneas adyacentes (S2 del artículo).
    Búsqueda en vecindario cuadrado de lado 'search_side_m'.
    """
    # Construir KDTree por cada par de líneas adyacentes para eficiencia simple
    # Primero, indexar por row_id
    grouped = {rid: sub.reset_index(drop=True) for rid, sub in df_corr.groupby('row_id')}
    rids = sorted(grouped.keys())

    ln_diffs = []

    half = search_side_m / 2.0

    for rid in rids:
        src = grouped[rid]
        if rid - 1 in grouped:
            ln_diffs.extend(_collect_ln_diffs_between(src, grouped[rid-1], z_col, half))
        if rid + 1 in grouped:
            ln_diffs.extend(_collect_ln_diffs_between(src, grouped[rid+1], z_col, half))

    ln_diffs = np.array(ln_diffs, dtype=float)
    ln_diffs = ln_diffs[np.isfinite(ln_diffs)]
    if ln_diffs.size == 0:
        return np.inf
    return ln_diffs.mean()

def _collect_ln_diffs_between(A, B, z_col, half_side):
    """
    Para cada punto en A, busca el vecino más cercano en B dentro de un cuadrado de lado '2*half_side'
    centrado en el punto de A. Si existe, añade ln|Δz|.
    """
    # KDTree en B (coordenadas corregidas)
    Bxy = np.c_[B['X_corr'].to_numpy(), B['Y_corr'].to_numpy()]
    tree = cKDTree(Bxy)
    Axy = np.c_[A['X_corr'].to_numpy(), A['Y_corr'].to_numpy()]

    ln_diffs = []
    # Primero, filtro espacial por cuadrado usando consulta por radio en círculo de radio half_side*sqrt(2)
    radius = half_side * np.sqrt(2.0)
    for i, (x, y) in enumerate(Axy):
        idxs = tree.query_ball_point([x, y], r=radius)
        if not idxs:
            continue
        # aplicar ventana cuadrada exacta
        candidates = []
        for j in idxs:
            if abs(Bxy[j,0] - x) <= half_side and abs(Bxy[j,1] - y) <= half_side:
                candidates.append(j)
        if not candidates:
            continue
        # vecino más cercano dentro del cuadrado
        d2 = (Bxy[candidates,0]-x)**2 + (Bxy[candidates,1]-y)**2
        jmin = candidates[int(np.argmin(d2))]
        dz = A[z_col].iat[i] - B[z_col].iat[jmin]
        val = np.log(abs(dz) + 1e-12)  # evita log(0)
        ln_diffs.append(val)
        
    return ln_diffs

def optimize_time_lag(
    df, z_col, x_col='X', y_col='Y', speed_col='Velocidad (km/h)', time_col='Vehicle date',
    dt_min=0.1, dt_max=2.0, dt_step=0.05, search_side_m=4.0
):
    """
    Bucle sobre Δt en [dt_min, dt_max] para minimizar promedio de ln|Δz| (S2; vecinos en líneas adyacentes).
    Devuelve Δt óptimo y DataFrame con coordenadas corregidas finales.
    """
    # 1) asigna líneas
    df1 = assign_row_ids(df, x_col=x_col, y_col=y_col)
    # 2) ordenar por tiempo si está
    if time_col in df1.columns:
        df1 = df1.sort_values(time_col).reset_index(drop=True)
    best_dt, best_score, best_df_corr = None, np.inf, None

    dts = np.arange(dt_min, dt_max + 1e-9, dt_step)
    for dt in dts:
        df_corr = translate_along_track(df1, dt, x_col=x_col, y_col=y_col,
                                        speed_col=speed_col, time_col=time_col)
        # mismo row_id que df1
        df_corr['row_id'] = df1['row_id'].values
        score = mean_log_abs_nn_diff(df_corr, z_col=z_col, search_side_m=search_side_m)
        if score < best_score:
            best_score, best_dt, best_df_corr = score, dt, df_corr

    return best_dt, best_score, best_df_corr