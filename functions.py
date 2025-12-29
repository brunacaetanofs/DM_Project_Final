import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mpl_colors
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.base import clone
from matplotlib.patches import RegularPolygon
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from mpl_toolkits.axes_grid1 import make_axes_locatable
from minisom import MiniSom
import scipy.cluster.hierarchy as sch


#Clustering metrics (SS, SSB, SSW, R2)

##calculates SS:
def get_ss(df, feats):
    """
    Calculate the sum of squares (SS) for the given DataFrame.
    The sum of squares is computed as the sum of the variances of each column
    multiplied by the number of non-NA/null observations minus one.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame for which the sum of squares is to be calculated.
    feats (list of str): A list of feature column names to be used in the calculation.
    
    Returns:
    float: The sum of squares of the DataFrame.
    """
    df_ = df[feats]
    ss = np.sum(df_.var() * (df_.count() - 1))
    return ss

##calculates SSB:
def get_ssb(df, feats, label_col):
    """
    Calculate the between-group sum of squares (SSB) for the given DataFrame.
    The between-group sum of squares is computed as the sum of the squared differences
    between the mean of each group and the overall mean, weighted by the number of observations
    in each group.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column in the DataFrame that contains the group labels.
    
    Returns:
    float: The between-group sum of squares of the DataFrame.
    """
    ssb_i = 0
    for i in np.unique(df[label_col]):
        df_ = df.loc[:, feats]
        X_ = df_.values
        X_k = df_.loc[df[label_col] == i].values
        
        ssb_i += (X_k.shape[0] * (np.square(X_k.mean(axis=0) - X_.mean(axis=0))))
    
    ssb = np.sum(ssb_i)
    return ssb

##calculates SSW:
def get_ssw(df, feats, label_col):
    """
    Calculate the sum of squared within-cluster distances (SSW) for a given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data.
    feats (list of str): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing cluster labels.
    
    Returns:
    float: The sum of squared within-cluster distances (SSW).
    """
    feats_label = feats + [label_col]
    df_k = df[feats_label].groupby(by=label_col).apply(
        lambda col: get_ss(col, feats),
        include_groups=False
    )
    return df_k.sum()

##calculates R^2:
def get_rsq(df, feats, label_col):
    """
    Calculate the R-squared value for a given DataFrame and features.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    feats (list): A list of feature column names to be used in the calculation.
    label_col (str): The name of the column containing the labels or cluster assignments.
    
    Returns:
    float: The R-squared value, representing the proportion of variance explained by the clustering.
    """
    df_sst_ = get_ss(df, feats)  # get total sum of squares
    df_ssw_ = get_ssw(df, feats, label_col)  # get ss within
    df_ssb_ = df_sst_ - df_ssw_  # get ss between
    # r2 = ssb/sst
    return (df_ssb_ / df_sst_)

#K selection

##tests different k values:
def get_r2_scores(df, feats, clusterer, min_k=1, max_k=9):
    """
    Loop over different values of k. To be used with sklearn clusterers.
    """
    r2_clust = {}
    for n in range(min_k, max_k):
        clust = clone(clusterer).set_params(n_clusters=n)
        labels = clust.fit_predict(df)
        df_concat = pd.concat([df,
                               pd.Series(labels, name='labels', index=df.index)], axis=1)
        r2_clust[n] = get_rsq(df_concat, feats, 'labels')
    return r2_clust



def visualize_silhouette_graf(df, range_clusters=[2, 3, 4, 5, 6]):
    
    # --- SETUP THE GRID ---
    # 2 Rows, 5 Columns = 10 slots available
    n_rows = 2
    n_cols = 5
    
    # Create the figure and array of axes
    # figsize is (width, height) - made it wide to fit 5 columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(25, 10))
    
    # Flatten allows us to iterate through the grid as a simple list (0 to 9)
    axes_flat = axes.flatten() 

    # --- LOOP THROUGH CLUSTERS ---
    for i, nclus in enumerate(range_clusters):
        
        # Safety check: stop if we have more clusters than plots
        if i >= len(axes_flat):
            break
            
        ax = axes_flat[i] # Get the current subplot
        
        # 1. K-Means
        clusterer = KMeans(n_clusters=nclus, init='k-means++', n_init=10, random_state=42)
        cluster_labels = clusterer.fit_predict(df)

        # 2. Average Score 
        silhouette_avg = silhouette_score(df, cluster_labels)
        print(f"For n_clusters = {nclus}, the average score is: {silhouette_avg:.4f}")

        # 3. Setup the subplot (ax)
        ax.set_xlim([-0.1, 1])
        # The (nclus + 1) * 10 is to insert blank space between silhouette plots
        ax.set_ylim([0, len(df) + (nclus + 1) * 10])

        sample_silhouette_values = silhouette_samples(df, cluster_labels)

        y_lower = 10
        for j in range(nclus):
            # Aggregate the silhouette scores for samples belonging to cluster j, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
            ith_cluster_silhouette_values.sort()

            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = cm.nipy_spectral(float(j) / nclus)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10 

        # 4. Titles and Labels
        ax.set_title(f"K = {nclus} (Avg: {silhouette_avg:.2f})", fontsize=11)
        ax.set_xlabel("Silhouette Coeff")
        ax.set_ylabel("Cluster Label")

        # Vertical line for average silhouette score
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        # Remove y-ticks for cleaner look
        ax.set_yticks([]) 

    # --- CLEANUP ---
    # Hide any empty subplots (if you have fewer clusters than grid slots)
    for k in range(len(range_clusters), len(axes_flat)):
        axes_flat[k].axis('off')

    plt.tight_layout()
    plt.show()

def plot_k_distance(df, features, k=None):
    """
    Plots the k-distance graph to help find the optimal EPS for DBSCAN.
    """
    data = df[features]
    
    # Automatic logic for k if not provided: 2 * dimensions
    if k is None:
        k = 2 * len(features)
    
    # 1. Fit Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    
    # 2. Sort distances
    # We take the distance to the k-th neighbor (column k-1)
    sorted_distances = np.sort(distances[:, k-1], axis=0)
    
    # 3. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_distances, color='darkgreen', linewidth=2)
    plt.title(f"k-Distance Plot (k={k}) for EPS estimation")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("k-distance (EPS candidate)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    return k # Returns k so you know what min_samples to use later

def get_dbscan(df, features, eps=1.8, min_samples=7):
    
    # 1. Executar o Modelo
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(df[features])
    
    # 2. Calcular Estatísticas
    # Contar quantos clientes há em cada label
    distribution = pd.Series(labels).value_counts().sort_index()
    
    # O número de clusters é o total de labels únicos, ignorando o -1 (ruído)
    n_clusters = len(distribution) - (1 if -1 in distribution.index else 0)
    
    n_noise = distribution.get(-1, 0)
    perc_noise = (n_noise / len(df)) * 100
    
    # 3. Print para leitura imediata
    print(f"Clusters found: {n_clusters}")
    print(f"Noise: {n_noise} customers ({perc_noise:.2f}%)")
    print("Cluster Distribution:")
    print(distribution)
    
    # 4. Calcular R2 (Opcional, mas útil)
    try:
        temp_df = df[features].copy()
        temp_df['labels_temp'] = labels
        # Se tiveres a tua função 'func' importada:
        r2 = get_rsq(temp_df, features, 'labels_temp')
        print(f"R2 Score: {r2:.4f}")
    except:
        pass
        
    
    # RETORNAR AS 3 COISAS QUE PEDISTE
    return labels, n_clusters, distribution

def get_hdbscan(df, features, min_cluster_size=200, min_samples=None, selection_method='eom'):
    """
    Executa o HDBSCAN e retorna:
    1. Labels (Array)
    2. Número de Clusters (Int)
    3. Distribuição dos Clientes (Series)
    """
    
    # Se min_samples não for definido, por defeito o HDBSCAN usa igual ao min_cluster_size, 
    # mas aqui deixamos explícito como None para ele decidir ou o utilizador definir
    
    # 1. Executar o Modelo
    hdb = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method=selection_method,
        n_jobs=-1 # Usa todos os processadores
    )
    
    labels = hdb.fit_predict(df[features])
    
    # 2. Calcular Estatísticas
    distribution = pd.Series(labels).value_counts().sort_index()
    
    # O número de clusters ignora o label -1 (ruído)
    n_clusters = len(distribution) - (1 if -1 in distribution.index else 0)
    
    n_noise = distribution.get(-1, 0)
    perc_noise = (n_noise / len(df)) * 100
    
    print(f"Clusters found: {n_clusters}")
    print(f"Noise: {n_noise} customers ({perc_noise:.2f}%)")
    print("Cluster distribution:")
    print(distribution)
    
    # 3. Calcular R2
    try:
        temp_df = df.copy()
        temp_df['labels_temp'] = labels
        
        # Chama a tua função auxiliar get_rsq (que já deve estar neste ficheiro)
        r2 = get_rsq(temp_df, features, 'labels_temp')
        print(f"R2 Score: {r2:.4f}")
    except Exception as e:
        pass
    
    return labels, n_clusters, distribution

def get_meanshift(df, features):
    # quantiles to test
    quantiles = [0.25, 0.2, 0.15, 0.1, 0.08, 0.06]

    for q in quantiles:
        bw = estimate_bandwidth(df[features], quantile=q, n_samples=500, random_state=42)
        
        if bw < 0.01: continue 
        
        ms = MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
        labels = ms.fit_predict(df[features])
        n_clusters = len(np.unique(labels))

        # if the result is valid (between 2 and 15 clusters)
        if 2 <= n_clusters <= 15:
            print(f"Used quantile {q} -> {n_clusters} clusters found.")
            return labels
    return labels

def get_n_components(df, features, cov_types=("diag", "full")):
    """
    Determine the optimal number of components for Gaussian Mixture Models
    using the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).
    """
    n_components = range(1, 11)

    for cov in cov_types:
        bic_values = []
        aic_values = []

        for k in n_components:
            model = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                n_init=3,
                random_state=1
            ).fit(df[features])

            bic_values.append(model.bic(df[features]))
            aic_values.append(model.aic(df[features]))

        plt.figure(figsize=(10, 6))
        plt.plot(n_components, bic_values, "o-", label="BIC")
        plt.plot(n_components, aic_values, "s-", label="AIC")
        plt.xlabel("n_components")
        plt.ylabel("Criterion value")
        plt.title(f"GMM model selection ({cov} covariance)")
        plt.xticks(list(n_components))
        plt.legend()
        plt.show()


def fit_gmm_segmentation(df, features, k, cov_type='full', rsq_func=None):
    """
    Fits a GMM model, assigns labels to the dataframe, and calculates R2 if a function is provided.
    
    Parameters:
    - df: The input dataframe
    - features: List of column names to use for clustering
    - k: Number of components 
    - cov_type: Covariance type (default 'full')
    - rsq_func: (Optional) Your custom get_rsq function
    
    Returns:
    - gmm_model: The fitted model object
    - df_result: Dataframe with a new 'gmm_labels' column
    """
    # 1. Initialize and Fit
    gmm = GaussianMixture(n_components=k, covariance_type=cov_type, 
                          n_init=10, init_params='kmeans', random_state=1)
    
    # 2. Predict Labels
    labels = gmm.fit_predict(df[features])
    
    # 3. Create Result Dataframe
    df_result = df.copy()
    df_result['gmm_labels'] = labels
    
    # 4. Calculate R^2 
    r2 = rsq_func(df_result, features, 'gmm_labels')
    print(f"GMM ({cov_type}, k={k}) R² score: {r2:.4f}")
    
    # Print cluster distribution
    print("\nCluster Sizes:")
    print(df_result['gmm_labels'].value_counts().sort_index())
    
    return gmm, df_result

def analyze_gmm_uncertainty(model, df, features, threshold):
    """
    Calculates assignment probabilities and plots the uncertainty distribution.
    
    Parameters:
    - model: The fitted GMM model
    - df: The dataframe containing the data
    - features: List of feature columns used for the model
    - threshold: Probability cutoff for 'certain' customers 
    """
    # 1. Get Probabilities
    probabilities = model.predict_proba(df[features])
    max_probs = probabilities.max(axis=1)
    
    # 2. Filter Certain vs Uncertain
    certain_mask = max_probs >= threshold
    uncertain_mask = ~certain_mask
    
    n_certain = certain_mask.sum()
    n_uncertain = uncertain_mask.sum()
    
    # 3. Print Statistics
    print(f"\n--- GMM Probability Analysis ---")
    print(f"Threshold: {threshold:.0%}")
    print(f"Clear assignments: {n_certain:,} customers")
    print(f"Uncertain assignments: {n_uncertain:,} customers ({n_uncertain/len(df):.1%})")
    
    # 4. Plot Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(max_probs, bins=50, edgecolor='black', alpha=0.7, color='#4c72b0')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Uncertainty threshold ({threshold})')
    
    plt.xlabel('Maximum Probability (Confidence)', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.title('Distribution of Segment Assignment Confidence', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def get_model_metrics(df, labels, model_name, perspective):
    """
    Calculates clustering metrics and returns them as a dictionary.
    
    Parameters:
    - df: The dataframe with features used for clustering.
    - labels: The cluster labels output by the model.
    - model_name: String name of the model (e.g., 'K-Means k=5').
    - perspective: String name of the perspective (e.g., 'Value', 'Behavioral').
    
    Returns:
    - dict: A dictionary containing the metrics.
    """
    
    # 1. Handle Noise (DBSCAN/HDBSCAN)
    # We filter out noise (-1) so metrics represent the validity of actual clusters
    mask = labels != -1
    if mask.sum() < len(df) and mask.sum() > 0:
        X_metrics = df[mask].copy()
        labels_metrics = labels[mask]
    else:
        X_metrics = df.copy()
        labels_metrics = labels

    # Check for valid number of clusters
    n_clusters = len(set(labels_metrics))
    if n_clusters < 2:
        print(f"Skipping {model_name}: Less than 2 clusters found.")
        return None

    # 2. Calculate Standard Scikit-Learn Metrics
    sil = silhouette_score(X_metrics, labels_metrics)
    db = davies_bouldin_score(X_metrics, labels_metrics)
    ch = calinski_harabasz_score(X_metrics, labels_metrics)
    
    # 3. Calculate R^2 
    # We use your existing get_rsq function. 
    # We must attach labels temporarily because get_rsq expects a column name.
    X_metrics['temp_labels'] = labels_metrics
    features = [c for c in X_metrics.columns if c != 'temp_labels']
    
    r2 = get_rsq(X_metrics, features, 'temp_labels')
    
    return {
        'Perspective': perspective,
        'Model Name': model_name,
        'Num_Clusters': n_clusters,
        'R2 Score': round(r2, 4),
        'Silhouette Score': round(sil, 3),        # Higher is better
        'Davies-Bouldin': round(db, 3),           # Lower is better
        'Calinski-Harabasz': round(ch, 1)         # Higher is better
    }


def plot_hexagons(som_matrix, som, ax, label='', cmap=None):
    """
    Draws a hexagonal grid based on the SOM matrix values.
    Adapted to use 'ax' directly for easier subplot management.
    """
    som_x, som_y = som.get_weights().shape[:2]
    
    # Normalize values to [0,1] for color mapping
    colornorm = mpl_colors.Normalize(vmin=np.min(som_matrix), vmax=np.max(som_matrix))
    
    # Loop through neurons
    for i in range(som_x):
        for j in range(som_y):
            # Get Euclidean coordinates for the hexagon center
            wx, wy = som.convert_map_to_euclidean((i, j))
            
            # Determine color
            if cmap is None:
                color = np.clip(som_matrix[i, j], 0, 1) # Grayscale if no cmap
            else:
                color = cmap(colornorm(som_matrix[i, j]))
            
            # Draw Hexagon
            hex = RegularPolygon(
                (wx, wy), 
                numVertices=6, 
                radius=np.sqrt(1/3),
                facecolor=color, 
                edgecolor='white', 
                linewidth=0.5
            )
            ax.add_patch(hex)
    
    # Visual adjustments
    ax.set_title(label, fontsize=10)
    
    ax.set_xlim(-1, som_x + 1)  # Adiciona +1 ou +2
    ax.set_ylim(-1, som_y + 1)
    
    ax.set_aspect('equal') # Isto é crucial para não ficarem "ovais"
    ax.axis('off')
    # Add Colorbar (Small bar on the right)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    if cmap is not None:
        sm = cm.ScalarMappable(norm=colornorm, cmap=cmap) 
        sm.set_array([]) 
        plt.colorbar(sm, cax=cax)
    else:
        cax.axis('off')

    return ax

def plot_som_diagnostics(som, data, figsize=(16, 7)):
    """
    Plots the Hits Map (Population) and U-Matrix (Distances) side-by-side.
    
    Parameters:
    - som: The trained MiniSom object.
    - data: The dataframe/numpy array used for training (to calculate hits).
    - figsize: Tuple for figure dimensions.
    """
    
    # 1. Calculate the Matrices
    frequencies = som.activation_response(data) # Hits map
    u_matrix = som.distance_map()               # U-Matrix
    
    # 2. Initialize Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- LEFT PLOT: HITS MAP (Cluster Density) ---
    # Shows how many customers are in each hexagon
    plot_hexagons(frequencies, som, ax1, label='Hits Map (Sample Density)', cmap=cm.Greens)
    
    # Add text annotations (Counts) for the Hits Map
    for i in range(frequencies.shape[0]):
        for j in range(frequencies.shape[1]):
            count = int(frequencies[i, j])
            if count > 0: # Only write if there are customers
                wx, wy = som.convert_map_to_euclidean((i, j))
                # Dynamic text color: White for dark cells, Black for light cells
                text_color = 'white' if count > np.max(frequencies) * 0.5 else 'black'
                ax1.text(wx, wy, str(count), ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')

    # --- RIGHT PLOT: U-MATRIX (Cluster Separation) ---
    # Shows the distance between neurons (Dark Red = Wall/Barrier, Blue = Valley/Cluster Center)
    plot_hexagons(u_matrix, som, ax2, label='U-Matrix (Neighbor Distances)', cmap=cm.RdYlBu_r)
    
    plt.tight_layout()
    plt.show()

def run_som_kmeans(som, data_values, n_clusters=4):
    """
    Applies K-Means clustering on the SOM weights and assigns labels to customers.
    
    Parameters:
    - som: Trained MiniSom object
    - data_values: Numpy array of the data used for training (e.g., customer[features].values)
    - n_clusters: Number of clusters to create
    
    Returns:
    - final_labels: List of cluster assignments for each customer
    - matrix_km: The grid of cluster labels (for plotting)
    """
    
    # 1. Prepare SOM weights
    weights = som.get_weights()
    x_dim, y_dim, n_features = weights.shape
    weights_flat = weights.reshape(-1, n_features)
    
    # 2. Run K-Means
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels_km = kmeans.fit_predict(weights_flat)
    
    # 3. Reshape for visualization
    matrix_km = labels_km.reshape(x_dim, y_dim)
    
    # 4. Plot the K-Means Map
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_hexagons(
        matrix_km, 
        som, 
        ax, 
        label=f'SOM + K-Means Clustering (k={n_clusters})', 
        cmap=cm.Spectral_r
    )
    plt.show()
    
    # 5. Assign labels to original customers
    # We find the winner neuron for each customer and assign its cluster label
    final_labels = []
    for x in data_values:
        w = som.winner(x)
        cluster_label = matrix_km[w[0], w[1]]
        final_labels.append(cluster_label)
        
    return final_labels


def run_som_hierarchical(som, data_values, n_clusters=5, cmap=cm.Spectral_r, figsize=(20, 8)):
    """
    Executa o clustering hierárquico nos pesos do SOM e mostra
    o Dendrograma e o Mapa resultante lado a lado.
    """
    
    # 1. Preparar os pesos do SOM
    weights = som.get_weights()
    x_dim, y_dim, n_features = weights.shape
    weights_flat = weights.reshape(-1, n_features)
    
    linkage_matrix = sch.linkage(weights_flat, method='ward')
    if n_clusters > 1:
        dist_x = linkage_matrix[-n_clusters, 2]
        dist_y = linkage_matrix[-(n_clusters-1), 2]
        cut_height = (dist_x + dist_y) / 2
    else:
        cut_height = 0 # Default fallback
    
    # 2. Criar a figura com 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # --- GRÁFICO DA ESQUERDA: DENDROGRAMA ---
    # Calcular a matriz de ligação (linkage)
    linkage_matrix = sch.linkage(weights_flat, method='ward')
    
    # Definir o eixo atual como o da esquerda (ax1) para o dendrograma desenhar lá
    plt.sca(ax1) 
    
    # Desenhar o dendrograma
    # no_labels=True esconde os números dos neurónios no eixo X para ficar mais limpo
    dend = sch.dendrogram(
        linkage_matrix, 
        no_labels=True, 
        color_threshold=cut_height, # colors
        above_threshold_color='grey' # color for the trunk above the cut
    )
    
    ax1.set_title("Dendrogram of SOM Neurons (Ward Linkage)", fontsize=14)
    ax1.set_xlabel('Neurons (Hexagons)')
    ax1.set_ylabel('Euclidean Distance')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Isto é uma aproximação visual baseada na altura das últimas fusões
    if n_clusters > 1:
        # Encontrar as alturas das últimas k-1 fusões para estimar a linha de corte
        last_merges = linkage_matrix[-(n_clusters-1):, 2]
        cut_height = (last_merges[0] + linkage_matrix[-n_clusters, 2]) / 2
        ax1.axhline(y=cut_height, c='grey', lw=2, linestyle='--', label=f'Approx. Cut for k={n_clusters}')
        ax1.legend()

    # --- GRÁFICO DA DIREITA: MAPA HIERÁRQUICO ---
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels_hc = hc.fit_predict(weights_flat)
    matrix_hc = labels_hc.reshape(x_dim, y_dim)
    
    # Usar a função plot_hexagons no eixo da direita (ax2)
    plot_hexagons(
        matrix_hc, 
        som, 
        ax2, 
        label=f'SOM + Hierarchical Clustering (k={n_clusters})', 
        cmap=cm.Spectral_r
    )
    
    plt.tight_layout()
    plt.show()

    final_labels = []
    for x in data_values:
        w = som.winner(x)
        cluster_label = matrix_hc[w[0], w[1]]
        final_labels.append(cluster_label)
        
    return final_labels



def analyze_feature(feature: pd.Series, feature_name: str):

    sns.set_style("whitegrid")
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['grid.alpha'] = 0.3

    # ---- Descriptive Statistics  ----
    desc = feature.describe()
    n_missing = feature.isna().sum()
    skewness = skew(feature.dropna())
    kurt = kurtosis(feature.dropna())

    print(f"--- Summary of {feature_name} ---")
    print(desc)
    print(f"Missing values: {n_missing}")
    print(f"Skewness: {skewness:.2f}")
    print(f"Kurtosis: {kurt:.2f}")

    spotify_green = '#1DB954'

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma + KDE
    sns.histplot(feature, kde=True, ax=axes[0], color=spotify_green)
    axes[0].set_title(f'{feature_name} Distribution')

    # Boxplot
    sns.boxplot(x=feature, ax=axes[1], color=spotify_green)
    axes[1].set_title(f'{feature_name} Boxplot')

    plt.tight_layout()
    plt.show()

from scipy.stats import linregress


def cumulative_customer(col, flight):
    total = flight.groupby("Loyalty#")[col].sum().reset_index()
    return total


def calculate_seasonality(df):
    df = df.copy()
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'

    df['Season'] = df['Month'].apply(get_season)

    # 2. Calcular Ratios por Estação (Summer_Ratio, etc.)
    season_counts = df.pivot_table(index='Loyalty#', columns='Season', values='NumFlights', aggfunc='sum', fill_value=0)
    total = season_counts.sum(axis=1).replace(0, 1) # Evitar divisão por zero
    season_ratios = season_counts.div(total, axis=0).add_suffix('_Ratio').reset_index()

    # 3. Índice de Sazonalidade (Desvio Padrão dos meses)
    monthly = df.pivot_table(index='Loyalty#', columns='Month', values='NumFlights', aggfunc='sum', fill_value=0)
    # Se desvio padrão é alto, cliente é muito sazonal
    season_index = (monthly.std(axis=1) / (monthly.mean(axis=1) + 0.001)).reset_index(name='Seasonality_Index')

    # Juntar as duas partes de sazonalidade
    return season_ratios.merge(season_index, on='Loyalty#')

def calculate_trend(df):
    # Função auxiliar para calcular declive (slope)
    def get_slope(series):
        if len(series) < 2: return 0
        # Cria uma linha de tendência (y=mx+b) e devolve o m (slope)
        return linregress(np.arange(len(series)), series.values)[0]

    df_sorted = df.sort_values(['Loyalty#', 'YearMonthDate'])


    trends = df_sorted.groupby('Loyalty#')['NumFlights'].apply(get_slope).reset_index(name='Flight_Trend_Slope')
    return trends

def plot_pca_variance(data,features,chosen_components=2,
                      threshold=0.80,figsize=(10, 5),random_state=42
                      ):
    """
    Plots individual and cumulative explained variance for PCA.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset containing the features.
    features : list
        List of feature names to use in PCA.
    chosen_components : int, default=2
        Number of principal components highlighted.
    threshold : float, default=0.80
        Variance threshold to show (e.g. 0.80 = 80%).
    figsize : tuple, default=(10, 5)
        Size of the matplotlib figure.
    random_state : int, default=42
        Random state for PCA reproducibility.
    """

    # Fit PCA with all components
    pca_full = PCA(n_components=len(features), random_state=random_state)
    pca_full.fit(data[features])

    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Individual variance
    ax.bar(
        range(1, len(features) + 1),
        explained_variance,
        alpha=0.6,
        label='Individual variance'
    )

    # Cumulative variance
    ax.plot(
        range(1, len(features) + 1),
        cumulative_variance,
        marker='o',
        linewidth=2,
        label='Cumulative variance'
    )

    # Threshold line
    ax.axhline(
        y=threshold,
        linestyle='--',
        linewidth=1.5,
        label=f'{int(threshold*100)}% threshold'
    )

    # Highlight chosen components
    chosen_var = cumulative_variance[chosen_components - 1]
    ax.axvline(
        x=chosen_components,
        linestyle=':',
        linewidth=2,
        label=f'Chosen ({chosen_components} PCs)'
    )
    ax.scatter(
        chosen_components,
        chosen_var,
        s=180,
        marker='*',
        zorder=10
    )
    ax.text(
        chosen_components + 0.15,
        chosen_var - 0.12,
        f'{chosen_var:.1%}\n({chosen_components} PCs)',
        fontsize=10,
        weight='bold'
    )

    # Formatting
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Variance Explained')
    ax.set_title('PCA – Variance Explained')
    ax.set_xticks(range(1, len(features) + 1))
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.0%}')
    )
    ax.grid(alpha=0.3)
    ax.legend(loc='center right')

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\n{'='*60}")
    print("Variance Explained by Each Component")
    print(f"{'='*60}")
    for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
        print(f"  PC{i+1}: {var:6.1%} | Cumulative: {cum_var:6.1%}")
    print(f"{'='*60}")
    print(
        f"\n{chosen_components} components capture "
        f"{chosen_var:.1%} of variance"
    )
    print(
        f"Trade-off: Simplicity vs. Information loss "
        f"({1 - chosen_var:.1%})"
    )



def plot_pca_clusters(df,label_col,pc1_col='pca_1',pc2_col='pca_2',var_explained=(0.0, 0.0),
                      title='2D PCA Projection of Clusters',figsize=(12, 8),point_size=30,alpha=0.6
                      ):
    """
    Plots a 2D PCA projection with cluster labels and centroids.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing PCA components and cluster labels.
    label_col : str
        Column name with cluster labels.
    pc1_col : str, default='pca_1'
        Column name for first principal component.
    pc2_col : str, default='pca_2'
        Column name for second principal component.
    var_explained : tuple, default=(0.0, 0.0)
        Explained variance for PC1 and PC2 (e.g. (0.42, 0.21)).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    point_size : int
        Size of scatter points.
    alpha : float
        Transparency of points.
    """

    fig, ax = plt.subplots(figsize=figsize)

    labels = sorted(df[label_col].unique())
    colors = cm.tab10(np.linspace(0, 1, max(len(labels), 10)))

    # Plot points by cluster
    for label in labels:
        cluster_data = df[df[label_col] == label]
        ax.scatter(
            cluster_data[pc1_col],
            cluster_data[pc2_col],
            c='none',
            edgecolors=[colors[label]],
            alpha=alpha,
            s=point_size,
            label=f'Cluster {label}'
        )

    # Plot centroids
    for label in labels:
        cluster_data = df[df[label_col] == label]
        centroid_x = cluster_data[pc1_col].mean()
        centroid_y = cluster_data[pc2_col].mean()

        ax.scatter(
            centroid_x,
            centroid_y,
            c='black',
            marker='X',
            s=200,
            edgecolors=colors[label],
            linewidths=1,
            zorder=10
        )

        ax.annotate(
            f'C{label}',
            (centroid_x + 0.15, centroid_y + 0.15),
            fontsize=14,
            weight='bold',
            ha='center',
            va='center',
            zorder=20
        )

    # Labels & formatting
    ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(loc='best', framealpha=1, title='Segments',
              fontsize=10, markerscale=1.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect(1)

    plt.tight_layout()
    plt.show()

def get_ss_variables(df):
    """Get the SS for each variable"""
    ss_vars = df.var() * (df.count() - 1)
    return ss_vars

def r2_variables(df, labels):
    """Get the R² for each variable"""
    sst_vars = get_ss_variables(df)
    ssw_vars = np.sum(df.groupby(labels).apply(get_ss_variables))
    return 1 - ssw_vars/sst_vars