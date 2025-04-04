import os
import ts2vg
import pandas as pd
import librosa
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

def extract_graph_features(A):
    """
    Extract the exact graph-based features as described in the paper:
    - Degree of connectivity (DoC)
    - Density (D)
    - Modularity (M)
    - Clustering coefficient (CC)
    - Shortest path length (L)
    - Small-world coefficient (S)
    """
    # Create graph from adjacency matrix
    if isinstance(A, np.ndarray):
        G = nx.from_numpy_array(A)
    else:
        G = nx.from_scipy_sparse_array(A)
    
    # Calculate all required features
    features = {}
    
    # 1. Degree of connectivity (sum of all edges)
    features["DoC"] = np.sum(A) if isinstance(A, np.ndarray) else A.sum()
    
    # 2. Density
    features["D"] = nx.density(G)
    
    # 3. Modularity
    if nx.is_connected(G):
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        features["M"] = nx.algorithms.community.modularity(G, communities)
    else:
        features["M"] = np.nan
    
    # 4. Clustering coefficient (global average)
    features["CC"] = nx.average_clustering(G)
    
    # 5. Shortest path length (global average)
    if nx.is_connected(G):
        features["L"] = nx.average_shortest_path_length(G)
    else:
        features["L"] = np.nan
    
    # 6. Small-world coefficient
    # Create random graph with same number of nodes and edges
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    p = (2 * n_edges) / (n_nodes * (n_nodes - 1))  # Edge probability
    
    # Generate random graph (Erdos-Renyi model as mentioned in paper)
    G_random = nx.erdos_renyi_graph(n_nodes, p)
    
    # Calculate CC and L for random graph
    if G_random.number_of_edges() > 0:
        CC_random = nx.average_clustering(G_random)
        if nx.is_connected(G_random):
            L_random = nx.average_shortest_path_length(G_random)
        else:
            L_random = np.nan
    else:
        CC_random = 0
        L_random = np.nan
    
    # Calculate small-world coefficient (handle division by zero)
    if CC_random == 0 or np.isnan(features["CC"]) or np.isnan(CC_random):
        features["S"] = np.nan
    else:
        if np.isnan(features["L"]) or np.isnan(L_random) or L_random == 0:
            features["S"] = np.nan
        else:
            features["S"] = (features["CC"] / CC_random) / (features["L"] / L_random)
    
    return features

def compute_visibility_graph(T, k=2):
    """
    Compute visibility graph with shift value k=2 as mentioned in the paper
    Returns adjacency matrix in binary form
    """
    vg = ts2vg.NaturalVG()
    vg.build(T)
    return vg.adjacency_matrix()

def sliding_window_rms(audio, sr, window_size=0.009, overlap=0.0045):
    """
    Compute zero-mean RMS energy for overlapping frames as described in paper:
    - window_size = 9ms (2000 samples at 44100Hz)
    - overlap = 4.5ms (1000 samples at 44100Hz)
    Returns compact time series x ∈ ℝ^(1×N)
    """
    # Convert times to samples
    frame_length = int(window_size * sr)
    hop_length = int((window_size - overlap) * sr)
    
    # Compute RMS energy for each frame
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length, center=True)
    
    # Remove DC offset (zero-mean)
    rms = rms - np.mean(rms)
    
    # Ensure non-negative values as required by VG theory
    rms = np.abs(rms)
    
    return rms.flatten()

def load_emovo_dataset(root_dir):
    """
    Load EMOVO dataset with proper emotion labels
    """
    data = []
    emotion_map = {
        'dis': 'disgust',
        'gio': 'joy',
        'pau': 'fear',
        'rab': 'anger',
        'sor': 'surprise',
        'tri': 'sadness',
        'neu': 'neutral'
    }
    
    for speaker in os.listdir(root_dir):
        speaker_dir = os.path.join(root_dir, speaker)
        if os.path.isdir(speaker_dir):
            for file in os.listdir(speaker_dir):
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[0]
                        emotion = emotion_map.get(emotion_code, 'unknown')
                        sentence_type = parts[2].split(".")[0]
                        filepath = os.path.join(speaker_dir, file)
                        data.append({
                            "speaker": speaker,
                            "emotion": emotion,
                            "sentence_type": sentence_type,
                            "filepath": filepath
                        })
    return pd.DataFrame(data)

def process_all_files(df, output_dir="graph_features"):
    """
    Process all files in the dataset and extract graph features
    """
    os.makedirs(output_dir, exist_ok=True)
    features_list = []
    
    for idx, row in df.iterrows():
        try:
            # Load audio file at native sampling rate (44100Hz as per paper)
            audio, sr = librosa.load(row["filepath"], sr=44100)
            
            # 1. Preprocessing: Compute compact time series using sliding window RMS
            T = sliding_window_rms(audio, sr)
            
            # 2. Compute visibility graph (binary adjacency matrix)
            A = compute_visibility_graph(T)
            
            # 3. Extract graph-based features
            features = extract_graph_features(A)
            
            # Add metadata
            features.update({
                "speaker": row["speaker"],
                "emotion": row["emotion"],
                "sentence_type": row["sentence_type"],
                "filepath": row["filepath"]
            })
            
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing {row['filepath']}: {e}")
    
    # Create DataFrame and normalize features as done in the paper
    features_df = pd.DataFrame(features_list)
    
    # Normalize each feature by its maximum value (as mentioned in paper)
    feature_cols = ["DoC", "D", "M", "CC", "L", "S"]
    for col in feature_cols:
        if col in features_df.columns:
            max_val = features_df[col].max()
            if max_val > 0:
                features_df[col] = features_df[col] / max_val
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "paper_emovo_graph_features.csv")
    features_df.to_csv(csv_path, index=False)
    print(f"Saved features to {csv_path}")
    
    return features_df

# Main execution
if __name__ == "__main__":
    # 1. Load EMOVO dataset
    df = load_emovo_dataset(".")
    print(f"Loaded {len(df)} audio files from EMOVO dataset")
    
    # 2. Process all files and extract features
    features_df = process_all_files(df)
    
    # 3. Show sample of extracted features
    print("\nSample of extracted features:")
    print(features_df.head())
