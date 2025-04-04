import os
import ts2vg
import pandas as pd
import librosa
import numpy as np
import networkx as nx

def extract_graph_features(A):
    """Extract graph features including average of adjacency matrix"""
    if isinstance(A, np.ndarray):
        G = nx.from_numpy_array(A)
        avg_A = np.mean(A)  # Calculate average of adjacency matrix
    else:
        G = nx.from_scipy_sparse_array(A)
        avg_A = A.mean()    # For sparse matrices
        
    features = {
        "DoC": np.sum(A) if isinstance(A, np.ndarray) else A.sum(),  # Degree of connectivity
        "Density": nx.density(G),
        "CC": nx.average_clustering(G),
        "L": nx.average_shortest_path_length(G) if nx.is_connected(G) else np.nan,
        "M": compute_modularity(G),
        "Avg_A": avg_A  
    }
    return features

def compute_modularity(G):
    """Compute modularity with community detection."""
    if nx.is_connected(G) and G.number_of_edges() > 0:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        return nx.algorithms.community.modularity(G, communities)
    return np.nan

def compute_visibility_graph(T):
    """Convert time series to visibility graph (binary adjacency)."""
    vg = ts2vg.NaturalVG()
    vg.build(T)
    return vg.adjacency_matrix()

def sliding_window_std(audio, window_size=1500, overlap=900):
    """Sliding window with standard deviation."""
    std_values = []
    hop = window_size - overlap
    for i in range(0, len(audio) - window_size + 1, hop):
        std_values.append(np.std(audio[i:i + window_size]))
    return np.array(std_values)

def load_emovo_dataset(root_dir):
    """Load EMOVO with emotion mapping."""
    emotion_map = {
        'dis': 'disgust', 'gio': 'joy', 'pau': 'fear',
        'rab': 'anger', 'sor': 'surprise', 'tri': 'sadness', 'neu': 'neutral'
    }
    
    data = []
    for speaker in os.listdir(root_dir):
        speaker_dir = os.path.join(root_dir, speaker)
        if os.path.isdir(speaker_dir):
            for file in os.listdir(speaker_dir):
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[0]
                        data.append({
                            "speaker": speaker,
                            "emotion": emotion_map.get(emotion_code, "unknown"),
                            "filepath": os.path.join(speaker_dir, file)
                        })
    return pd.DataFrame(data)

def process_all_files(df, window_size=1500, overlap=900, output_dir="features"):
    """Process all files with the given window/overlap."""
    os.makedirs(output_dir, exist_ok=True)
    features_list = []
    print("Processing: ws:: ")
    print(window_size)

    for idx, row in df.iterrows():
        try:
            print(idx)
            audio, sr = librosa.load(row["filepath"], sr=None)
            T = sliding_window_std(audio, window_size, overlap)
            A = compute_visibility_graph(T)
            features = extract_graph_features(A)
            
            features.update({
                "speaker": row["speaker"],
                "emotion": row["emotion"],
                "filepath": row["filepath"],
                "window_size": window_size
            })
            features_list.append(features)
            print("Extracted features: ")
            print(features)
        except Exception as e:
            print(f"Error processing {row['filepath']}: {str(e)}")
    
    csv_path = os.path.join(output_dir, f"project_emovo_features_{window_size}.csv")
    pd.DataFrame(features_list).to_csv(csv_path, index=False)
    print(f"Saved {len(features_list)} records to {csv_path}")

# Main execution
if __name__ == "__main__":
    df = load_emovo_dataset(".")
    print(f"Loaded EMOVO dataset with {len(df)} files")
    
    # Test multiple window sizes
    process_all_files(df, window_size=1000, overlap=500)
    process_all_files(df, window_size=2000, overlap=1000)
    process_all_files(df, window_size=3000, overlap=1000)
