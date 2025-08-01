#preprocessor.py
import pandas as pd
import re
from typing import Dict, List, Tuple
import os

def create_movie_lookup_table(dat_file_path: str, output_csv_path: str) -> Dict[str, int]:
    """
    Convert .dat movie file to numerical lookup table CSV with sequence encoding + padding
    
    Args:
        dat_file_path: Path to movies.dat file (format: movieId::title::genres)
        output_csv_path: Output CSV file path
        
    Returns:
        genre_to_id: Dictionary mapping genre names to numerical IDs
    """
    
    # Global genre vocabulary for dynamic encoding
    genre_to_id = {}
    next_genre_id = 1  # Start from 1, reserve 0 for padding
    
    # Lists to store processed data
    movie_data = []
    
    # Read and process the .dat file
    with open(dat_file_path, 'r', encoding='iso-8859-1') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            # Parse the line: movieId::title::genres
            parts = line.split('::')
            if len(parts) != 3:
                continue
                
            movie_id = int(parts[0])
            title = parts[1]
            genres_str = parts[2]
            
            # Extract year from title using regex
            year_match = re.search(r'\((\d{4})\)', title)
            year = int(year_match.group(1)) if year_match else 0
            
            # Clean title (remove year part)
            cleaned_title = re.sub(r'\s*\(\d{4}\)\s*$', '', title).strip()
            
            # Process genres
            genres = genres_str.split('|')
            
            # Encode genres dynamically
            genre_ids = []
            for genre in genres:
                if genre not in genre_to_id:
                    # Assign new ID to unseen genre
                    genre_to_id[genre] = next_genre_id
                    next_genre_id += 1
                genre_ids.append(genre_to_id[genre])
            
            # Apply sequence encoding with fixed length 3 and padding
            encoded_genres = encode_sequence_with_padding(genre_ids, max_length=3)
            
            # Create row data with tensor-ready format
            row_data = {
                'movie_id': movie_id,
                'year': year,
                'genres': str(encoded_genres)  # Convert to string format [1,2,3]
            }
            
            movie_data.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(movie_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # Save processed data
    df.to_csv(output_csv_path, index=False)
    
    # Save genre vocabulary for future reference
    vocab_path = output_csv_path.replace('.csv', '_genre_vocab.csv')
    vocab_df = pd.DataFrame([
        {'genre': genre, 'genre_id': genre_id} 
        for genre, genre_id in genre_to_id.items()
    ])
    vocab_df.to_csv(vocab_path, index=False)
    
    print(f"Processed {len(movie_data)} movies")
    print(f"Found {len(genre_to_id)} unique genres")
    print(f"Saved lookup table to: {output_csv_path}")
    print(f"Saved genre vocabulary to: {vocab_path}")
    
    return genre_to_id


def encode_sequence_with_padding(sequence: List[int], max_length: int = 3) -> List[int]:
    """
    Encode sequence with fixed length padding
    
    Args:
        sequence: List of integers to encode
        max_length: Fixed length for output sequence
        
    Returns:
        List of integers with fixed length (padded with 0 or truncated)
    """
    # Truncate if longer than max_length
    if len(sequence) > max_length:
        encoded = sequence[:max_length]
    else:
        # Pad with 0s if shorter than max_length  
        encoded = sequence + [0] * (max_length - len(sequence))
    
    return encoded


def load_processed_movie_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load processed movie data and convert genres back to tensor format
    
    Args:
        csv_path: Path to processed movies CSV
        
    Returns:
        Tuple of (DataFrame with tensor-ready genres, genre_vocabulary)
    """
    df = pd.read_csv(csv_path)
    
    # Convert string representation back to lists for torch tensor
    import ast
    df['genres'] = df['genres'].apply(ast.literal_eval)
    
    # Load vocabulary
    vocab_path = csv_path.replace('.csv', '_genre_vocab.csv')
    genre_vocab = load_genre_vocabulary(vocab_path)
    
    return df, genre_vocab


def convert_to_torch_tensor(df: pd.DataFrame) -> 'torch.Tensor':
    """
    Convert genres column to PyTorch tensor
    
    Args:
        df: DataFrame with genres column containing lists
        
    Returns:
        PyTorch tensor of shape (num_movies, 3)
    """
    import torch
    
    # Convert list of lists to tensor
    genres_tensor = torch.tensor(df['genres'].tolist(), dtype=torch.long)
    return genres_tensor
    """
    Load previously saved genre vocabulary
    
    Args:
        vocab_path: Path to genre vocabulary CSV file
        
    Returns:
        Dictionary mapping genre names to IDs
    """
    if not os.path.exists(vocab_path):
        return {}
        
    vocab_df = pd.read_csv(vocab_path)
    return dict(zip(vocab_df['genre'], vocab_df['genre_id']))

def load_genre_vocabulary(vocab_path: str) -> Dict[str, int]:
    """
    Load previously saved genre vocabulary
    
    Args:
        vocab_path: Path to genre vocabulary CSV file
        
    Returns:
        Dictionary mapping genre names to IDs
    """
    if not os.path.exists(vocab_path):
        return {}
        
    vocab_df = pd.read_csv(vocab_path)
    return dict(zip(vocab_df['genre'], vocab_df['genre_id']))

# Example usage
if __name__ == "__main__":
    # Process the MovieLens data
    dat_file = "../generative-recommenders-pl/tmp/ml-1m/movies.dat"
    output_csv = "../generative-recommenders-pl/tmp/processed/ml-1m/movies_encoded.csv"
    
    genre_vocab = create_movie_lookup_table(dat_file, output_csv)
    
    # Load and display sample results
    df = pd.read_csv(output_csv)
    print("\nSample processed data:")
    print(df.head(5))
    
    # Demo torch tensor conversion
    df_loaded, vocab = load_processed_movie_data(output_csv)
    print(f"\nLoaded data with tensor-ready format:")
    print(df_loaded.head(3))
    
    # Convert to torch tensor
    try:
        import torch
        genres_tensor = convert_to_torch_tensor(df_loaded)
        print(f"\nTorch tensor shape: {genres_tensor.shape}")
        print(f"Sample tensor data:\n{genres_tensor[:3]}")
    except ImportError:
        print("PyTorch not available for tensor conversion demo")
    
    print(f"\nGenre vocabulary sample:")
    for genre, genre_id in list(genre_vocab.items())[:10]:
        print(f"{genre}: {genre_id}")
