import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
import uuid

def parse_json_field(field):
    """Safely parse a JSON field, return empty dict if parsing fails."""
    try:
        return json.loads(field) if isinstance(field, str) else {}
    except:
        return {}

def get_text_for_embedding(row):
    """Create a text representation of a product for embedding calculation."""
    # Safely extract fields with sensible defaults
    title = str(row.get("title", "")) if "title" in row else ""
    brand = str(row.get("brand", "")) if "brand" in row else ""
    
    # Handle attributes field that might be missing or in various formats
    attributes = {}
    if "attributes" in row:
        if isinstance(row["attributes"], dict):
            attributes = row["attributes"]
        else:
            attributes = parse_json_field(row["attributes"])
    
    attr_string = " ".join([f"{k}:{v}" for k, v in attributes.items()])
    return f"{title} {brand} {attr_string}".strip()

def deduplicate_products(df, threshold=0.85, model_name="all-MiniLM-L6-v2", auto_pick="Lowest Price"):
    """
    Deduplicate products using semantic similarity.
    
    Args:
        df: Input DataFrame with product data
        threshold: Similarity threshold for considering products as duplicates
        model_name: Sentence transformer model name
        auto_pick: Strategy for picking representative product ("Lowest Price" or "First Seen")
        
    Returns:
        Tuple of (deduplicated DataFrame, pairs DataFrame with similarity info)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")
    
    # Create internal IDs for processing while preserving original IDs
    has_original_product_id = "product_id" in df.columns
    
    if has_original_product_id:
        # Store original product IDs
        df["original_product_id"] = df["product_id"]
    
    # Create internal UUIDs for processing
    df["internal_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    
    # Create text for embedding
    df["text_for_embedding"] = df.apply(get_text_for_embedding, axis=1)
    
    # Generate embeddings
    texts = df["text_for_embedding"].tolist()
    internal_ids = df["internal_id"].tolist()
    
    try:
        embeddings = model.encode(texts, convert_to_tensor=True)
        embeddings = [e.cpu().numpy() for e in embeddings]
    except:
        # Fallback if convert_to_tensor doesn't work
        embeddings = model.encode(texts)
    
    # Find similar pairs
    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            try:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim >= threshold:
                    pairs.append({
                        "internal_id_1": internal_ids[i],
                        "internal_id_2": internal_ids[j],
                        "similarity": sim
                    })
            except Exception as e:
                continue
    
    # Build graph and find connected components
    G = nx.Graph()
    for row in pairs:
        G.add_edge(row["internal_id_1"], row["internal_id_2"])
    
    df_lookup = df.set_index("internal_id")
    to_keep = set()
    
    # For each group of duplicates, select a representative product
    for component in nx.connected_components(G):
        group_ids = list(component)
        group_df = df[df["internal_id"].isin(group_ids)]
        
        if auto_pick == "Lowest Price" and "price" in df.columns:
            # Check if price column contains valid numeric values
            if pd.api.types.is_numeric_dtype(group_df["price"]) or all(pd.to_numeric(group_df["price"], errors='coerce').notna()):
                if not pd.api.types.is_numeric_dtype(group_df["price"]):
                    group_df["price"] = pd.to_numeric(group_df["price"], errors='coerce')
                
                # Select product with lowest price
                best_product = group_df.loc[group_df["price"].idxmin()]
                best_pid = best_product["internal_id"]
            else:
                # If price is not numeric, fall back to first product
                best_pid = group_ids[0]
        else:
            # First seen strategy
            best_pid = group_ids[0]
            
        to_keep.add(best_pid)
    
    # Add all products that weren't part of any duplicate group
    all_ids = set(df["internal_id"])
    to_keep.update(all_ids - set(G.nodes))
    
    # Prepare the final deduplicated DataFrame
    df_result = df[df["internal_id"].isin(to_keep)].copy()
    
    # Handle product IDs in the final result
    if has_original_product_id:
        # Restore original product IDs
        df_result["product_id"] = df_result["original_product_id"]
        df_result.drop(columns=["original_product_id", "internal_id", "text_for_embedding"], errors="ignore", inplace=True)
    else:
        # If there were no product IDs, use the internal IDs as product IDs
        df_result["product_id"] = df_result["internal_id"]
        df_result.drop(columns=["internal_id", "text_for_embedding"], errors="ignore", inplace=True)
    
    # Ensure the dataframe index is reset before returning
    df_result = df_result.reset_index(drop=True)
    
    # Convert pairs DataFrame to use original product IDs if available
    if pairs and has_original_product_id:
        id_mapping = dict(zip(df["internal_id"], df["original_product_id"]))
        for pair in pairs:
            pair["product_id_1"] = id_mapping.get(pair["internal_id_1"])
            pair["product_id_2"] = id_mapping.get(pair["internal_id_2"])
            pair.pop("internal_id_1")
            pair.pop("internal_id_2")
    elif pairs:
        # If no original product IDs, keep using internal IDs but rename columns
        for pair in pairs:
            pair["product_id_1"] = pair.pop("internal_id_1")
            pair["product_id_2"] = pair.pop("internal_id_2")
    
    return df_result, pd.DataFrame(pairs if pairs else [])