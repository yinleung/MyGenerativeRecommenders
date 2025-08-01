from typing import Dict, NamedTuple, Optional, Tuple

import torch


class SequentialFeatures(NamedTuple):
    # (B,) x int64. Requires past_lengths[i] > 0 \forall i.
    past_lengths: torch.Tensor
    # (B, N,) x int64. 0 denotes valid ids.
    past_ids: torch.Tensor
    # (B, N, D) x float.
    past_embeddings: Optional[torch.Tensor]
    # Implementation-specific payloads.
    # e.g., past timestamps, past event_types (e.g., clicks, likes), etc.
    past_payloads: Dict[str, torch.Tensor]


def seq_features_from_row(
    row,
    device: torch.device,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    """
    Convert batch from new reco_dataset to SequentialFeatures format
    
    This function now handles TWO scenarios:
    1. Test data: Has separate _target_* fields (for compatibility testing)
    2. Real data: Target is last item in sequence (real dataset behavior)
    
    Args:
        row: Batch dict from new RecoDataset or test data
        device: Target device
        max_output_length: Maximum output sequence length
        
    Returns:
        Tuple of (SequentialFeatures, target_ids, target_ratings)
    """
    
    # Map new dataset keys to expected keys
    historical_lengths = row["length"].to(device)  # [B]
    historical_ids = row["item_ids"].to(device)  # [B, N]
    
    # Handle optional sequence features
    if "ratings" in row:
        historical_ratings = row["ratings"].to(device)  # [B, N]
    else:
        historical_ratings = torch.ones_like(historical_ids, dtype=torch.float32, device=device)
    
    if "timestamps" in row:
        historical_timestamps = row["timestamps"].to(device)  # [B, N]
    else:
        B, N = historical_ids.shape
        historical_timestamps = torch.arange(N, device=device).float().unsqueeze(0).expand(B, N)
    
    # Extract target - handle both test data and real data scenarios
    if "_target_ids" in row:
        # Test scenario: explicit target fields
        target_ids = row["_target_ids"].to(device).unsqueeze(1)  # [B, 1]
        target_ratings = row["_target_ratings"].to(device).unsqueeze(1)  # [B, 1]
        target_timestamps = row["_target_timestamps"].to(device).unsqueeze(1)  # [B, 1]
        
        # Historical sequences are already correct, no modification needed
        
    else:
        # Real data scenario: target is last item in sequence
        batch_size = historical_lengths.size(0)
        target_indices = (historical_lengths - 1).clamp(min=0)  # Last valid position
        
        # Gather target values
        target_ids = historical_ids.gather(1, target_indices.unsqueeze(1))  # [B, 1]
        target_ratings = historical_ratings.gather(1, target_indices.unsqueeze(1))  # [B, 1]  
        target_timestamps = historical_timestamps.gather(1, target_indices.unsqueeze(1))  # [B, 1]
        
        # Remove target from historical sequences (for training)
        # Create mask to zero out the target position
        mask = torch.arange(historical_ids.size(1), device=device).unsqueeze(0) < target_indices.unsqueeze(1)
        historical_ids = historical_ids * mask.long()
        historical_ratings = historical_ratings * mask.float()
        historical_timestamps = historical_timestamps * mask.float()
        
        # Adjust lengths (subtract 1 since we removed target)
        historical_lengths = target_indices  # Length without target

    # Handle max_output_length padding if needed
    if max_output_length > 0:
        B = historical_lengths.size(0)
        
        # Pad sequences to accommodate output generation
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps = torch.cat(
            [
                historical_timestamps,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_timestamps.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        
        # Set target timestamp at the appropriate position
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )

    # Prepare additional payloads (auxiliary features)
    exclude_keys = {
        "user_id", "length", "item_ids", "ratings", "timestamps",
        "_target_ids", "_target_ratings", "_target_timestamps"  # Exclude test metadata
    }
    
    additional_payloads = {}
    for key, value in row.items():
        if key not in exclude_keys:
            # Move tensor to device
            if isinstance(value, torch.Tensor):
                additional_payloads[key] = value.to(device)
            else:
                additional_payloads[key] = torch.tensor(value, device=device)

    # Create SequentialFeatures
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,  # Will be filled by embedding layer
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
            **additional_payloads  # Include additional columns like sex, age_group, etc.
        },
    )
    
    return features, target_ids, target_ratings


# Note: Original repo uses same function for all scenarios (train/val/test/predict)
# For inference/predict, target is extracted but ignored (seq_features, _, _ = seq_features_from_row())


# Utility function to convert from old format to new format (for compatibility)
def convert_old_to_new_format(old_row: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert old dataset format to new dataset format
    
    Maps:
    - 'history_lengths' -> 'length'
    - 'historical_ids' -> 'item_ids'  
    - 'historical_ratings' -> 'ratings'
    - 'historical_timestamps' -> 'timestamps'
    """
    new_row = {}
    
    # Map key names
    key_mapping = {
        'history_lengths': 'length',
        'historical_ids': 'item_ids',
        'historical_ratings': 'ratings', 
        'historical_timestamps': 'timestamps'
    }
    
    for old_key, new_key in key_mapping.items():
        if old_key in old_row:
            new_row[new_key] = old_row[old_key]
    
    # Copy other keys as-is
    for key, value in old_row.items():
        if key not in key_mapping:
            new_row[key] = value
            
    return new_row


# Utility function to convert from new format to old format  
def convert_new_to_old_format(new_row: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert new dataset format to old dataset format
    
    Maps:
    - 'length' -> 'history_lengths'
    - 'item_ids' -> 'historical_ids'
    - 'ratings' -> 'historical_ratings'
    - 'timestamps' -> 'historical_timestamps'
    """
    old_row = {}
    
    # Map key names (reverse mapping)
    key_mapping = {
        'length': 'history_lengths',
        'item_ids': 'historical_ids', 
        'ratings': 'historical_ratings',
        'timestamps': 'historical_timestamps'
    }
    
    for new_key, old_key in key_mapping.items():
        if new_key in new_row:
            old_row[old_key] = new_row[new_key]
    
    # Copy other keys as-is
    for key, value in new_row.items():
        if key not in key_mapping:
            old_row[key] = value
            
    return old_row



def seq_features_from_row_debug(
    row,
    device: torch.device,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    """
    Debug version of seq_features_from_row with detailed logging
    """
    print("🔍 DEBUG: seq_features_from_row called")
    print(f"📦 Batch keys: {list(row.keys())}")
    print(f"📏 Batch shapes:")
    for key, value in row.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: {value.shape} ({value.dtype})")
        elif hasattr(value, '__len__'):
            print(f"   {key}: len={len(value)} (type={type(value)})")
        else:
            print(f"   {key}: {value} (type={type(value)})")
    
    print(f"🎯 Looking for required keys...")
    
    # Check for length keys
    length_keys = ['length', 'history_lengths', 'past_lengths']
    found_length_key = None
    for key in length_keys:
        if key in row:
            found_length_key = key
            print(f"✅ Found length key: {key}")
            break
    
    if not found_length_key:
        print(f"❌ ERROR: No length key found! Available keys: {list(row.keys())}")
        raise KeyError(f"Expected one of {length_keys}, got keys: {list(row.keys())}")
    
    # Check for item_ids keys
    item_keys = ['item_ids', 'historical_ids', 'past_ids']
    found_item_key = None
    for key in item_keys:
        if key in row:
            found_item_key = key
            print(f"✅ Found item key: {key}")
            break
    
    if not found_item_key:
        print(f"❌ ERROR: No item key found! Available keys: {list(row.keys())}")
        raise KeyError(f"Expected one of {item_keys}, got keys: {list(row.keys())}")
    
    # Get the data
    print(f"📊 Extracting data...")
    historical_lengths = row[found_length_key].to(device)
    historical_ids = row[found_item_key].to(device)
    
    print(f"   historical_lengths: {historical_lengths.shape} = {historical_lengths[:3]}")
    print(f"   historical_ids: {historical_ids.shape}")
    print(f"   historical_ids[0][:10]: {historical_ids[0][:10]}")
    
    # Check for ratings
    rating_keys = ['ratings', 'historical_ratings', 'past_ratings']
    found_rating_key = None
    for key in rating_keys:
        if key in row:
            found_rating_key = key
            print(f"✅ Found rating key: {key}")
            break
    
    if found_rating_key:
        historical_ratings = row[found_rating_key].to(device)
        print(f"   historical_ratings: {historical_ratings.shape}")
    else:
        print(f"⚠️ No ratings found, using ones")
        historical_ratings = torch.ones_like(historical_ids, dtype=torch.float32, device=device)
    
    # Check for timestamps
    timestamp_keys = ['timestamps', 'historical_timestamps', 'past_timestamps']
    found_timestamp_key = None
    for key in timestamp_keys:
        if key in row:
            found_timestamp_key = key
            print(f"✅ Found timestamp key: {key}")
            break
    
    if found_timestamp_key:
        historical_timestamps = row[found_timestamp_key].to(device)
        print(f"   historical_timestamps: {historical_timestamps.shape}")
    else:
        print(f"⚠️ No timestamps found, using sequential")
        B, N = historical_ids.shape
        historical_timestamps = torch.arange(N, device=device).float().unsqueeze(0).expand(B, N)
    
    # Check for explicit targets
    target_keys = ['_target_ids', 'target_ids']
    has_explicit_targets = any(key in row for key in target_keys)
    print(f"🎯 Has explicit targets: {has_explicit_targets}")
    
    if has_explicit_targets:
        print(f"📤 Using explicit targets...")
        # Find target keys
        target_id_key = None
        for key in ['_target_ids', 'target_ids']:
            if key in row:
                target_id_key = key
                break
                
        target_rating_key = None
        for key in ['_target_ratings', 'target_ratings']:
            if key in row:
                target_rating_key = key
                break
                
        target_timestamp_key = None
        for key in ['_target_timestamps', 'target_timestamps']:
            if key in row:
                target_timestamp_key = key
                break
        
        if target_id_key:
            target_ids = row[target_id_key].to(device)
            if target_ids.dim() == 1:
                target_ids = target_ids.unsqueeze(1)
            print(f"   target_ids: {target_ids.shape} = {target_ids[:3].flatten()}")
        else:
            raise KeyError("Found explicit target indicators but no target_ids")
            
        if target_rating_key:
            target_ratings = row[target_rating_key].to(device)
            if target_ratings.dim() == 1:
                target_ratings = target_ratings.unsqueeze(1)
            print(f"   target_ratings: {target_ratings.shape}")
        else:
            target_ratings = torch.ones_like(target_ids, dtype=torch.float32)
            print(f"   target_ratings: using ones {target_ratings.shape}")
        
        if target_timestamp_key:
            target_timestamps = row[target_timestamp_key].to(device)
            if target_timestamps.dim() == 1:
                target_timestamps = target_timestamps.unsqueeze(1)
            print(f"   target_timestamps: {target_timestamps.shape}")
        else:
            target_timestamps = torch.zeros_like(target_ids, dtype=torch.float32)
            print(f"   target_timestamps: using zeros {target_timestamps.shape}")
    
    else:
        print(f"📤 Extracting targets from sequences...")
        batch_size = historical_lengths.size(0)
        target_indices = (historical_lengths - 1).clamp(min=0)
        print(f"   target_indices: {target_indices}")
        
        # Gather target values
        target_ids = historical_ids.gather(1, target_indices.unsqueeze(1))
        target_ratings = historical_ratings.gather(1, target_indices.unsqueeze(1))
        target_timestamps = historical_timestamps.gather(1, target_indices.unsqueeze(1))
        
        print(f"   extracted target_ids: {target_ids.shape} = {target_ids[:3].flatten()}")
        print(f"   extracted target_ratings: {target_ratings.shape}")
        
        # Remove target from historical sequences
        mask = torch.arange(historical_ids.size(1), device=device).unsqueeze(0) < target_indices.unsqueeze(1)
        historical_ids = historical_ids * mask.long()
        historical_ratings = historical_ratings * mask.float()
        historical_timestamps = historical_timestamps * mask.float()
        
        # Adjust lengths
        historical_lengths = target_indices
        print(f"   adjusted historical_lengths: {historical_lengths}")
    
    # Handle max_output_length padding
    if max_output_length > 0:
        print(f"🔄 Adding padding for max_output_length={max_output_length}")
        B = historical_lengths.size(0)
        
        # Pad sequences
        historical_ids = torch.cat([
            historical_ids,
            torch.zeros((B, max_output_length), dtype=historical_ids.dtype, device=device)
        ], dim=1)
        historical_ratings = torch.cat([
            historical_ratings,
            torch.zeros((B, max_output_length), dtype=historical_ratings.dtype, device=device)
        ], dim=1)
        historical_timestamps = torch.cat([
            historical_timestamps,
            torch.zeros((B, max_output_length), dtype=historical_timestamps.dtype, device=device)
        ], dim=1)
        
        # Set target timestamp
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )
        
        print(f"   padded shapes: ids={historical_ids.shape}, ratings={historical_ratings.shape}")
    
    # Process additional payloads
    exclude_keys = {
        "user_id", "length", "item_ids", "ratings", "timestamps",
        "history_lengths", "historical_ids", "historical_ratings", "historical_timestamps",
        "past_lengths", "past_ids", "past_ratings", "past_timestamps",
        "_target_ids", "_target_ratings", "_target_timestamps",
        "target_ids", "target_ratings", "target_timestamps"
    }
    
    additional_payloads = {}
    print(f"🔧 Processing additional payloads...")
    for key, value in row.items():
        if key not in exclude_keys:
            if isinstance(value, torch.Tensor):
                additional_payloads[key] = value.to(device)
                print(f"   added {key}: {value.shape}")
            else:
                additional_payloads[key] = torch.tensor(value, device=device)
                print(f"   added {key}: {torch.tensor(value).shape}")
    
    # Create SequentialFeatures
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
            **additional_payloads
        },
    )
    
    print(f"✅ DEBUG: seq_features_from_row completed successfully")
    print(f"   past_lengths: {features.past_lengths.shape}")
    print(f"   past_ids: {features.past_ids.shape}") 
    print(f"   target_ids: {target_ids.shape}")
    print(f"   payloads: {list(features.past_payloads.keys())}")
    
    return features, target_ids, target_ratings


# Also add this debug wrapper to your training code
def debug_batch_keys(batch, step_name="Unknown"):
    """Quick debug function to inspect batch structure"""
    print(f"\n🔍 DEBUG {step_name} - Batch inspection:")
    print(f"📦 Keys: {list(batch.keys())}")
    for key, value in batch.items():
        if hasattr(value, 'shape'):
            print(f"   {key}: {value.shape} ({value.dtype})")
        else:
            print(f"   {key}: {type(value)}")
    print("="*50)


# Replace the original function temporarily in your training code:
# Add this to your training script right after importing features:

# from features import seq_features_from_row_debug
# 
# # In your TencentRetrievalModel class, replace:
# # seq_features, target_ids, target_ratings = seq_features_from_row(...)
# # with:
# # seq_features, target_ids, target_ratings = seq_features_from_row_debug(...)