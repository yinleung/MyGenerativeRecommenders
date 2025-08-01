#!/home/svu/e1101894/gr_codes/.venv/bin/python3
# -*- coding: utf-8 -*-
"""
Tencent AMGR Competition Evaluation Code
腾讯全模态序列生成式推荐竞赛评估代码

This script evaluates trained models using the custom features.py and reco_dataset.py modules
for HR@K and NDCG@K metrics on test data.
"""

import sys
import os
import pandas as pd
import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
import torch.multiprocessing
from datetime import datetime
from pathlib import Path
from functools import partial
import numpy as np

# Path setup for local imports
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import custom modules
from features import seq_features_from_row, SequentialFeatures
from reco_dataset import RecoDataset, FeatureLookupTable

print(f"Current directory: {current_dir}")
print(f"Added to path: {src_path}")

# Import original generative_recommenders_pl modules
from generative_recommenders_pl.models.retrieval import Retrieval
from generative_recommenders_pl.models.sequential_encoders.hstu import HSTU
from generative_recommenders_pl.models.embeddings.embeddings import LocalEmbeddingModule
from generative_recommenders_pl.models.preprocessors import LearnablePositionalEmbeddingInputFeaturesPreprocessor
from generative_recommenders_pl.models.postprocessors.postprocessors import L2NormEmbeddingPostprocessor
from generative_recommenders_pl.models.similarity.dot_product import DotProductSimilarity
from generative_recommenders_pl.models.negatives_samples.negative_sampler import LocalNegativesSampler
from generative_recommenders_pl.models.indexing.candidate_index import CandidateIndex
from generative_recommenders_pl.models.indexing.top_k import MIPSBruteForceTopK
from generative_recommenders_pl.models.losses.autoregressive_losses import SampledSoftmaxLoss
from generative_recommenders_pl.models.metrics.retrieval import RetrievalMetrics

torch.multiprocessing.set_sharing_strategy("file_system")


class TencentDataModule(L.LightningDataModule):
    """
    Evaluation DataModule for Tencent AMGR competition
    腾讯竞赛评估数据模块 - 专为测试评估设计
    """
    
    def __init__(
        self,
        # Core data paths
        main_data_file: str,
        item_features_path: str,
        
        # Dataset parameters
        max_sequence_length: int = 200,
        batch_size: int = 128,
        num_workers: int = 4,
        prefetch_factor: int = 4,
        
        # Additional features
        additional_columns: list = None,
        sequence_prefix: str = "sequence_",
        user_id_column: str = None,
        
        # Processing parameters  
        shift_id_by: int = 1,
        chronological: bool = True,
    ):
        super().__init__()
        
        self.main_data_file = main_data_file
        self.item_features_path = item_features_path
        
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
        self.additional_columns = additional_columns or []
        self.sequence_prefix = sequence_prefix
        self.user_id_column = user_id_column
        
        self.shift_id_by = shift_id_by
        self.chronological = chronological
        
        # Will be set during setup
        self.max_item_id = None
        self.all_item_ids = None
        
    def setup(self, stage: str = None):
        """Setup test dataset for evaluation"""
        
        # Test dataset - keep all items (ignore_last_n=0)
        self.test_dataset = RecoDataset(
            main_data_file=self.main_data_file,
            item_features_path=self.item_features_path,
            max_sequence_length=self.max_sequence_length,
            ignore_last_n=0,  # Keep full sequences for evaluation
            shift_id_by=self.shift_id_by,
            chronological=self.chronological,
            sample_ratio=1.0,  # Use full sequences
            additional_columns=self.additional_columns,
            sequence_prefix=self.sequence_prefix,
            user_id_column=self.user_id_column,
        )
        
        # Setup item information
        self._setup_item_info()
    
    def _setup_item_info(self):
        """Setup item information from feature lookup table"""
        if os.path.exists(self.item_features_path):
            item_df = pd.read_csv(self.item_features_path)
            # Auto-detect item ID column
            id_column = None
            for col in item_df.columns:
                if col.lower().endswith('_id') or col.lower().endswith('id'):
                    id_column = col
                    break
            else:
                id_column = item_df.columns[0]
            
            # Get all item IDs and add shift
            original_item_ids = item_df[id_column].tolist()
            self.all_item_ids = [item_id + self.shift_id_by for item_id in original_item_ids]
            self.max_item_id = max(self.all_item_ids) + 1
            
            print(f"Loaded {len(self.all_item_ids)} items from {self.item_features_path}")
            print(f"Max item ID: {self.max_item_id}")
        else:
            print(f"Warning: Item features file not found: {self.item_features_path}")
            self.all_item_ids = list(range(1, 10000))
            self.max_item_id = 10000
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch):
        """Convert batch format for compatibility"""
        if not batch:
            return {}
        
        converted_batch = {}
        
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            
            if key == 'item_ids':
                # For test data, we need both historical and target
                stacked = torch.stack(values, dim=0)  # [B, L]
                converted_batch['item_ids'] = stacked
                
                # Extract target from last valid position
                lengths = torch.tensor([sample['length'] for sample in batch])
                target_indices = (lengths - 1).clamp(min=0)
                target_ids = stacked.gather(1, target_indices.unsqueeze(1))
                converted_batch['target_ids'] = target_ids.squeeze(1)  # [B]
                
                # Historical sequence (all but last)
                converted_batch['historical_ids'] = stacked[:, :-1]  # [B, L-1]
                
            elif key == 'length':
                lengths = torch.tensor(values)
                converted_batch['length'] = lengths
                converted_batch['history_lengths'] = lengths - 1
                
            elif key == 'ratings':
                stacked = torch.stack(values, dim=0)
                converted_batch['ratings'] = stacked
                converted_batch['historical_ratings'] = stacked[:, :-1]
                
                # Extract target ratings
                lengths = torch.tensor([sample['length'] for sample in batch])
                target_indices = (lengths - 1).clamp(min=0)
                target_ratings = stacked.gather(1, target_indices.unsqueeze(1))
                converted_batch['target_ratings'] = target_ratings.squeeze(1)
                
            elif key == 'timestamps':
                stacked = torch.stack(values, dim=0)
                converted_batch['timestamps'] = stacked
                converted_batch['historical_timestamps'] = stacked[:, :-1]
                
                # Extract target timestamps
                lengths = torch.tensor([sample['length'] for sample in batch])
                target_indices = (lengths - 1).clamp(min=0)
                target_timestamps = stacked.gather(1, target_indices.unsqueeze(1))
                converted_batch['target_timestamps'] = target_timestamps.squeeze(1)
                
            else:
                converted_batch[key] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
        
        return converted_batch


class TencentRetrievalModel(Retrieval):
    """
    Extended Retrieval model for Tencent AMGR evaluation
    支持腾讯竞赛评估的检索模型
    """
    
    def __init__(self, *args, **kwargs):
        self.max_output_length = kwargs.pop('max_output_length', 10)
        super().__init__(*args, **kwargs)
    
    def test_step(self, batch, batch_idx):
        """Test step for evaluation metrics"""
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1
        )
        
        # Get embeddings
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)
        
        # Retrieve top-k predictions
        top_k_ids, top_k_scores = self.retrieve(seq_features)
        
        # Update metrics
        self.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
        
        # Return metrics for logging
        return {
            "test_top_k_ids": top_k_ids,
            "test_target_ids": target_ids,
            "test_scores": top_k_scores
        }
    
    def on_test_epoch_end(self):
        """Compute and log final test metrics"""
        metrics = self.metrics.compute()
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            self.log(f"test/{metric_name}", metric_value, sync_dist=True)
        
        # Reset metrics for next epoch
        self.metrics.reset()
        
        return metrics


def evaluate_tencent_amgr_model(
    # Model checkpoint
    checkpoint_path: str,
    
    # Data files
    main_data_file: str,
    item_features_path: str,
    
    # Output settings
    output_file: str = "tencent_amgr_predictions.csv",
    results_file: str = "tencent_amgr_results.txt",
    
    # Model parameters (should match training)
    max_sequence_length: int = 200,
    gr_output_length: int = 10,
    item_embedding_dim: int = 50,
    
    # Evaluation parameters  
    batch_size: int = 128,
    k_values: list = [10, 50, 100, 200],  # K values for HR@K and NDCG@K
    
    # Additional features
    additional_columns: list = None,
    sequence_prefix: str = "sequence_",
    user_id_column: str = None,
    
    # Hardware settings
    use_gpu: bool = True,
    devices: int = 1,
):
    """
    Evaluate Tencent AMGR model on test data
    评估腾讯AMGR模型在测试数据上的性能
    
    Args:
        checkpoint_path: Path to model checkpoint
        main_data_file: Path to test data CSV  
        item_features_path: Path to item features CSV
        output_file: Output predictions CSV file
        results_file: Output results text file
        max_sequence_length: Maximum sequence length (should match training)
        gr_output_length: Generative recommendation output length
        item_embedding_dim: Item embedding dimension
        batch_size: Evaluation batch size
        k_values: K values for evaluation metrics
        additional_columns: Additional user feature columns
        sequence_prefix: Prefix for sequence columns
        user_id_column: User ID column name
        use_gpu: Whether to use GPU
        devices: Number of devices to use
    
    Returns:
        Dictionary of evaluation metrics
    """
    
    print("🚀 Starting Tencent AMGR Model Evaluation")
    print("=" * 60)
    
    # Hardware configuration
    gpu_available = torch.cuda.is_available()
    use_gpu_actual = use_gpu and gpu_available
    
    if use_gpu and not gpu_available:
        print("⚠️ GPU requested but CUDA not available, falling back to CPU")
    
    accelerator = "gpu" if use_gpu_actual else "cpu"
    devices_actual = devices if use_gpu_actual else 1
    precision = "16-mixed" if use_gpu_actual else 32
    
    print(f"🖥️ Hardware Configuration:")
    print(f"  - CUDA Available: {gpu_available}")
    print(f"  - Using: {accelerator.upper()}")
    print(f"  - Devices: {devices_actual}")
    print(f"  - Precision: {precision}")
    if use_gpu_actual:
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"\n📂 Data Configuration:")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Test Data: {main_data_file}")
    print(f"  - Item Features: {item_features_path}")
    print(f"  - Additional Columns: {additional_columns}")
    
    # Create evaluation data module
    print("\n📊 Creating evaluation data module...")
    datamodule = TencentDataModule(
        main_data_file=main_data_file,
        item_features_path=item_features_path,
        max_sequence_length=max_sequence_length,
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=4,
        additional_columns=additional_columns,
        sequence_prefix=sequence_prefix,
        user_id_column=user_id_column,
        shift_id_by=1,  # Should match training
        chronological=True,
    )
    
    # Setup datamodule to get item information
    datamodule.setup("test")
    num_items = datamodule.max_item_id
    all_item_ids_tensor = torch.tensor(datamodule.all_item_ids, dtype=torch.long)
    
    print(f"  - Number of items: {num_items}")
    print(f"  - Test dataset size: {len(datamodule.test_dataset)}")
    
    # Create model components (should match training configuration)
    print("\n🔧 Creating model components...")
    
    # Embeddings
    embeddings = LocalEmbeddingModule(
        num_items=num_items,
        item_embedding_dim=item_embedding_dim
    )
    
    # Preprocessor
    preprocessor = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=0.2
    )
    
    # HSTU sequence encoder
    sequence_encoder = HSTU(
        max_sequence_len=max_sequence_length,
        max_output_len=gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
        num_blocks=2,
        num_heads=1,
        attention_dim=item_embedding_dim,
        linear_dim=item_embedding_dim,
        linear_dropout_rate=0.2,
        attn_dropout_rate=0.0,
        normalization="rel_bias",
        linear_config="uvqk",
        linear_activation="silu",
        concat_ua=False,
        enable_relative_attention_bias=True
    )
    
    # Postprocessor
    postprocessor = L2NormEmbeddingPostprocessor(
        embedding_dim=item_embedding_dim,
        eps=1e-6
    )
    
    # Similarity
    similarity = DotProductSimilarity()
    
    # Negative sampler
    negatives_sampler = LocalNegativesSampler(
        l2_norm=True,
        l2_norm_eps=1e-6,
        all_item_ids=datamodule.all_item_ids
    )
    
    # Candidate index
    candidate_index = CandidateIndex(
        k=max(k_values),  # Use maximum k for evaluation
        top_k_module=MIPSBruteForceTopK(),
        ids=all_item_ids_tensor
    )
    
    # Loss function (not used in evaluation but required for model)
    loss = SampledSoftmaxLoss(
        num_to_sample=128,
        softmax_temperature=0.05
    )
    
    # Metrics with requested k values
    metrics = RetrievalMetrics(
        k=max(k_values),
        at_k_list=k_values
    )
    
    # Optimizer configuration (not used in evaluation)
    optimizer = partial(torch.optim.AdamW, lr=0.001)
    scheduler = partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="max")
    configure_optimizer_params = {"monitor": "val/ndcg@100", "interval": "epoch", "frequency": 1}
    
    # Create model
    print("🤖 Creating Tencent Retrieval model...")
    model = TencentRetrievalModel(
        datamodule=datamodule,
        embeddings=embeddings,
        preprocessor=preprocessor,
        sequence_encoder=sequence_encoder,
        postprocessor=postprocessor,
        similarity=similarity,
        negatives_sampler=negatives_sampler,
        candidate_index=candidate_index,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        configure_optimizer_params=configure_optimizer_params,
        gr_output_length=gr_output_length,
        item_embedding_dim=item_embedding_dim,
        max_output_length=gr_output_length,
        compile_model=False
    )
    
    # Create trainer for evaluation
    print("⚡ Creating evaluation trainer...")
    trainer = L.Trainer(
        devices=devices_actual,
        accelerator=accelerator,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=False,
    )
    
    # Load checkpoint and evaluate
    print(f"\n🧪 Starting evaluation...")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    try:
        # Test the model
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=checkpoint_path,
            verbose=True
        )
        
        print("\n✅ Evaluation completed successfully!")
        
        # Extract metrics from results
        if test_results and len(test_results) > 0:
            metrics_dict = test_results[0]
            
            # Print results
            print("\n📊 Evaluation Results:")
            print("=" * 60)
            
            # HR@K metrics
            print("Hit Rate (HR@K):")
            for k in k_values:
                hr_key = f"test/hr@{k}"
                if hr_key in metrics_dict:
                    print(f"  HR@{k:3d}: {metrics_dict[hr_key]:.4f}")
            
            print("\nNormalized DCG (NDCG@K):")
            for k in k_values:
                ndcg_key = f"test/ndcg@{k}"
                if ndcg_key in metrics_dict:
                    print(f"  NDCG@{k:3d}: {metrics_dict[ndcg_key]:.4f}")
            
            # MRR
            if "test/mrr" in metrics_dict:
                print(f"\nMean Reciprocal Rank (MRR): {metrics_dict['test/mrr']:.4f}")
            
            # Save results to file
            if results_file:
                print(f"\n💾 Saving results to: {results_file}")
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write("Tencent AMGR Competition - Evaluation Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Model Checkpoint: {checkpoint_path}\n")
                    f.write(f"Test Data: {main_data_file}\n")
                    f.write(f"Item Features: {item_features_path}\n")
                    f.write(f"Test Dataset Size: {len(datamodule.test_dataset)}\n")
                    f.write(f"Number of Items: {num_items}\n")
                    f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("Hit Rate (HR@K):\n")
                    for k in k_values:
                        hr_key = f"test/hr@{k}"
                        if hr_key in metrics_dict:
                            f.write(f"  HR@{k:3d}: {metrics_dict[hr_key]:.4f}\n")
                    
                    f.write("\nNormalized DCG (NDCG@K):\n")
                    for k in k_values:
                        ndcg_key = f"test/ndcg@{k}"
                        if ndcg_key in metrics_dict:
                            f.write(f"  NDCG@{k:3d}: {metrics_dict[ndcg_key]:.4f}\n")
                    
                    if "test/mrr" in metrics_dict:
                        f.write(f"\nMean Reciprocal Rank (MRR): {metrics_dict['test/mrr']:.4f}\n")
                    
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("All Metrics:\n")
                    for key, value in metrics_dict.items():
                        if key.startswith("test/"):
                            f.write(f"  {key}: {value:.6f}\n")
            
            return metrics_dict
            
        else:
            print("⚠️ No test results returned")
            return {}
            
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


if __name__ == "__main__":
    # Example usage for Tencent AMGR evaluation
    evaluation_results = evaluate_tencent_amgr_model(
        # Model checkpoint
        checkpoint_path="./logs/tencent_amgr/2025-07-28_15-16-30/checkpoints/last.ckpt",
        #checkpoint_path="../generative-recommenders-pl/logs/train/runs/2025-07-04_16-23-11/checkpoints/last.ckpt",
        
        # Data files (can be same as training or different test set)
        main_data_file="../generative-recommenders-pl/tmp/ml-1m/sasrec_format_by_user_test.csv",
        item_features_path="../generative-recommenders-pl/tmp/processed/ml-1m/movies_encoded.csv",
        
        # Output settings
        output_file="tencent_amgr_predictions.csv",
        results_file="tencent_amgr_evaluation_results.txt",
        
        # Model parameters (should match training)
        max_sequence_length=200,
        gr_output_length=10,
        item_embedding_dim=50,
        
        # Evaluation parameters
        batch_size=128,
        k_values=[10, 50, 100, 200],  # Standard evaluation metrics
        
        # Additional features from main CSV file
        additional_columns=[
            'sex', 'age_group', 'occupation', 'zip_code'  # User features
            # For Tencent data, might be: 'user_city', 'ad_category', 'advertiser_id', etc.
        ],
        sequence_prefix="sequence_",
        user_id_column=None,  # Will auto-detect
        
        # Hardware settings
        use_gpu=True,
        devices=1,
    )
    
    print("\n🎉 Tencent AMGR evaluation completed successfully!")
    print("腾讯AMGR竞赛评估已成功完成！")
    
    # Print final summary
    if evaluation_results:
        print("\n📈 Final Summary:")
        print("-" * 40)
        for k in [10, 50, 100, 200]:
            hr_key = f"test/hr@{k}"
            ndcg_key = f"test/ndcg@{k}"
            if hr_key in evaluation_results and ndcg_key in evaluation_results:
                print(f"K={k:3d}: HR={evaluation_results[hr_key]:.4f}, NDCG={evaluation_results[ndcg_key]:.4f}")
        
        if "test/mrr" in evaluation_results:
            print(f"MRR: {evaluation_results['test/mrr']:.4f}")
