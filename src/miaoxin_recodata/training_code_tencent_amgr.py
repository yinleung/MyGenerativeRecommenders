#!/home/svu/e1101894/gr_codes/.venv/bin/python3
# -*- coding: utf-8 -*-
"""
Tencent AMGR Competition Training Code
腾讯全模态序列生成式推荐竞赛训练代码

This script integrates the new features.py and reco_dataset.py modules
for the All-Modality Generative Recommendation competition.
"""

import sys
import os

# Path setup for local imports
current_dir = os.getcwd()
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# Import new local modules
from features import seq_features_from_row, SequentialFeatures
from features import seq_features_from_row_debug, debug_batch_keys
from reco_dataset import RecoDataset, FeatureLookupTable

print(f"Current directory: {current_dir}")
print(f"Added to path: {src_path}")

import torch
torch.set_float32_matmul_precision('medium')
import lightning as L
import torch.multiprocessing
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from functools import partial

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

# Callbacks and loggers
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichModelSummary, RichProgressBar
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

torch.multiprocessing.set_sharing_strategy("file_system")


class TencentDataModule(L.LightningDataModule):
    """
    Simplified DataModule for Tencent AMGR competition using new RecoDataset
    简化的腾讯竞赛数据模块，只需要main_data_file和item_features_path
    """
    
    def __init__(
        self,
        # Core data paths - these are all that's needed!
        main_data_file: str,  # e.g., "sasrec_format_by_user_train.csv"
        item_features_path: str,  # e.g., "movies_encoded.csv"
        
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
        shift_id_by: int = 1,  # Important: shift IDs to avoid 0
        chronological: bool = True,
        sample_ratio: float = 1.0,
    ):
        super().__init__()
        
        # Store core parameters
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
        self.sample_ratio = sample_ratio
        
        # Will be set during setup
        self.max_item_id = None
        self.all_item_ids = None
        
    def setup(self, stage: str = None):
        """Setup datasets - simplified to use only main_data_file"""
        
        if stage == "fit" or stage is None:
            # Training dataset - remove last item as target
            self.train_dataset = RecoDataset(
                main_data_file=self.main_data_file,
                item_features_path=self.item_features_path,
                max_sequence_length=self.max_sequence_length,
                ignore_last_n=1,  # For training, remove last item as target
                shift_id_by=self.shift_id_by,
                chronological=self.chronological,
                sample_ratio=self.sample_ratio,
                additional_columns=self.additional_columns,
                sequence_prefix=self.sequence_prefix,
                user_id_column=self.user_id_column,
            )
            
            # Validation dataset - same as training but keep all items
            self.val_dataset = RecoDataset(
                main_data_file=self.main_data_file,
                item_features_path=self.item_features_path,
                max_sequence_length=self.max_sequence_length,
                ignore_last_n=0,  # For validation, keep all items
                shift_id_by=self.shift_id_by,
                chronological=self.chronological,
                sample_ratio=1.0,  # Use full sequence for validation
                additional_columns=self.additional_columns,
                sequence_prefix=self.sequence_prefix,
                user_id_column=self.user_id_column,
            )
        
        if stage == "test" or stage is None:
            # Test dataset - same as validation
            self.test_dataset = RecoDataset(
                main_data_file=self.main_data_file,
                item_features_path=self.item_features_path,
                max_sequence_length=self.max_sequence_length,
                ignore_last_n=0,  # For testing, keep all items
                shift_id_by=self.shift_id_by,
                chronological=self.chronological,
                sample_ratio=1.0,
                additional_columns=self.additional_columns,
                sequence_prefix=self.sequence_prefix,
                user_id_column=self.user_id_column,
            )
        
        # Setup item information from feature lookup table
        self._setup_item_info()
    
    def _setup_item_info(self):
        """Setup item information from feature lookup table"""
        # Load item features to get all item IDs
        if os.path.exists(self.item_features_path):
            item_df = pd.read_csv(self.item_features_path)
            # Auto-detect item ID column
            id_column = None
            for col in item_df.columns:
                if col.lower().endswith('_id') or col.lower().endswith('id'):
                    id_column = col
                    break
            else:
                id_column = item_df.columns[0]  # Fallback to first column
            
            # Get all item IDs and add shift
            original_item_ids = item_df[id_column].tolist()
            self.all_item_ids = [item_id + self.shift_id_by for item_id in original_item_ids]
            self.max_item_id = max(self.all_item_ids) + 1
            
            print(f"Loaded {len(self.all_item_ids)} items from {self.item_features_path}")
            print(f"Max item ID: {self.max_item_id}")
        else:
            # Fallback: estimate from data
            print(f"Warning: Item features file not found: {self.item_features_path}")
            self.all_item_ids = list(range(1, 10000))  # Default range
            self.max_item_id = 10000
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch):
        """Convert new format to old format for compatibility"""
        if not batch:
            return {}
        
        converted_batch = {}
        
        for key in batch[0].keys():
            values = [sample[key] for sample in batch]
            
            if key == 'item_ids':
                # Split into historical + target
                stacked = torch.stack(values, dim=0)  # [B, L]
                
                # 🔧 FIX: 正确提取target - 使用actual length而不是固定位置
                converted_batch['item_ids'] = stacked  # 完整序列
                converted_batch['historical_ids'] = stacked[:, :-1]  # [B, L-1] 
                
                # 🔧 修复：根据实际长度提取target
                lengths = torch.tensor([sample['length'] for sample in batch])
                batch_size = stacked.size(0)
                
                # 提取每个序列的最后一个有效item作为target
                target_indices = (lengths - 1).clamp(min=0)  # 最后一个有效位置
                target_ids = stacked.gather(1, target_indices.unsqueeze(1))  # [B, 1]
                converted_batch['target_ids'] = target_ids.squeeze(1)  # [B]
                
            elif key == 'length':
                lengths = torch.tensor(values)
                converted_batch['length'] = lengths
                converted_batch['history_lengths'] = lengths - 1  # 历史长度
                
            elif key == 'ratings':
                stacked = torch.stack(values, dim=0)
                converted_batch['ratings'] = stacked
                converted_batch['historical_ratings'] = stacked[:, :-1]
                
                # 🔧 修复：根据实际长度提取target ratings
                lengths = torch.tensor([sample['length'] for sample in batch])
                target_indices = (lengths - 1).clamp(min=0)
                target_ratings = stacked.gather(1, target_indices.unsqueeze(1))
                converted_batch['target_ratings'] = target_ratings.squeeze(1)  # [B]
                
            elif key == 'timestamps':
                stacked = torch.stack(values, dim=0) 
                converted_batch['timestamps'] = stacked
                converted_batch['historical_timestamps'] = stacked[:, :-1]
                
                # 🔧 修复：根据实际长度提取target timestamps
                lengths = torch.tensor([sample['length'] for sample in batch])
                target_indices = (lengths - 1).clamp(min=0)
                target_timestamps = stacked.gather(1, target_indices.unsqueeze(1))
                converted_batch['target_timestamps'] = target_timestamps.squeeze(1)  # [B]
                
            else:
                converted_batch[key] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
        
        return converted_batch


class TencentRetrievalModel(Retrieval):
    """
    Extended Retrieval model for Tencent AMGR competition
    支持腾讯竞赛特定需求的检索模型
    """
    
    def __init__(self, *args, **kwargs):
        # Extract max_output_length from kwargs for our usage
        self.max_output_length = kwargs.pop('max_output_length', 10)
        super().__init__(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        """Training step with new features.py integration"""
        # Convert batch to SequentialFeatures format using new features.py
        # 使用原始的 gr_output_length + 1，然后通过切片调整
        
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch, 
            device=self.device,
            max_output_length=self.gr_output_length + 1  # ✅ 恢复原始逻辑：11
        )
        #debug_batch_keys(batch, "training_step")
        #
        #seq_features, target_ids, target_ratings = seq_features_from_row_debug(
        #    batch, 
        #    device=self.device,
        #    max_output_length=self.gr_output_length + 1
        #)        
        
        # Add target_ids at the end of the past_ids (like in original code)
        seq_features.past_ids.scatter_(
            dim=1,
            index=seq_features.past_lengths.view(-1, 1),
            src=target_ids.view(-1, 1),
        )
        
        # Get embeddings using get_item_embeddings method (like in original code)
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        # Replace past_embeddings in seq_features
        seq_features = seq_features._replace(past_embeddings=input_embeddings)
        
        # Use original training logic with converted features
        return self._compute_loss(seq_features, target_ids, target_ratings, stage="train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step with new features.py integration"""
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device, 
            max_output_length=self.gr_output_length + 1  # ✅ 恢复原始逻辑：11
        )
        #debug_batch_keys(batch, "validation_step")
        #
        ## 使用调试版本
        #seq_features, target_ids, target_ratings = seq_features_from_row_debug(
        #    batch,
        #    device=self.device, 
        #    max_output_length=self.gr_output_length + 1
        #)        
        
        # Get embeddings using get_item_embeddings method
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)
        
        return self._compute_loss(seq_features, target_ids, target_ratings, stage="val")
    
    def test_step(self, batch, batch_idx):
        """Test step with new features.py integration"""
        seq_features, target_ids, target_ratings = seq_features_from_row(
            batch,
            device=self.device,
            max_output_length=self.gr_output_length + 1  # ✅ 恢复原始逻辑：11
        )
        
        # Get embeddings using get_item_embeddings method
        input_embeddings = self.embeddings.get_item_embeddings(seq_features.past_ids)
        seq_features = seq_features._replace(past_embeddings=input_embeddings)
        
        return self._compute_loss(seq_features, target_ids, target_ratings, stage="test")
    
    def _compute_loss(self, seq_features, target_ids, target_ratings, stage="train"):
        """Compute loss and metrics using original Retrieval logic"""
        try:
            if stage == "train":
                # Training: compute loss like in original training_step
                seq_embeddings, _ = self.forward(seq_features)  # [B, X]
                
                # Prepare loss
                supervision_ids = seq_features.past_ids
                
                # Update negative sampler embeddings if needed
                if hasattr(self.negatives_sampler, '_item_emb'):
                    self.negatives_sampler._item_emb = self.embeddings._item_emb
                
                # Dense to jagged features (from original code)
                jagged_features = self.dense_to_jagged(
                    lengths=seq_features.past_lengths,
                    output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
                    supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                    supervision_embeddings=seq_features.past_embeddings[:, 1:, :],  # [B, N - 1, D]
                    supervision_weights=(supervision_ids[:, 1:] != 0).float(),  # ar_mask
                )
                
                loss = self.loss.jagged_forward(
                    negatives_sampler=self.negatives_sampler,
                    similarity=self.similarity,
                    **jagged_features,
                )
                
                self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
                return loss
                
            else:
                # Validation/test: compute metrics like in original validation_step
                top_k_ids, top_k_scores = self.retrieve(seq_features)
                self.metrics.update(top_k_ids=top_k_ids, target_ids=target_ids)
                
                # Log a dummy loss for validation
                loss = torch.tensor(0.0, device=self.device)
                self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
                return {"loss": loss}
        
        except Exception as e:
            print(f"Error in _compute_loss: {e}")
            import traceback
            traceback.print_exc()
            # Return a dummy loss to avoid crash
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
            return {"loss": loss}


def train_tencent_amgr_model(
    # Simplified data paths - only these two are needed!
    main_data_file: str,
    item_features_path: str,
    
    # Output settings
    output_dir: str = "logs/tencent_amgr",
    
    # Training parameters
    max_epochs: int = 500,
    min_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    
    # Model parameters
    max_sequence_length: int = 200,
    gr_output_length: int = 10,
    item_embedding_dim: int = 50,
    
    # Additional features
    additional_columns: list = None,
    sequence_prefix: str = "sequence_",
    user_id_column: str = None,
    
    # GPU settings
    use_gpu: bool = True,  # ✅ 新增GPU设置
    devices: int = 1,      # ✅ GPU设备数量
    
    # Other parameters
    seed: int = 42,
):
    """
    Simplified training function for Tencent AMGR competition
    简化的腾讯AMGR竞赛训练函数 - 只需要两个数据文件！
    
    Args:
        main_data_file: Path to main data CSV (e.g., "sasrec_format_by_user_train.csv")
        item_features_path: Path to item features CSV (e.g., "movies_encoded.csv")
        output_dir: Directory to save outputs
        max_epochs: Maximum training epochs
        min_epochs: Minimum training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        max_sequence_length: Maximum sequence length
        gr_output_length: Generative recommendation output length
        item_embedding_dim: Item embedding dimension
        additional_columns: Additional user feature columns
        sequence_prefix: Prefix for sequence columns
        user_id_column: User ID column name
        use_gpu: Whether to use GPU if available
        devices: Number of GPU devices to use
        seed: Random seed
    """
    
    # Set seed for reproducibility
    if seed is not None:
        L.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # ✅ GPU配置和检测
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
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Training run directory: {run_dir}")
    
    # Create simplified data module using new TencentDataModule
    print("Creating simplified Tencent data module...")
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
        shift_id_by=1,  # Shift IDs to avoid 0
        chronological=True,
        sample_ratio=1.0,
    )
    
    # Setup datamodule to get item information
    datamodule.setup("fit")
    num_items = datamodule.max_item_id
    all_item_ids_tensor = torch.tensor(datamodule.all_item_ids, dtype=torch.long)
    
    print(f"Number of items: {num_items}")
    
    # Create model components
    print("Creating model components...")
    
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
    
    # HSTU sequence encoder (FIXED: correct max_sequence_len and max_output_len)
    sequence_encoder = HSTU(
        max_sequence_len=max_sequence_length,  # 200
        max_output_len=gr_output_length + 1,   # 11 (10 + 1)
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
        k=200,
        top_k_module=MIPSBruteForceTopK(),
        ids=all_item_ids_tensor
    )
    
    # Loss function
    loss = SampledSoftmaxLoss(
        num_to_sample=128,
        softmax_temperature=0.05
    )
    
    # Metrics
    metrics = RetrievalMetrics(
        k=200,
        at_k_list=[10, 50, 100, 200]
    )
    
    # Create optimizer and scheduler as partial functions
    optimizer = partial(
        torch.optim.AdamW,
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay
    )
    
    scheduler = partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="max",
        factor=0.1,
        patience=10,
        threshold=1e-4
    )
    
    configure_optimizer_params = {
        "monitor": "val/ndcg@100",
        "interval": "epoch",
        "frequency": 1
    }
    
    # Create model using TencentRetrievalModel
    print("Creating Tencent Retrieval model...")
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
        max_output_length=gr_output_length,  # Pass to our custom model
        compile_model=False
    )
    
    # Create callbacks
    print("Setting up callbacks...")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="epoch_{epoch:03d}",
        save_last=True,
        save_top_k=1,
        monitor="val/ndcg@100",
        mode="max",
        verbose=True,
        save_weights_only=False,
        every_n_epochs=1
    )
    
    early_stopping = EarlyStopping(
        monitor="val/ndcg@100",
        mode="max",
        patience=20,
        verbose=True,
        check_finite=True
    )
    
    callbacks = [
        checkpoint_callback,
        early_stopping,
        RichModelSummary(max_depth=3),
        RichProgressBar(),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Create loggers
    print("Setting up loggers...")
    
    loggers = [
        CSVLogger(save_dir=run_dir / "csv", name="", version=0, flush_logs_every_n_steps=1),
        TensorBoardLogger(save_dir=run_dir / "tensorboard", name="", version=0, log_graph=False)
    ]
    
    # Create trainer
    print("Creating trainer...")
    trainer = L.Trainer(
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        benchmark=True if torch.cuda.is_available() else False,
        deterministic=False,
        default_root_dir=run_dir,
        num_sanity_val_steps=2,
        reload_dataloaders_every_n_epochs=0,
        detect_anomaly=False,
    )
    
    # Save configuration
    config_file = run_dir / "tencent_amgr_config.txt"
    with open(config_file, "w", encoding='utf-8') as f:
        f.write("Tencent AMGR Competition Training Configuration\n")
        f.write("==============================================\n\n")
        f.write(f"Data Files:\n")
        f.write(f"  Main Data: {main_data_file}\n")
        f.write(f"  Item Features: {item_features_path}\n\n")
        f.write(f"Dataset Configuration:\n")
        f.write(f"  Max sequence length: {max_sequence_length}\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Number of items: {num_items}\n")
        f.write(f"  Additional columns: {additional_columns}\n")
        f.write(f"  Sequence prefix: {sequence_prefix}\n")
        f.write(f"  User ID column: {user_id_column}\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Architecture: HSTU\n")
        f.write(f"  Item embedding dim: {item_embedding_dim}\n")
        f.write(f"  GR output length: {gr_output_length}\n")
        f.write(f"  Number of blocks: 2\n")
        f.write(f"  Number of heads: 1\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Max epochs: {max_epochs}\n")
        f.write(f"  Min epochs: {min_epochs}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Weight decay: {weight_decay}\n")
        f.write(f"  Optimizer: AdamW (betas=[0.9, 0.98])\n")
        f.write(f"  Scheduler: ReduceLROnPlateau\n")
        f.write(f"  Monitor metric: val/ndcg@100\n")
        f.write(f"  Seed: {seed}\n")
        f.write(f"Hardware Configuration:\n")
        f.write(f"  CUDA Available: {gpu_available}\n")
        f.write(f"  Accelerator: {accelerator}\n")
        f.write(f"  Devices: {devices_actual}\n")
        f.write(f"  Precision: {precision}\n")
        if use_gpu_actual:
            f.write(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
        f.write(f"\n")
    
    print(f"Configuration saved to: {config_file}")
    
    # Start training
    print("\nStarting Tencent AMGR training...")
    print(f"Training on: {trainer.accelerator}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    try:
        trainer.fit(model=model, datamodule=datamodule)
        
        # Test on best checkpoint
        print("\nTesting on best checkpoint...")
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Best checkpoint: {best_model_path}")
            test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path=best_model_path)
        else:
            print("No best checkpoint found, testing with current weights...")
            test_results = trainer.test(model=model, datamodule=datamodule)
        
        print(f"Test results: {test_results}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    
    # Save final results
    print(f"\nTencent AMGR training completed!")
    print(f"Outputs saved to: {run_dir}")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Metrics CSV: {run_dir / 'csv/version_0/metrics.csv'}")
    
    return trainer, model


if __name__ == "__main__":
    # Simplified example usage for Tencent AMGR competition
    trainer, model = train_tencent_amgr_model(
        # Only two data files needed! 只需要两个数据文件！
        main_data_file="../generative-recommenders-pl/tmp/ml-1m/sasrec_format_by_user_train.csv",
        item_features_path=".movies_encoded.csv",
        
        # Output settings
        output_dir="logs/tencent_amgr",
        
        # Training parameters
        max_epochs=500,
        min_epochs=10,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.001,
        
        # Model parameters  
        max_sequence_length=200,
        gr_output_length=10,
        item_embedding_dim=50,
        
        # Additional features from main CSV file
        additional_columns=[
            'sex', 'age_group', 'occupation', 'zip_code'  # User features from main CSV
            # For Tencent data, add: 'user_city', 'ad_category', 'advertiser_id', etc.
        ],
        sequence_prefix="sequence_",
        user_id_column=None,  # Will auto-detect
        
        # Seed for reproducibility
        seed=42,
    )
    
    print("\nTencent AMGR training script completed successfully!")
    print("已成功完成腾讯AMGR训练脚本！")
    print("现在只需要两个文件：主数据文件 + 物品特征文件")
