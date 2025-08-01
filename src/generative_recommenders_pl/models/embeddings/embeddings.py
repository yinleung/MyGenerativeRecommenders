import abc

import torch

from generative_recommenders_pl.models.utils.initialization import truncated_normal
from generative_recommenders_pl.utils.logger import RankedLogger

log = RankedLogger(__name__)

# 7.22 Modification
# 读取年份数据并创建预计算的lookup table
import pandas as pd
try:
    df = pd.read_csv("/home/wenlk/xlli/MyGenerativeRecommenders/tmp/processed/ml-1m/movies.csv")
    item2year = {int(row["movie_id"]): int(row["year"]) for _, row in df.iterrows()}
except Exception as e:
    print(f"Warning: Could not load movies data: {e}")
    item2year = {}


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_all_embeddings(self, item_ids: torch.Tensor, item_years:torch.Tensor):
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor):
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        # self._item_embedding_dim: int = item_embedding_dim
        # self._item_emb = torch.nn.Embedding(
        #     num_items + 1, 50, padding_idx=0
        # )
        # self._year_emb = torch.nn.Embedding(
        #     num_items + 1, 50, padding_idx=0
        # )
        self._item_embedding_dim: int = item_embedding_dim
        # 根据配置的一半来创建子嵌入
        half_dim = item_embedding_dim // 2
        self._item_emb = torch.nn.Embedding(
            num_items + 1, half_dim, padding_idx=0
        )
        self._year_emb = torch.nn.Embedding(
            num_items + 1, half_dim, padding_idx=0
        )
        
        # 创建预计算的tensor lookup table
        max_item_id = max(item2year.keys()) if item2year else num_items
        year_lookup_table = torch.zeros(max_item_id + 1, dtype=torch.long)
        
        for item_id, year in item2year.items():
            year_lookup_table[item_id] = year
        
        # 注册为buffer，自动处理device移动，不参与梯度计算
        self.register_buffer('year_lookup_table', year_lookup_table)

        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name or "_year_emb" in name:
                log.info(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                log.info(f"Skipping initializing params {name} - not configured")

    def lookup_year_ids(self, item_ids: torch.Tensor) -> torch.Tensor:
        valid_indices = torch.clamp(item_ids, 0, self.year_lookup_table.size(0) - 1)
        return self.year_lookup_table[valid_indices]

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_emb = self._item_emb(item_ids)       # [*, 25]
        year_emb = self._year_emb(self.lookup_year_ids(item_ids))  # [*, 25]
        return torch.cat([item_emb, year_emb], dim=-1)  # [*, 50]

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self):
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                log.info(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                log.info(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim