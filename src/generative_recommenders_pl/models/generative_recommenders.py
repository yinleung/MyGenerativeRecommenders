from typing import Any, Union

import hydra
import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig

from generative_recommenders_pl.data.reco_dataset import RecoDataModule
from generative_recommenders_pl.models.embeddings.embeddings import EmbeddingModule
from generative_recommenders_pl.models.indexing.candidate_index import CandidateIndex
from generative_recommenders_pl.models.losses.autoregressive_losses import (
    AutoregressiveLoss,
)
from generative_recommenders_pl.models.negatives_samples.negative_sampler import (
    NegativesSampler,
)
from generative_recommenders_pl.models.postprocessors.postprocessors import (
    OutputPostprocessorModule,
)
from generative_recommenders_pl.models.preprocessors import (
    InputFeaturesPreprocessorModule,
)
from generative_recommenders_pl.models.similarity.ndp_module import NDPModule
from generative_recommenders_pl.models.utils import ops
from generative_recommenders_pl.models.utils.features import SequentialFeatures
from generative_recommenders_pl.utils.logger import RankedLogger

from generative_recommenders_pl.models.optimizers.muon import Muon
from generative_recommenders_pl.models.optimizers.scion import Scion

log = RankedLogger(__name__)


class GenerativeRecommenders(L.LightningModule):
    def __init__(
        self,
        datamodule: RecoDataModule | DictConfig,
        embeddings: EmbeddingModule | DictConfig,
        preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negatives_sampler: NegativesSampler | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
        optimizer1: Union[DictConfig, Any],
        optimizer2: Union[DictConfig, Any],
        scheduler1: Union[DictConfig, Any],
        scheduler2: Union[DictConfig, Any],
        configure_optimizer_params: DictConfig,
        gr_output_length: int,
        item_embedding_dim: int,
        compile_model: bool,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.optimizer1 = hydra.utils.instantiate(optimizer1)
        self.optimizer2 = hydra.utils.instantiate(optimizer2)

        self.scheduler1 = hydra.utils.instantiate(scheduler1)
        self.scheduler2 = hydra.utils.instantiate(scheduler2)

        if optimizer1 is None and optimizer2 is not None:
            self.optimizer1 = hydra.utils.instantiate(optimizer2)
            self.optimizer2 = None
        elif optimizer1 is not None:
            self.optimizer1 = hydra.utils.instantiate(optimizer1)
            self.optimizer2 = hydra.utils.instantiate(optimizer2) if optimizer2 is not None else None
        else:
            raise ValueError("At least one optimizer (optimizer1 or optimizer2) must be provided.")

        if scheduler1 is None and scheduler2 is not None:
            self.scheduler1 = hydra.utils.instantiate(scheduler2)
            self.scheduler2 = None
        elif scheduler1 is not None:
            self.scheduler1 = hydra.utils.instantiate(scheduler1)
            self.scheduler2 = hydra.utils.instantiate(scheduler2) if scheduler2 is not None else None
        else:
            self.scheduler1 = None
            self.scheduler2 = None

        self.configure_optimizer_params: dict[str, Any] = configure_optimizer_params

        self.gr_output_length: int = gr_output_length
        self.item_embedding_dim: int = item_embedding_dim
        self.compile_model: bool = compile_model

        self.__hydra_init_submodules(
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
        )

    def __hydra_init_submodules(
        self,
        datamodule: RecoDataModule,
        embeddings: EmbeddingModule | DictConfig,
        preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        sequence_encoder: torch.nn.Module | DictConfig,
        postprocessor: OutputPostprocessorModule | DictConfig,
        similarity: NDPModule | DictConfig,
        negatives_sampler: NegativesSampler | DictConfig,
        candidate_index: CandidateIndex | DictConfig,
        loss: AutoregressiveLoss | DictConfig,
        metrics: torchmetrics.Metric | DictConfig,
    ) -> None:
        def init_embedding_module(embeddings: EmbeddingModule) -> EmbeddingModule:
            if isinstance(embeddings, DictConfig):
                kwargs = {}
                if "num_items" not in embeddings:
                    kwargs["num_items"] = datamodule.max_item_id
                if "item_embedding_dim" not in embeddings:
                    kwargs["item_embedding_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(embeddings, **kwargs)
            else:
                return embeddings

        def init_preprocessor_module(
            preprocessor: InputFeaturesPreprocessorModule | DictConfig,
        ) -> InputFeaturesPreprocessorModule:
            if isinstance(embeddings, DictConfig):
                kwargs = {}
                if "max_sequence_len" not in preprocessor:
                    kwargs["max_sequence_len"] = (
                        datamodule.max_sequence_length + self.gr_output_length + 1
                    )
                if "embedding_dim" not in preprocessor:
                    kwargs["embedding_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(preprocessor, **kwargs)
            else:
                return preprocessor

        def init_sequence_encoder_module(
            sequence_encoder: torch.nn.Module | DictConfig,
        ) -> torch.nn.Module:
            if isinstance(sequence_encoder, DictConfig):
                kwargs = {}
                if "max_sequence_len" not in sequence_encoder:
                    kwargs["max_sequence_len"] = datamodule.max_sequence_length
                if "max_output_len" not in sequence_encoder:
                    kwargs["max_output_len"] = self.gr_output_length + 1
                if "embedding_dim" not in sequence_encoder:
                    kwargs["embedding_dim"] = self.item_embedding_dim
                if "item_embedding_dim" not in sequence_encoder:
                    kwargs["item_embedding_dim"] = self.item_embedding_dim
                if "attention_dim" not in sequence_encoder:
                    kwargs["attention_dim"] = self.item_embedding_dim
                if "linear_dim" not in sequence_encoder:
                    kwargs["linear_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(sequence_encoder, **kwargs)
            else:
                return sequence_encoder

        def init_postprocessor_module(
            postprocessor: OutputPostprocessorModule | DictConfig,
        ) -> OutputPostprocessorModule:
            if isinstance(postprocessor, DictConfig):
                kwargs = {}
                if "embedding_dim" not in postprocessor:
                    kwargs["embedding_dim"] = self.item_embedding_dim
                return hydra.utils.instantiate(postprocessor, **kwargs)
            else:
                return postprocessor

        def init_similarity_module(similarity: NDPModule | DictConfig) -> NDPModule:
            if isinstance(similarity, DictConfig):
                return hydra.utils.instantiate(similarity)
            else:
                return similarity

        def init_negatives_sampler_module(
            negatives_sampler: NegativesSampler | DictConfig,
        ) -> NegativesSampler:
            if isinstance(negatives_sampler, DictConfig):
                kwargs = {}
                if negatives_sampler["_target_"].endswith("LocalNegativesSampler"):
                    if "num_items" not in negatives_sampler:
                        kwargs["all_item_ids"] = datamodule.all_item_ids
                return hydra.utils.instantiate(negatives_sampler, **kwargs)
            else:
                return negatives_sampler

        def init_candidate_index_module(
            candidate_index: CandidateIndex,
        ) -> CandidateIndex:
            if isinstance(candidate_index, DictConfig):
                kwargs = {}
                if "ids" not in candidate_index:
                    kwargs["ids"] = datamodule.all_item_ids
                return hydra.utils.instantiate(candidate_index, **kwargs)
            else:
                return candidate_index

        def init_loss_module(
            loss: AutoregressiveLoss | DictConfig,
        ) -> AutoregressiveLoss:
            if isinstance(loss, DictConfig):
                return hydra.utils.instantiate(loss)
            else:
                return loss

        def init_metrics_module(
            metrics: torchmetrics.Metric | DictConfig,
        ) -> torchmetrics.Metric:
            if isinstance(metrics, DictConfig):
                return hydra.utils.instantiate(metrics)
            else:
                return metrics

        self.embeddings: EmbeddingModule = init_embedding_module(embeddings)
        self.preprocessor: InputFeaturesPreprocessorModule = init_preprocessor_module(
            preprocessor
        )
        self.sequence_encoder: torch.nn.Module = init_sequence_encoder_module(
            sequence_encoder
        )
        self.postprocessor: OutputPostprocessorModule = init_postprocessor_module(
            postprocessor
        )
        self.similarity: NDPModule = init_similarity_module(similarity)
        self.negatives_sampler: NegativesSampler = init_negatives_sampler_module(
            negatives_sampler
        )
        self.candidate_index: CandidateIndex = init_candidate_index_module(
            candidate_index
        )
        self.loss: AutoregressiveLoss = init_loss_module(loss)
        self.metrics: torchmetrics.Metric = init_metrics_module(metrics)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage (str): One of 'fit', 'validate', 'test', or 'predict'.
        """
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        """
        Configure optimizers and learning rate schedulers.

        If only one optimizer is provided, all model parameters will be passed to it.
        If two optimizers are provided, parameters will be split into groups.
        """
        raw_model = self.trainer.model

        optimizers = []
        schedulers = []

        configure_params = getattr(self, "configure_optimizer_params", {})

        # === Count how many optimizers are defined ===
        has_opt1 = hasattr(self, "optimizer1") and self.optimizer1 is not None
        has_opt2 = hasattr(self, "optimizer2") and self.optimizer2 is not None

        num_optimizers = has_opt1 + has_opt2

        if num_optimizers == 0:
            raise ValueError("At least one optimizer (optimizer1 or optimizer2) must be provided.")

        if num_optimizers == 1:
            # === Only one optimizer: use all parameters ===
            # all_params = list(raw_model.parameters())
            if has_opt1:
                optimizer = self.optimizer1(params=raw_model.parameters())
                optimizers.append(optimizer)
                if hasattr(self, "scheduler1") and self.scheduler1 is not None:
                    scheduler = self.scheduler1(optimizer=optimizer)
                    schedulers.append({"scheduler": scheduler, **configure_params})
            else:
                optimizer = self.optimizer2(params=all_params)
                optimizers.append(optimizer)
                if hasattr(self, "scheduler2") and self.scheduler2 is not None:
                    scheduler = self.scheduler2(optimizer=optimizer)
                    schedulers.append({"scheduler": scheduler, **configure_params})

        else:
            # === Two optimizers: split param groups ===
            embed_params = [
                p for n, p in raw_model.named_parameters()
                if "embed" in n
            ]
            scalar_params = [
                p for p in raw_model.parameters()
                if p.ndim < 2
            ]
            hall_params = set(embed_params + scalar_params)
            hidden_matrix_params = [p for p in raw_model.parameters() if p not in hall_params and p.ndim >= 2]
            all_params = set(embed_params + scalar_params + hidden_matrix_params)
            other_params = [p for p in raw_model.parameters() if p not in all_params]


            optimizer1 = self.optimizer1(params=scalar_params + embed_params)
            optimizer2 = self.optimizer2(params=hidden_matrix_params + other_params)

            optimizers.extend([optimizer1, optimizer2])

            if hasattr(self, "scheduler1") and self.scheduler1 is not None:
                sched1 = self.scheduler1(optimizer=optimizer1)
                schedulers.append({"scheduler": sched1, **configure_params})

            if hasattr(self, "scheduler2") and self.scheduler2 is not None:
                sched2 = self.scheduler2(optimizer=optimizer2)
                schedulers.append({"scheduler": sched2, **configure_params})

        return optimizers, schedulers

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # Call the superclass's state_dict method to get the full state dictionary
        state_dict = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # List of module names you don't want to save
        modules_to_exclude = [
            "similarity",
            "negatives_sampler",
            "candidate_index",
            "loss",
            "metrics",
        ]

        # Remove the keys corresponding to the modules to exclude
        keys_to_remove = [
            key
            for key in state_dict.keys()
            for module_name in modules_to_exclude
            if key.startswith(prefix + module_name)
        ]
        for key in keys_to_remove:
            del state_dict[key]

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        # since we removed some keys from the state_dict, we need to set strict=False
        super().load_state_dict(state_dict, strict=False)

    def forward(
        self, seq_features: SequentialFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lightning calls this inside the training loop.

        Args:
            seq_features (SequentialFeatures): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
            cached_states: The cached states.
        """
        # input features preprocessor
        past_lengths, user_embeddings, valid_mask, aux_mask = self.preprocessor(
            past_lengths=seq_features.past_lengths,
            past_ids=seq_features.past_ids,
            past_embeddings=seq_features.past_embeddings,
            past_payloads=seq_features.past_payloads,
        )

        # sequence encoder
        user_embeddings, cached_states = self.sequence_encoder(
            past_lengths=past_lengths,
            user_embeddings=user_embeddings,
            valid_mask=valid_mask,
            past_payloads=seq_features.past_payloads,
        )

        if aux_mask is not None:
            user_embeddings, _ = ops.mask_dense_by_aux_mask(
                user_embeddings,
                aux_mask,
                past_lengths,
                max_lengths=seq_features.past_ids.size(1),
            )

        # output postprocessor
        encoded_embeddings = self.postprocessor(user_embeddings)
        return encoded_embeddings, cached_states

    def dense_to_jagged(
        self, lengths: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Convert dense tensor to jagged tensor.

        Args:
            lengths (torch.Tensor): The lengths tensor.
            **kwargs: The dict with the dense tensor to be converted.

        Returns:
            dict[str, torch.Tensor]: The jagged tensor.
        """
        jagged_id_offsets = ops.asynchronous_complete_cumsum(lengths)
        output = {}
        if "supervision_ids" in kwargs:
            output["supervision_ids"] = (
                ops.dense_to_jagged(
                    kwargs.pop("supervision_ids").unsqueeze(-1).float(),
                    jagged_id_offsets,
                )
                .squeeze(1)
                .long()
            )

        if "supervision_weights" in kwargs:
            output["supervision_weights"] = ops.dense_to_jagged(
                kwargs.pop("supervision_weights").unsqueeze(-1), jagged_id_offsets
            ).squeeze(1)
        for key, value in kwargs.items():
            output[key] = ops.dense_to_jagged(value, jagged_id_offsets)
        return output
