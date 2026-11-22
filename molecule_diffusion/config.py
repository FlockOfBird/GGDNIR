from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DataConfig:
    dataset_path: str = "data/250k_zinc_clean.csv"
    smiles_column: str = "smiles"
    split_column: str | None = None
    label_column: str | None = "qed"
    property_name: str | None = "qed"
    task_type: str = "regression"
    max_length: int = 160
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    batch_size: int = 128
    num_workers: int = 0
    seed: int = 42
    split_strategy: str = "scaffold"
    canonicalize_smiles: bool = True
    drop_invalid_smiles: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EncoderConfig:
    model_name: str = "DeepChem/ChemBERTa-100M-MLM"
    pooling: str = "mean"
    freeze_encoder: bool = True
    local_files_only: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LatentProjectionConfig:
    mode: str = "none"
    enabled: bool = False
    output_dim: int = 256
    fit_max_tokens: int = 50_000
    seed: int = 42

    @property
    def learned(self) -> bool:
        return self.mode == "learned"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecoderConfig:
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    label_smoothing: float = 0.0
    latent_noise_std: float = 0.1
    robust_loss_weight: float = 0.5
    max_decode_length: int = 160

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DiffusionConfig:
    # Bound to the selected encoder's hidden size after embeddings load.
    latent_dim: int | None = None
    # Fixed by the active pipeline; legacy fields remain for checkpoint metadata.
    denoiser_architecture: str = "mlp"
    hidden_dim: int = 768
    time_embedding_dim: int = 256
    num_attention_heads: int = 8
    dim_feedforward: int = 1536
    max_sequence_length: int = 160
    num_steps: int = 500
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    num_residual_layers: int = 6
    dropout: float = 0.1
    conv_channels: int = 384
    conv_num_layers: int = 8
    conv_kernel_size: int = 3
    conv_dilation_cycle_length: int = 7
    mlp_hidden_dim: int = 6144
    mlp_token_hidden_dim: int = 256
    mlp_num_layers: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassifierConfig:
    hidden_dim: int = 256
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


PropertyRegressorConfig = ClassifierConfig


@dataclass
class GuidanceClassifierConfig:
    hidden_dim: int = 512
    dropout: float = 0.1
    time_embedding_dim: int = 512
    num_hidden_layers: int = 3
    distill_from_classifier: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


LatentHeadConfig = GuidanceClassifierConfig
GuidanceRegressorConfig = GuidanceClassifierConfig


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    diffusion_learning_rate: float = 1e-4
    decoder_learning_rate: float = 1e-4
    projection_learning_rate: float = 1e-5
    projection_reconstruction_weight: float = 1.0
    projection_variance_weight: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    artifact_dir: str = "artifacts"
    checkpoint_dir: str = "artifacts/checkpoints"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SamplingConfig:
    num_samples: int = 1024
    batch_size: int = 64
    guidance_scale: float = 0.0
    guidance_gamma: float = 1.0
    temperature: float = 1.0
    output_path: str = "artifacts/experiments/sampling/<run-id>/samples.csv"

    def to_dict(self) -> dict:
        return asdict(self)
