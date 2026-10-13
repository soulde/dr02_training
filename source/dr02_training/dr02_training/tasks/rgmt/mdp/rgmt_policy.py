"""RGMT Transformer policies and feature dimensions for DR02 Pro."""

import copy

import torch
import torch.nn as nn
from rsl_rl.modules import (
    CrossAttention,
    EmpiricalNormalization,
    HiddenState,
    MLP,
    SinusoidalPositionalEncoding,
)
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict

from dr02_asset.profile import JOINT_ORDER

RGMT_PROPRIO_HISTORY_LENGTH = 10
RGMT_COMMAND_WINDOW_LENGTH = 21
RGMT_EMBEDDING_DIM = 128
DR02_ACTUATED_JOINT_COUNT = len(JOINT_ORDER)

DR02_PROPRIOCEPTIVE_FRAME_DIM = 6 + 3 * DR02_ACTUATED_JOINT_COUNT
DR02_COMMAND_FRAME_DIM = 9 + DR02_ACTUATED_JOINT_COUNT


class RGMTPolicyModel(nn.Module):
    """Common history/command encoder and RSL-RL-compatible model API."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        current_groups: list[str],
        history_group: str,
        command_group: str,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        activation: str = "elu",
        embedding_dim: int = RGMT_EMBEDDING_DIM,
        num_heads: int = 4,
        history_layers: int = 1,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        if obs_set not in obs_groups:
            raise KeyError(f"observation set '{obs_set}' is not defined")
        self.obs_groups = list(obs_groups[obs_set])
        self.current_groups = list(current_groups)
        self.history_group = history_group
        self.command_group = command_group
        required_groups = {*self.current_groups, history_group, command_group}
        if not required_groups.issubset(self.obs_groups):
            missing_groups = sorted(required_groups - set(self.obs_groups))
            raise ValueError(f"model groups are missing from obs_groups['{obs_set}']: {missing_groups}")
        for group in self.obs_groups:
            if group not in obs:
                raise KeyError(f"observation group '{group}' is missing")
        if not self.current_groups:
            raise ValueError("current_groups must contain at least one vector observation")
        if any(obs[group].ndim != 2 for group in self.current_groups):
            raise ValueError("current_groups must contain rank-2 vector observations")
        if obs[self.current_groups[0]].shape[-1] != DR02_PROPRIOCEPTIVE_FRAME_DIM:
            raise ValueError(
                f"the first current group must contain {DR02_PROPRIOCEPTIVE_FRAME_DIM} proprioceptive features"
            )
        if obs[history_group].ndim != 3 or obs[command_group].ndim != 3:
            raise ValueError("history_group and command_group must contain rank-3 sequence observations")
        if obs[history_group].shape[1] != RGMT_PROPRIO_HISTORY_LENGTH:
            raise ValueError(f"history_group must contain {RGMT_PROPRIO_HISTORY_LENGTH} frames")
        if obs[command_group].shape[1] != RGMT_COMMAND_WINDOW_LENGTH:
            raise ValueError(f"command_group must contain {RGMT_COMMAND_WINDOW_LENGTH} frames")
        if obs[history_group].shape[-1] != DR02_PROPRIOCEPTIVE_FRAME_DIM:
            raise ValueError(f"history_group must contain {DR02_PROPRIOCEPTIVE_FRAME_DIM} features per frame")
        if obs[command_group].shape[-1] != DR02_COMMAND_FRAME_DIM:
            raise ValueError(f"command_group must contain {DR02_COMMAND_FRAME_DIM} features per frame")
        if embedding_dim <= 0 or num_heads <= 0 or embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be positive and divisible by num_heads")
        if history_layers <= 0 or feedforward_dim <= 0:
            raise ValueError("history_layers and feedforward_dim must be positive")

        self._example_obs_shapes = {group: tuple(obs[group].shape) for group in self.obs_groups}
        self.embedding_dim = embedding_dim
        self.current_input_dims = [obs[group].shape[-1] for group in self.current_groups]
        self.history_input_dim = obs[history_group].shape[-1]
        self.command_input_dim = obs[command_group].shape[-1]
        self.current_dim = sum(self.current_input_dims)
        self.obs_normalization = obs_normalization
        self.current_normalizer = (
            EmpiricalNormalization(self.current_dim) if obs_normalization else nn.Identity()
        )
        self.history_normalizer = EmpiricalNormalization(self.history_input_dim) if obs_normalization else nn.Identity()
        self.command_normalizer = EmpiricalNormalization(self.command_input_dim) if obs_normalization else nn.Identity()
        self.history_projection = MLP(self.history_input_dim, embedding_dim, [embedding_dim], "gelu")
        self.command_projection = MLP(self.command_input_dim, embedding_dim, [embedding_dim], "gelu")
        self.query_projection = MLP(embedding_dim, embedding_dim, [embedding_dim], "gelu")
        self.history_position = SinusoidalPositionalEncoding(embedding_dim, max_len=RGMT_PROPRIO_HISTORY_LENGTH)
        self.command_position = SinusoidalPositionalEncoding(embedding_dim, max_len=RGMT_COMMAND_WINDOW_LENGTH)
        self.history_encoder = _make_encoder(embedding_dim, num_heads, history_layers, feedforward_dim, dropout)
        self.cross_attention = CrossAttention(embedding_dim, num_heads, dropout)
        self.command_query_norm = nn.LayerNorm(embedding_dim)
        self.command_ffn_norm = nn.LayerNorm(embedding_dim)
        self.command_ffn = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
            nn.Dropout(dropout),
        )
        self.command_output_norm = nn.LayerNorm(embedding_dim)

        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim
        self.mlp = MLP(self.current_dim + embedding_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Compute a value or deterministic/stochastic residual action."""
        if masks is not None and not self.is_recurrent:
            obs = unpad_trajectories(obs, masks)
        output = self.mlp(self.get_latent(obs))
        if self.distribution is None:
            return output
        if stochastic_output:
            self.distribution.update(output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(output)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Encode current state, causal history, and future/past command context."""
        del masks, hidden_state
        current = self.current_normalizer(torch.cat([obs[group] for group in self.current_groups], dim=-1))
        history = self.history_normalizer(obs[self.history_group])
        history = self.history_position(self.history_projection(history))
        causal_mask = torch.triu(
            torch.ones(history.shape[1], history.shape[1], dtype=torch.bool, device=history.device), diagonal=1
        )
        history = self.history_encoder(history, mask=causal_mask)
        history_query = history.max(dim=1, keepdim=True).values

        query = self.query_projection(history_query)
        command = self.command_normalizer(obs[self.command_group])
        command = self.command_position(self.command_projection(command))
        attended = query + self.cross_attention(self.command_query_norm(query), command, command)
        command_embedding = self.command_output_norm(attended + self.command_ffn(self.command_ffn_norm(attended)))
        fused = command_embedding.squeeze(1)
        return torch.cat((current, fused), dim=-1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset recurrent state (no-op because temporal context is explicit in observations)."""
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        """Return ``None`` because temporal context is carried by the input observations."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach recurrent state (no-op for this feed-forward model)."""
        del dones

    @property
    def output_mean(self) -> torch.Tensor:
        """Mean of the current output distribution."""
        return self.distribution.mean  # type: ignore[union-attr]

    @property
    def output_std(self) -> torch.Tensor:
        """Standard deviation of the current output distribution."""
        return self.distribution.std  # type: ignore[union-attr]

    @property
    def output_entropy(self) -> torch.Tensor:
        """Entropy of the current output distribution."""
        return self.distribution.entropy  # type: ignore[union-attr]

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Raw parameters of the current output distribution."""
        return self.distribution.params  # type: ignore[union-attr]

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute output log probabilities."""
        return self.distribution.log_prob(outputs)  # type: ignore[union-attr]

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two distribution parameterizations."""
        return self.distribution.kl_divergence(old_params, new_params)  # type: ignore[union-attr]

    def update_normalization(self, obs: TensorDict) -> None:
        """Update empirical statistics for current, history, and command observations."""
        if not self.obs_normalization:
            return
        self.current_normalizer.update(torch.cat([obs[group] for group in self.current_groups], dim=-1))
        self.history_normalizer.update(obs[self.history_group].reshape(-1, self.history_input_dim))
        self.command_normalizer.update(obs[self.command_group].reshape(-1, self.command_input_dim))

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-compatible deterministic model taking an ordered tensor list."""
        return _TorchRGMTModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-compatible deterministic model with one tensor input per group."""
        return _OnnxRGMTModel(self, verbose)


class RGMTActor(RGMTPolicyModel):
    """RGMT actor predicting Gaussian residual joint-position actions."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **kwargs,
    ) -> None:
        if output_dim != DR02_ACTUATED_JOINT_COUNT:
            raise ValueError(f"RGMT actor output must contain {DR02_ACTUATED_JOINT_COUNT} joint actions")
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)


class RGMTCritic(RGMTPolicyModel):
    """RGMT asymmetric critic predicting a scalar value."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **kwargs,
    ) -> None:
        if output_dim != 1:
            raise ValueError("RGMT critic output must be a scalar")
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)


def _make_encoder(
    embedding_dim: int, num_heads: int, num_layers: int, feedforward_dim: int, dropout: float
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=num_heads,
        dim_feedforward=feedforward_dim,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=num_layers,
        norm=nn.LayerNorm(embedding_dim),
        enable_nested_tensor=False,
    )


class _TorchRGMTModel(nn.Module):
    """TorchScript wrapper over the actor/critic's configured ordered observation inputs."""

    current_indices: list[int]

    def __init__(self, model: RGMTPolicyModel) -> None:
        super().__init__()
        self.input_shapes = [tuple(model._example_obs_shapes[group]) for group in model.obs_groups]
        self.current_indices = [model.obs_groups.index(group) for group in model.current_groups]
        self.history_index = model.obs_groups.index(model.history_group)
        self.command_index = model.obs_groups.index(model.command_group)
        self.current_normalizer = copy.deepcopy(model.current_normalizer)
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.command_normalizer = copy.deepcopy(model.command_normalizer)
        self.history_projection = copy.deepcopy(model.history_projection)
        self.command_projection = copy.deepcopy(model.command_projection)
        self.history_position = copy.deepcopy(model.history_position)
        self.command_position = copy.deepcopy(model.command_position)
        self.query_projection = copy.deepcopy(model.query_projection)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.cross_attention = copy.deepcopy(model.cross_attention)
        self.command_query_norm = copy.deepcopy(model.command_query_norm)
        self.command_ffn_norm = copy.deepcopy(model.command_ffn_norm)
        self.command_ffn = copy.deepcopy(model.command_ffn)
        self.command_output_norm = copy.deepcopy(model.command_output_norm)
        self.mlp = copy.deepcopy(model.mlp)
        self.pooling = "max"
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from tensors in the configured observation order."""
        current = self.current_normalizer(torch.cat([inputs[index] for index in self.current_indices], dim=-1))
        history = self.history_normalizer(inputs[self.history_index])
        history = self.history_position(self.history_projection(history))
        causal_mask = torch.triu(
            torch.ones(history.shape[1], history.shape[1], dtype=torch.bool, device=history.device), diagonal=1
        )
        history = self.history_encoder(history, mask=causal_mask)
        history_query = history.max(dim=1, keepdim=True).values
        command = self.command_normalizer(inputs[self.command_index])
        query = self.query_projection(history_query)
        command = self.command_position(self.command_projection(command))
        attended = query + self.cross_attention(self.command_query_norm(query), command, command)
        command_embedding = self.command_output_norm(attended + self.command_ffn(self.command_ffn_norm(attended)))
        latent = torch.cat((current, command_embedding.squeeze(1)), dim=-1)
        return self.deterministic_output(self.mlp(latent))

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op)."""
        pass


class _OnnxRGMTModel(nn.Module):
    """ONNX wrapper exposing named inputs for each selected observation group."""

    def __init__(self, model: RGMTPolicyModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.policy = _TorchRGMTModel(model)
        self.group_names = model.obs_groups
        self.input_shapes = [tuple(model._example_obs_shapes[group]) for group in model.obs_groups]

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from positional group tensors."""
        return self.policy(list(inputs))

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return a single-batch dummy tensor for each observation group."""
        return tuple(torch.zeros((1, *shape[1:])) for shape in self.input_shapes)

    @property
    def input_names(self) -> list[str]:
        """ONNX input names matching the selected observation groups."""
        return self.group_names

    @property
    def output_names(self) -> list[str]:
        """ONNX output name."""
        return ["actions"]
