"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        dims=[state_dim, *hidden_dims, chunk_size*action_dim]
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net=nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        pred=self.net(state)
        pred=pred.view(-1,self.chunk_size, self.action_dim)
        return nn.functional.mse_loss(pred,action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        pred=self.net(state)
        pred=pred.view(-1,self.chunk_size, self.action_dim)
        return pred


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        input_dim=state_dim+action_dim*chunk_size+1
        output_dim=action_dim*chunk_size
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)    

    def _predict_velocity(
        self,
        state: torch.Tensor,          # (B, state_dim)
        action_chunk: torch.Tensor,   # (B, chunk_size, action_dim)
        tau: torch.Tensor,            # (B, 1)
    ) -> torch.Tensor:
        B = state.shape[0]
        action_flat = action_chunk.view(B, -1)
        x = torch.cat([state, action_flat, tau], dim=1)
        v = self.net(x)
        return v.view(B, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        B = state.shape[0]
        device = state.device

        A_true=action_chunk
        A0=torch.randn_like(A_true)
        tau=torch.rand(B,1,device=device)
        tau_expanded=tau.view(B,1,1)
        A_tau=tau_expanded*A_true+(1-tau_expanded)*A0

        target_velocity=(A_true-A0)
        pred_velocity=self._predict_velocity(state, A_tau, tau)

        return nn.functional.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        B= state.shape[0]
        device=state.device
        
        A= torch.randn(B, self.chunk_size, self.action_dim, device=device)
        dt=1.0/num_steps
        for i in range(num_steps):
            tau=torch.full((B,1), i*dt, device=device)
            v=self._predict_velocity(state, A, tau)
            A=A+v*dt
        return A

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
