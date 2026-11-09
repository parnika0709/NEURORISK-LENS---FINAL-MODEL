"""Loss functions for BraTS segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """Combined Dice + weighted Cross-Entropy loss."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
        class_weights: tuple[float, ...] | None = (
            0.10,
            1.68,
            0.81,
            1.42,
        ),
    ) -> None:
        super().__init__()

        if dice_weight < 0 or ce_weight < 0:
            raise ValueError("Loss weights must be non-negative.")

        if dice_weight + ce_weight <= 0:
            raise ValueError("At least one loss component must be enabled.")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.smooth = smooth

        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)

            if not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Class weights must be finite and positive.")

            self.register_buffer("class_weights", weights)
        else:
            self.register_buffer("class_weights", None)

    def dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate per-class soft Dice loss."""

        probs = torch.softmax(logits, dim=1)
        n_classes = logits.shape[1]

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=n_classes,
        ).float()

        dims = list(range(target_one_hot.ndim))
        channel_dim = dims.pop()
        dims.insert(1, channel_dim)

        target_one_hot = target_one_hot.permute(*dims)

        reduction_dims = [0] + list(range(2, target_one_hot.ndim))

        intersection = (
            probs * target_one_hot
        ).sum(dim=reduction_dims)

        cardinality = (
            probs + target_one_hot
        ).sum(dim=reduction_dims)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            cardinality + self.smooth
        )

        if not self.include_background:
            dice = dice[1:]

        return 1.0 - dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined weighted CE and Dice loss."""

        if logits.ndim < 4:
            raise ValueError(
                f"Expected logits with shape (B,C,...), got {tuple(logits.shape)}"
            )

        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match "
                f"logits spatial shape {tuple(logits.shape[:1] + logits.shape[2:])}"
            )

        if self.class_weights is not None:
            if logits.shape[1] != self.class_weights.numel():
                raise ValueError(
                    f"Expected {self.class_weights.numel()} output classes, "
                    f"got {logits.shape[1]}"
                )

        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=self.class_weights,
        )

        dl = self.dice_loss(logits, target)

        return self.dice_weight * dl + self.ce_weight * ce


class ClassWeightedDiceCELoss(nn.Module):
    """Imbalance-aware loss combining per-class weighted soft Dice and weighted Cross-Entropy.

    Specifically addresses the severe volumetric imbalance between the small enhancing
    tumor core (ET, ~1-5% of tumor volume), necrotic non-enhancing core (NCR/NET),
    larger peritumoral edema (ED), and vast background voxels (>98% of scan).

    Parameters
    ----------
    dice_weight : float
        Scalar multiplier for the Dice loss component.
    ce_weight : float
        Scalar multiplier for the Cross-Entropy loss component.
    include_background : bool
        Whether to compute Dice over background class (index 0). Typically False.
    smooth : float
        Laplace smoothing factor to avoid 0/0 division.
    ce_class_weights : tuple of float, optional
        Weights applied to each class in Cross-Entropy: (bg, ncr, ed, et).
    dice_class_weights : tuple of float, optional
        Per-class weights for soft Dice aggregation. If include_background=False,
        corresponds to foreground classes (ncr, ed, et).
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        include_background: bool = False,
        smooth: float = 1e-5,
        ce_class_weights: tuple[float, ...] | None = (0.10, 1.68, 0.81, 1.42),
        dice_class_weights: tuple[float, ...] | None = (1.5, 0.8, 1.5),
    ) -> None:
        super().__init__()

        if dice_weight < 0 or ce_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
        if dice_weight + ce_weight <= 0:
            raise ValueError("At least one loss component must be enabled.")

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background = include_background
        self.smooth = smooth

        if ce_class_weights is not None:
            ce_w = torch.tensor(ce_class_weights, dtype=torch.float32)
            if not torch.isfinite(ce_w).all() or (ce_w <= 0).any():
                raise ValueError("CE class weights must be finite and positive.")
            self.register_buffer("ce_class_weights", ce_w)
        else:
            self.register_buffer("ce_class_weights", None)

        if dice_class_weights is not None:
            dice_w = torch.tensor(dice_class_weights, dtype=torch.float32)
            if not torch.isfinite(dice_w).all() or (dice_w <= 0).any():
                raise ValueError("Dice class weights must be finite and positive.")
            self.register_buffer("dice_class_weights", dice_w)
        else:
            self.register_buffer("dice_class_weights", None)

    def dice_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate weighted per-class soft Dice loss."""
        probs = torch.softmax(logits, dim=1)
        n_classes = logits.shape[1]

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=n_classes,
        ).float()

        dims = list(range(target_one_hot.ndim))
        channel_dim = dims.pop()
        dims.insert(1, channel_dim)
        target_one_hot = target_one_hot.permute(*dims)

        reduction_dims = [0] + list(range(2, target_one_hot.ndim))

        intersection = (probs * target_one_hot).sum(dim=reduction_dims)
        cardinality = (probs + target_one_hot).sum(dim=reduction_dims)

        per_class_dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if not self.include_background:
            per_class_dice = per_class_dice[1:]

        if self.dice_class_weights is not None:
            if per_class_dice.shape[0] != self.dice_class_weights.numel():
                raise ValueError(
                    f"Dice class weights length ({self.dice_class_weights.numel()}) "
                    f"does not match evaluated classes ({per_class_dice.shape[0]})."
                )
            normalized_w = self.dice_class_weights / self.dice_class_weights.sum()
            weighted_dice = (per_class_dice * normalized_w).sum()
            return 1.0 - weighted_dice

        return 1.0 - per_class_dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined class-imbalance aware loss."""
        if logits.ndim < 4:
            raise ValueError(
                f"Expected logits with shape (B,C,...), got {tuple(logits.shape)}"
            )

        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match "
                f"logits spatial shape {tuple(logits.shape[:1] + logits.shape[2:])}"
            )

        if self.ce_class_weights is not None:
            if logits.shape[1] != self.ce_class_weights.numel():
                raise ValueError(
                    f"Expected {self.ce_class_weights.numel()} output classes, "
                    f"got {logits.shape[1]}"
                )

        ce = F.cross_entropy(
            logits,
            target.long(),
            weight=self.ce_class_weights,
        )

        dl = self.dice_loss(logits, target)

        return self.dice_weight * dl + self.ce_weight * ce
