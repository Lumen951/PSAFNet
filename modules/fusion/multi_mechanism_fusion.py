import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from PSAFNet import CrossAttention, TCNLayer


class MultiMechanismFusion(nn.Module):
    """
    Multi-Mechanism Fusion Module

    Inspired by TSception's multi-scale design, enhanced with three
    complementary fusion mechanisms:
    1. Cross-Attention (from PSAFNet) - Inter-pathway information flow
    2. Gated Fusion - Adaptive pathway selection
    3. Adaptive Weighted Fusion - Learnable pathway balancing

    Args:
        input_dim (int): Input feature dimension from each pathway
        hidden_dim (int): Hidden dimension for TCN layers
        output_dim (int): Output classes (default: 2)
        num_tcn_layers (int): Number of TCN layers (default: 3)
        mmd_sigma (float): Bandwidth for MMD loss (default: 1.0)
    """
    def __init__(self, input_dim, hidden_dim, output_dim=2,
                 num_tcn_layers=3, mmd_sigma=1.0):
        super(MultiMechanismFusion, self).__init__()

        self.input_dim = input_dim
        self.mmd_sigma = mmd_sigma

        # Mechanism 1: Cross-Attention (bidirectional)
        self.cross_attention_fast = CrossAttention(input_dim)
        self.cross_attention_slow = CrossAttention(input_dim)

        # Mechanism 2: Gated Fusion
        self.gate_fast = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.gate_slow = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        # Mechanism 3: Adaptive Weighted Fusion
        # Initialize at 0.5 (equal weight), learnable during training
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))

        # Fusion projection: 3 mechanisms concatenated
        fusion_input_dim = input_dim * 3

        # TCN layers for temporal modeling
        self.tcn_layers = nn.ModuleList()
        for i in range(num_tcn_layers):
            in_dim = fusion_input_dim if i == 0 else hidden_dim
            dilation = 2 ** i
            self.tcn_layers.append(
                TCNLayer(in_dim, hidden_dim, kernel_size=3, dilation=dilation)
            )

        # Classifier head
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, fast_features, slow_features):
        """
        Forward pass with multi-mechanism fusion

        Args:
            fast_features: [batch, feature_dim, 1, time] from fast pathway
            slow_features: [batch, feature_dim, 1, time] from slow pathway

        Returns:
            output: [batch, num_classes] classification logits
            mmd_loss: scalar, distribution alignment loss
        """
        batch_size, feature_dim, _, time_points = fast_features.size()

        # Handle different time dimensions from fast and slow pathways
        # Align to the minimum time dimension
        min_time = min(fast_features.size(3), slow_features.size(3))
        fast_feat = fast_features[:, :, :, :min_time]
        slow_feat = slow_features[:, :, :, :min_time]

        # Reshape: [B, C, 1, T] -> [B, T, C]
        fast_feat = fast_feat.squeeze(2).permute(0, 2, 1)
        slow_feat = slow_feat.squeeze(2).permute(0, 2, 1)

        # === Mechanism 1: Bidirectional Cross-Attention ===
        # Fast pathway attends to slow pathway
        fast_att = self.cross_attention_fast(fast_feat, slow_feat)
        # Slow pathway attends to fast pathway
        slow_att = self.cross_attention_slow(slow_feat, fast_feat)
        # Combine with residual connections
        fused_1 = fast_att + slow_att  # [B, T, C]

        # === Mechanism 2: Gated Fusion ===
        # Compute global statistics for gate generation
        fast_avg = fast_feat.mean(dim=1)  # [B, C]
        slow_avg = slow_feat.mean(dim=1)  # [B, C]
        gate_f = self.gate_fast(fast_avg).unsqueeze(1)  # [B, 1, C]
        gate_s = self.gate_slow(slow_avg).unsqueeze(1)  # [B, 1, C]
        fused_2 = gate_f * fast_feat + gate_s * slow_feat  # [B, T, C]

        # === Mechanism 3: Adaptive Weighted Fusion ===
        # Constrain weight to [0, 1] via sigmoid
        alpha = torch.sigmoid(self.adaptive_weight)
        fused_3 = alpha * fast_feat + (1 - alpha) * slow_feat  # [B, T, C]

        # === Concatenate all fusion mechanisms ===
        multi_fused = torch.cat([fused_1, fused_2, fused_3], dim=2)  # [B, T, 3C]

        # === TCN processing ===
        # [B, T, 3C] -> [B, 3C, T]
        multi_fused = multi_fused.permute(0, 2, 1)

        for tcn_layer in self.tcn_layers:
            multi_fused = tcn_layer(multi_fused)  # [B, hidden, T]

        # Global average pooling over time
        multi_fused = multi_fused.mean(dim=2)  # [B, hidden]

        # Classification
        output = self.fc(multi_fused)  # [B, num_classes]

        # Compute MMD loss for distribution alignment
        mmd_loss = self.compute_mmd_loss(fast_feat, slow_feat)

        return output, mmd_loss

    def compute_mmd_loss(self, feature1, feature2):
        """
        Compute Maximum Mean Discrepancy (MMD) loss

        MMD measures the distance between two distributions using
        kernel methods in reproducing kernel Hilbert space (RKHS).

        Args:
            feature1: [batch, time, features] from fast pathway
            feature2: [batch, time, features] from slow pathway

        Returns:
            mmd_loss: scalar
        """
        # Compute pairwise distances
        diff1 = feature1.unsqueeze(1) - feature1.unsqueeze(0)  # [B, B, T, C]
        diff2 = feature2.unsqueeze(1) - feature2.unsqueeze(0)
        diff_cross = feature1.unsqueeze(1) - feature2.unsqueeze(0)

        # Squared Euclidean distance
        dist1 = torch.sum(diff1 ** 2, dim=-1)  # [B, B, T]
        dist2 = torch.sum(diff2 ** 2, dim=-1)
        dist_cross = torch.sum(diff_cross ** 2, dim=-1)

        # Gaussian RBF kernel
        K_XX = torch.exp(-dist1 / (2 * self.mmd_sigma ** 2))
        K_YY = torch.exp(-dist2 / (2 * self.mmd_sigma ** 2))
        K_XY = torch.exp(-dist_cross / (2 * self.mmd_sigma ** 2))

        # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        mmd_loss = torch.mean(K_XX + K_YY - 2 * K_XY)

        return mmd_loss


if __name__ == "__main__":
    # Quick test
    fusion = MultiMechanismFusion(input_dim=16, hidden_dim=24, output_dim=2)

    fast_feat = torch.randn(4, 16, 1, 25)
    slow_feat = torch.randn(4, 16, 1, 25)

    output, mmd_loss = fusion(fast_feat, slow_feat)

    print(f"Fast features: {fast_feat.shape}")
    print(f"Slow features: {slow_feat.shape}")
    print(f"Output: {output.shape}")
    print(f"MMD loss: {mmd_loss.item():.4f}")

    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"Total parameters: {total_params:,}")
