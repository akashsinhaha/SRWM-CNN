"""
DeepOSWSRM: Deep Feature Collaborative CNN for Water Super-Resolution Mapping
CORRECTED VERSION - Matches Paper Architecture Exactly

Implementation of the method from:
"Super-resolution water body mapping with a feature collaborative CNN model 
by fusing Sentinel-1 and Sentinel-2 images" (Yin et al., 2024)

Key Corrections:
1. Added max-pooling in feature extraction modules
2. Changed skip connections from concatenation to element-wise summation
3. Implemented CUS (Conv + UpSampling) modules
4. Changed from residual blocks to regular conv blocks
5. Corrected multi-scale feature fusion approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv2d → BatchNorm → ReLU
    This is the fundamental building block used throughout the network
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StackedConvBlocks(nn.Module):
    """
    Stacked Convolutional Blocks with Max-Pooling
    
    CORRECTION: Paper uses regular conv blocks (not residual) with max-pooling
    after each block to create multi-scale features.
    
    Args:
        in_channels: Number of input channels
        base_channels: Number of output channels
        num_blocks: Number of stacked conv blocks (default: 5)
    """
    def __init__(self, in_channels, base_channels=64, num_blocks=5):
        super(StackedConvBlocks, self).__init__()
        
        # Initial convolution to transform input to base_channels
        self.conv_in = ConvBlock(in_channels, base_channels)
        
        # Stacked conv blocks - each followed by max-pooling
        self.blocks = nn.ModuleList([
            ConvBlock(base_channels, base_channels) 
            for _ in range(num_blocks)
        ])
        
        # Max pooling reduces spatial dimensions
        self.pools = nn.ModuleList([
            nn.MaxPool2d(2, 2) for _ in range(num_blocks)
        ])
    
    def forward(self, x):
        """
        Forward pass through stacked blocks
        
        Returns:
            x: Final pooled features
            features: List of features before each pooling (for skip connections)
        """
        x = self.conv_in(x)
        
        features = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            features.append(x)  # Save before pooling for skip connections
            x = pool(x)
        
        return x, features


class CUSModule(nn.Module):
    """
    CUS (Conv + UpSampling) Module
    
    CORRECTION: Paper uses Conv followed by bilinear upsampling
    This is used for multi-scale feature fusion in the decoder
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        scale_factor: Upsampling scale factor
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(CUSModule, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.upsample = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=True
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class SpatialChannelAttention(nn.Module):
    """
    Combined Spatial and Channel Attention Module
    Enhances important features while suppressing less relevant ones
    """
    def __init__(self, channels, reduction=16):
        super(SpatialChannelAttention, self).__init__()
        
        # Channel attention components
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        # Spatial attention components
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att
        
        return x


class WaterFractionUnmixing(nn.Module):
    """
    Water Fraction Unmixing Module using Pseudo-Siamese CNN
    
    CORRECTED: Uses StackedConvBlocks with max-pooling, then upsamples back
    to original resolution for fraction estimation.
    
    Architecture:
    - Sentinel-1 → StackedConvBlocks (with pooling)
    - Sentinel-2 → StackedConvBlocks (with pooling)
    - Concatenate → Fusion → Upsample → Fraction prediction
    """
    def __init__(self, sentinel1_channels=2, sentinel2_channels=4, base_channels=64):
        super(WaterFractionUnmixing, self).__init__()
        
        # Feature extractors with max-pooling (5 blocks = 2^5 = 32x downsampling)
        self.s1_extractor = StackedConvBlocks(
            sentinel1_channels, 
            base_channels, 
            num_blocks=5
        )
        self.s2_extractor = StackedConvBlocks(
            sentinel2_channels, 
            base_channels, 
            num_blocks=5
        )
        
        # Fusion network
        self.fusion_conv = nn.Sequential(
            ConvBlock(base_channels * 2, base_channels),
            ConvBlock(base_channels, base_channels)
        )
        
        # Upsample back to original resolution (32x upsampling to match 5 pooling layers)
        self.upsample = nn.Upsample(
            scale_factor=32, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Final fraction prediction with custom activation
        self.fraction_conv = nn.Conv2d(base_channels, 1, kernel_size=1)
    
    def forward(self, sentinel1, sentinel2):
        """
        Forward pass to estimate water fraction
        
        Args:
            sentinel1: Sentinel-1 SAR image [B, 2, H, W]
            sentinel2: Sentinel-2 optical image [B, 4, H, W]
        
        Returns:
            fraction: Water fraction map [B, 1, H, W] with values in [0, 1]
        """
        # Extract features from both sensors (features at 1/32 resolution)
        s1_features, _ = self.s1_extractor(sentinel1)
        s2_features, _ = self.s2_extractor(sentinel2)
        
        # Concatenate features
        fused_features = torch.cat([s1_features, s2_features], dim=1)
        
        # Process fused features
        fused_features = self.fusion_conv(fused_features)
        
        # Upsample to original resolution
        fused_features = self.upsample(fused_features)
        
        # Predict water fraction with activation: [1 + tanh(x)] / 2
        # This ensures output is in range [0, 1]
        fraction = self.fraction_conv(fused_features)
        fraction = (1 + torch.tanh(fraction)) / 2
        
        return fraction


class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net style architecture
    ConvBlock followed by Max Pooling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder block with transpose convolution
    
    CRITICAL CORRECTION: Uses element-wise summation for skip connections
    (not concatenation as in standard U-Net)
    
    This matches the paper's architecture where skip connections are
    added element-wise rather than concatenated.
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.transpose_conv = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )
        self.conv = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip=None):
        """
        Forward pass with element-wise summation
        
        Args:
            x: Input features from previous layer
            skip: Skip connection features from encoder
        
        Returns:
            Output features after upsampling and convolution
        """
        # Transpose convolution for upsampling
        x = self.transpose_conv(x)
        
        # CORRECTED: Element-wise summation (not concatenation!)
        if skip is not None:
            # Ensure dimensions match
            if x.shape != skip.shape:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = x + skip  # Element-wise addition as per paper
        
        # Convolution
        x = self.conv(x)
        
        return x


class SuperResolutionMapping(nn.Module):
    """
    Super-Resolution Water Body Mapping Module
    
    CORRECTED: Encoder-Decoder with element-wise skip connections
    and CUS modules for multi-scale feature fusion
    
    Architecture:
    - Initial upsampling of fraction map to target resolution
    - Encoder path with max-pooling
    - Decoder path with transpose convolutions
    - Element-wise summation for skip connections (CORRECTED)
    - CUS modules for multi-scale fusion
    - Final classification layer
    """
    def __init__(self, in_channels=1, base_channels=64, scale_factor=4):
        super(SuperResolutionMapping, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial upsampling to target resolution
        self.initial_upsample = nn.Upsample(
            scale_factor=scale_factor, 
            mode='bilinear', 
            align_corners=True
        )
        
        # Encoder path
        self.conv_in = ConvBlock(in_channels, base_channels)
        self.enc1 = EncoderBlock(base_channels, base_channels * 2)
        self.enc2 = EncoderBlock(base_channels * 2, base_channels * 4)
        self.enc3 = EncoderBlock(base_channels * 4, base_channels * 8)
        self.enc4 = EncoderBlock(base_channels * 8, base_channels * 16)
        
        # Bottleneck with attention
        self.bottleneck = ConvBlock(base_channels * 16, base_channels * 16)
        self.attention = SpatialChannelAttention(base_channels * 16)
        
        # Decoder path with element-wise summation
        # Decoder path - need 1x1 conv to match dimensions for element-wise addition
        self.dec1 = DecoderBlock(base_channels * 16, base_channels * 8)
        self.skip1_adjust = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=1)

        self.dec2 = DecoderBlock(base_channels * 8, base_channels * 4)
        self.skip2_adjust = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)

        self.dec3 = DecoderBlock(base_channels * 4, base_channels * 2)
        self.skip3_adjust = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)

        self.dec4 = DecoderBlock(base_channels * 2, base_channels)
        self.skip4_adjust = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        
        # CUS modules for multi-scale feature fusion
        # Upsample decoder features to final resolution
        self.cus1 = CUSModule(base_channels * 8, base_channels, scale_factor=8)
        self.cus2 = CUSModule(base_channels * 4, base_channels, scale_factor=4)
        self.cus3 = CUSModule(base_channels * 2, base_channels, scale_factor=2)
        # dec4 is already at full resolution
        
        # Final fusion and classification
        # Input: 4 * base_channels (from 4 decoder levels)
        self.fusion_conv = ConvBlock(base_channels * 4, base_channels)
        self.final_conv = nn.Conv2d(base_channels, 2, kernel_size=1)
    
    def forward(self, fraction):
        """
        Forward pass for super-resolution mapping
        
        Args:
            fraction: Water fraction map [B, 1, H, W]
        
        Returns:
            Water map logits [B, 2, H*scale, W*scale]
        """
        # Upsample fraction to target resolution
        x = self.initial_upsample(fraction)
        
        # Encoder path - save skip connections
        x = self.conv_in(x)
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        
        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.attention(x)
        
        # Decoder path with element-wise summation of skip connections
        # Decoder path - adjust skip dimensions then add
        x = self.dec1(x, self.skip1_adjust(skip4))
        dec1_out = x

        x = self.dec2(x, self.skip2_adjust(skip3))
        dec2_out = x

        x = self.dec3(x, self.skip3_adjust(skip2))
        dec3_out = x

        x = self.dec4(x, self.skip4_adjust(skip1))
        dec4_out = x
        
        # Multi-scale feature fusion using CUS modules
        # Bring all decoder outputs to the same (final) resolution
        ms1 = self.cus1(dec1_out)  # 8x upsampling
        ms2 = self.cus2(dec2_out)  # 4x upsampling
        ms3 = self.cus3(dec3_out)  # 2x upsampling
        ms4 = dec4_out             # Already at target resolution
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([ms1, ms2, ms3, ms4], dim=1)
        
        # Final fusion and classification
        fused = self.fusion_conv(multi_scale)
        output = self.final_conv(fused)
        
        return output


class DeepOSWSRM(nn.Module):
    """
    Complete DeepOSWSRM Model - CORRECTED VERSION
    
    Two-stage architecture:
    1. Water Fraction Unmixing: Estimates coarse water fraction from S1+S2
    2. Super-Resolution Mapping: Generates fine-resolution water map
    
    Args:
        sentinel1_channels: Number of Sentinel-1 channels (default: 2 for VV, VH)
        sentinel2_channels: Number of Sentinel-2 channels (default: 4 for B,G,R,NIR)
        scale_factor: Super-resolution scale factor (2, 4, or 6)
        base_channels: Base number of channels in the network
    
    Key Corrections from Original Implementation:
    - Added max-pooling in feature extraction
    - Changed skip connections to element-wise summation
    - Implemented CUS modules for upsampling
    - Corrected multi-scale feature fusion
    """
    def __init__(self, sentinel1_channels=2, sentinel2_channels=4, 
                 scale_factor=4, base_channels=64):
        super(DeepOSWSRM, self).__init__()
        self.scale_factor = scale_factor
        
        # Water fraction unmixing module (Pseudo-Siamese CNN)
        self.unmixing = WaterFractionUnmixing(
            sentinel1_channels=sentinel1_channels,
            sentinel2_channels=sentinel2_channels,
            base_channels=base_channels
        )
        
        # Super-resolution mapping module (Encoder-Decoder)
        self.srm = SuperResolutionMapping(
            in_channels=1,
            base_channels=base_channels,
            scale_factor=scale_factor
        )
    
    def forward(self, sentinel1, sentinel2):
        """
        Forward pass through the complete model
        
        Args:
            sentinel1: Sentinel-1 SAR image [B, 2, H, W]
            sentinel2: Sentinel-2 optical image [B, 4, H, W]
        
        Returns:
            water_fraction: Coarse-resolution water fraction [B, 1, H, W]
            water_map: Fine-resolution water map logits [B, 2, H*scale, W*scale]
        """
        # Stage 1: Estimate water fraction
        water_fraction = self.unmixing(sentinel1, sentinel2)
        
        # Stage 2: Super-resolution mapping
        water_map = self.srm(water_fraction)
        
        return water_fraction, water_map
    
    def predict(self, sentinel1, sentinel2):
        """
        Prediction mode with softmax activation
        
        Returns:
            water_fraction: Water fraction map [B, 1, H, W]
            water_map_probs: Probability map for water class [B, 1, H*scale, W*scale]
            water_map_binary: Binary water map [B, 1, H*scale, W*scale]
        """
        self.eval()
        with torch.no_grad():
            water_fraction, water_map_logits = self.forward(sentinel1, sentinel2)
            water_map_probs = F.softmax(water_map_logits, dim=1)
            water_map_binary = torch.argmax(water_map_probs, dim=1, keepdim=True)
        
        return water_fraction, water_map_probs[:, 1:2], water_map_binary


# Loss functions
class AdaptiveFractionCrossEntropyLoss(nn.Module):
    """
    Adaptive fraction-based cross-entropy loss (Equation 5 in paper)
    
    Higher weights for pixels with smaller water fractions,
    making the model focus more on boundary regions.
    
    Loss = -Σ [exp(η*f) * (m*log(p) + (1-m)*log(1-p))]
    where:
        η = eta parameter (default: -0.5)
        f = water fraction
        m = ground truth label
        p = predicted probability
    """
    def __init__(self, eta=-0.5):
        super(AdaptiveFractionCrossEntropyLoss, self).__init__()
        self.eta = eta
    
    def forward(self, predictions, targets, fractions):
        """
        Args:
            predictions: Model predictions [B, 2, H, W]
            targets: Ground truth binary maps [B, 1, H, W]
            fractions: Water fraction values [B, 1, H, W]
        
        Returns:
            Adaptive cross-entropy loss
        """
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        water_prob = probs[:, 1:2]  # Probability of water class
        
        # Adaptive weight based on fraction: exp(η * f)
        # Lower fractions (boundaries) get higher weights
        weight = torch.exp(self.eta * fractions)
        
        # Binary cross-entropy with adaptive weight
        targets = targets.float()
        loss = -weight * (
            targets * torch.log(water_prob + 1e-7) + 
            (1 - targets) * torch.log(1 - water_prob + 1e-7)
        )
        
        return loss.mean()


class DeepOSWSRMLoss(nn.Module):
    """
    Combined loss function for DeepOSWSRM
    
    L_total = L_frac + λ * L_SRM
    
    Where:
        L_frac: MSE loss for water fraction estimation
        L_SRM: Adaptive cross-entropy loss for super-resolution mapping
        λ: Weight balancing the two losses (default: 1.0)
    """
    def __init__(self, lambda_weight=1.0, eta=-0.5):
        super(DeepOSWSRMLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.mse_loss = nn.MSELoss()
        self.adaptive_ce_loss = AdaptiveFractionCrossEntropyLoss(eta=eta)
    
    def forward(self, pred_fraction, pred_map, target_fraction, target_map):
        """
        Compute combined loss
        
        Args:
            pred_fraction: Predicted water fraction [B, 1, H, W]
            pred_map: Predicted fine-resolution water map [B, 2, H', W']
            target_fraction: Target water fraction [B, 1, H, W]
            target_map: Target fine-resolution water map [B, 1, H', W']
        
        Returns:
            total_loss: Combined loss
            loss_frac: Fraction loss component
            loss_srm: SRM loss component
        """
        # Fraction loss (MSE)
        loss_frac = self.mse_loss(pred_fraction, target_fraction)
        
        # SRM loss (Adaptive CE)
        # Interpolate predicted fraction to fine resolution for adaptive weighting
        pred_fraction_fine = F.interpolate(
            pred_fraction, 
            size=target_map.shape[2:],
            mode='bilinear',
            align_corners=True
        )
        loss_srm = self.adaptive_ce_loss(pred_map, target_map, pred_fraction_fine)
        
        # Combined loss
        total_loss = loss_frac + self.lambda_weight * loss_srm
        
        return total_loss, loss_frac, loss_srm


if __name__ == "__main__":
    """Test the corrected model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing CORRECTED DeepOSWSRM Model")
    print(f"Using device: {device}\n")
    
    # Create model with scale factor 4
    model = DeepOSWSRM(
        sentinel1_channels=2,
        sentinel2_channels=4,
        scale_factor=4,
        base_channels=64
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")
    
    # Test forward pass
    print("Testing forward pass...")
    batch_size = 2
    height, width = 64, 64
    
    sentinel1 = torch.randn(batch_size, 2, height, width).to(device)
    sentinel2 = torch.randn(batch_size, 4, height, width).to(device)
    
    print(f"Input shapes:")
    print(f"  Sentinel-1: {sentinel1.shape}")
    print(f"  Sentinel-2: {sentinel2.shape}")
    
    water_fraction, water_map = model(sentinel1, sentinel2)
    
    print(f"\nOutput shapes:")
    print(f"  Water fraction: {water_fraction.shape}")
    print(f"  Water map logits: {water_map.shape}")
    
    expected_height = height * 4
    expected_width = width * 4
    assert water_map.shape == (batch_size, 2, expected_height, expected_width), \
        f"Expected shape {(batch_size, 2, expected_height, expected_width)}, got {water_map.shape}"
    
    print("\n✓ Forward pass successful!")
    
    # Test loss function
    print("\nTesting loss function...")
    target_fraction = torch.rand(batch_size, 1, height, width).to(device)
    target_map = torch.randint(0, 2, (batch_size, 1, height*4, width*4)).to(device)
    
    loss_fn = DeepOSWSRMLoss(lambda_weight=1.0, eta=-0.5)
    total_loss, loss_frac, loss_srm = loss_fn(
        water_fraction, water_map, target_fraction, target_map
    )
    
    print(f"Loss values:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Fraction loss: {loss_frac.item():.4f}")
    print(f"  SRM loss: {loss_srm.item():.4f}")
    
    print("\n✓ Loss computation successful!")
    
    # Test prediction mode
    print("\nTesting prediction mode...")
    pred_fraction, pred_prob, pred_binary = model.predict(sentinel1, sentinel2)
    
    print(f"Prediction outputs:")
    print(f"  Fraction: {pred_fraction.shape}")
    print(f"  Probability: {pred_prob.shape}")
    print(f"  Binary map: {pred_binary.shape}")
    print(f"  Fraction range: [{pred_fraction.min():.3f}, {pred_fraction.max():.3f}]")
    print(f"  Probability range: [{pred_prob.min():.3f}, {pred_prob.max():.3f}]")
    print(f"  Binary values: {pred_binary.unique()}")
    
    print("\n✓ Prediction mode successful!")
    
    # Architecture summary
    print("\n" + "="*60)
    print("KEY CORRECTIONS FROM ORIGINAL IMPLEMENTATION:")
    print("="*60)
    print("✓ Added max-pooling in StackedConvBlocks")
    print("✓ Changed skip connections to element-wise summation")
    print("✓ Implemented CUS (Conv + UpSampling) modules")
    print("✓ Corrected multi-scale feature fusion")
    print("✓ Changed from residual blocks to regular conv blocks")
    print("="*60)
    print("\nModel is ready for training!")
    print("Use this corrected model for better results matching the paper.")