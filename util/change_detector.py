# utils/change_detector.py
"""
AI-Powered Change Detection with Pretrained UNet Models
Lightweight change detection using segmentation architectures
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
import rasterio
from rasterio.plot import reshape_as_image
import segmentation_models_pytorch as smp
from typing import Tuple, Dict, List, Optional
import joblib
import logging
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightUNet(nn.Module):
    """
    Lightweight UNet for change detection
    """
    def __init__(self, in_channels=6, out_channels=2):
        super(LightweightUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)
        
        # Output
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        pool1 = F.max_pool2d(enc1, kernel_size=2, stride=2)
        
        enc2 = self.enc2(pool1)
        pool2 = F.max_pool2d(enc2, kernel_size=2, stride=2)
        
        enc3 = self.enc3(pool2)
        pool3 = F.max_pool2d(enc3, kernel_size=2, stride=2)
        
        enc4 = self.enc4(pool3)
        
        # Decoder
        up3 = self.up3(enc4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(merge3)
        
        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(merge2)
        
        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(merge1)
        
        # Output
        out = self.final(dec1)
        return out


class PretrainedChangeDetector:
    """
    Change detector using pretrained segmentation models
    """
    
    def __init__(self, model_name: str = 'unet', backbone: str = 'resnet18'):
        """
        Initialize with pretrained model
        
        Args:
            model_name: 'unet', 'deeplab', 'fcn', 'lightweight'
            backbone: Backbone architecture for SMP models
        """
        self.model_name = model_name
        self.backbone = backbone
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = self._get_transforms()
        self.is_trained = False
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Initialized {model_name} with {backbone} on {self.device}")
    
    def _initialize_model(self):
        """Initialize pretrained model"""
        if self.model_name == 'lightweight':
            # Lightweight custom UNet
            self.model = LightweightUNet(in_channels=6, out_channels=2)
            
        elif self.model_name == 'unet':
            # UNet with pretrained encoder from SMP
            self.model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights='imagenet',
                in_channels=6,  # 3 channels for each image
                classes=2,      # Change vs No-change
                activation=None
            )
            
        elif self.model_name == 'deeplab':
            # DeepLabV3 with pretrained ResNet
            self.model = deeplabv3_resnet50(pretrained=True)
            # Modify first layer for 6 channels
            self.model.backbone.conv1 = nn.Conv2d(
                6, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Modify classifier for 2 classes
            self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
            
        elif self.model_name == 'fcn':
            # FCN with pretrained ResNet
            self.model = fcn_resnet50(pretrained=True)
            # Modify first layer for 6 channels
            self.model.backbone.conv1 = nn.Conv2d(
                6, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Modify classifier for 2 classes
            self.model.classifier[4] = nn.Conv2d(512, 2, kernel_size=1)
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _get_transforms(self):
        """Get image transformations"""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def prepare_image_pair(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare image pair for model input
        
        Args:
            image1: First image (H, W, C)
            image2: Second image (H, W, C)
            
        Returns:
            Tensor of shape (1, 6, H, W)
        """
        # Ensure both images have 3 channels
        if len(image1.shape) == 2:
            image1 = np.stack([image1] * 3, axis=-1)
        if len(image2.shape) == 2:
            image2 = np.stack([image2] * 3, axis=-1)
        
        # Resize if needed (to 256x256 for pretrained models)
        if image1.shape[:2] != (256, 256):
            image1 = cv2.resize(image1, (256, 256))
            image2 = cv2.resize(image2, (256, 256))
        
        # Concatenate along channel dimension
        combined = np.concatenate([image1, image2], axis=-1)
        
        # Apply transformations
        transformed = self.transform(image=combined)
        tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return tensor.to(self.device)
    
    def detect_changes(
        self,
        image_before: np.ndarray,
        image_after: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        """
        Detect changes between two images
        
        Args:
            image_before: Image from before period
            image_after: Image from after period
            confidence_threshold: Threshold for binary classification
            
        Returns:
            Dictionary with change masks and probabilities
        """
        # Prepare input
        input_tensor = self.prepare_image_pair(image_before, image_after)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            
            if isinstance(output, dict):
                # For DeepLab/FCN models
                logits = output['out']
            else:
                # For UNet models
                logits = output
            
            # Apply softmax for probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get change probability (class 1)
            change_prob = probs[0, 1].cpu().numpy()
            
            # Resize to original dimensions if needed
            if image_before.shape[:2] != (256, 256):
                change_prob = cv2.resize(
                    change_prob, 
                    (image_before.shape[1], image_before.shape[0])
                )
            
            # Create binary mask
            binary_mask = (change_prob > confidence_threshold).astype(np.uint8) * 255
            
            # Calculate change statistics
            change_stats = self._calculate_change_statistics(
                change_prob, binary_mask, image_before.shape
            )
        
        return {
            'probability': change_prob,
            'binary': binary_mask,
            'statistics': change_stats
        }
    
    def _calculate_change_statistics(
        self,
        change_prob: np.ndarray,
        binary_mask: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> Dict:
        """
        Calculate change statistics
        
        Args:
            change_prob: Change probability map
            binary_mask: Binary change mask
            original_shape: Original image shape
            
        Returns:
            Dictionary of change statistics
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        stats_dict = {
            'change_area_pixels': np.sum(binary_mask > 0),
            'change_area_percentage': (np.sum(binary_mask > 0) / np.prod(original_shape)) * 100,
            'num_change_regions': max(0, num_labels - 1),
            'mean_change_probability': np.mean(change_prob),
            'max_change_probability': np.max(change_prob),
            'change_intensity_distribution': self._calculate_intensity_distribution(change_prob)
        }
        
        # Region-based statistics
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            stats_dict.update({
                'avg_region_area': np.mean(areas),
                'largest_region_area': np.max(areas),
                'region_area_std': np.std(areas),
                'region_centroids': centroids[1:].tolist()
            })
        
        return stats_dict
    
    def _calculate_intensity_distribution(self, change_prob: np.ndarray) -> Dict:
        """Calculate intensity distribution of change probabilities"""
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(change_prob.flatten(), bins=bins)
        total_pixels = np.prod(change_prob.shape)
        percentages = (hist / total_pixels) * 100
        
        return {
            'low_change': percentages[0] + percentages[1],      # 0-0.4
            'moderate_change': percentages[2] + percentages[3], # 0.4-0.8
            'high_change': percentages[4]                       # 0.8-1.0
        }
    
    def detect_ndvi_changes(
        self,
        ndvi_before: np.ndarray,
        ndvi_after: np.ndarray,
        method: str = 'threshold'
    ) -> Dict[str, np.ndarray]:
        """
        Detect vegetation changes using NDVI
        
        Args:
            ndvi_before: NDVI image from before period
            ndvi_after: NDVI image from after period
            method: 'threshold', 'unet', 'gradient'
            
        Returns:
            Dictionary with vegetation change analysis
        """
        if method == 'unet':
            # Use UNet for semantic segmentation of vegetation changes
            return self.detect_changes(ndvi_before, ndvi_after)
        
        elif method == 'threshold':
            # Simple threshold-based approach
            ndvi_diff = ndvi_after - ndvi_before
            
            # Define thresholds for vegetation change
            significant_gain = ndvi_diff > 0.2    # Large increase
            moderate_gain = (ndvi_diff > 0.1) & (ndvi_diff <= 0.2)
            stable = (ndvi_diff >= -0.1) & (ndvi_diff <= 0.1)
            moderate_loss = (ndvi_diff >= -0.2) & (ndvi_diff < -0.1)
            significant_loss = ndvi_diff < -0.2   # Large decrease
            
            # Create classification map
            classification = np.zeros_like(ndvi_diff, dtype=np.uint8)
            classification[significant_gain] = 4  # Dark green
            classification[moderate_gain] = 3     # Light green
            classification[stable] = 2            # Yellow
            classification[moderate_loss] = 1     # Orange
            classification[significant_loss] = 0  # Red
            
            # Calculate statistics
            total_pixels = np.prod(ndvi_diff.shape)
            stats = {
                'significant_gain_percentage': np.sum(significant_gain) / total_pixels * 100,
                'moderate_gain_percentage': np.sum(moderate_gain) / total_pixels * 100,
                'stable_percentage': np.sum(stable) / total_pixels * 100,
                'moderate_loss_percentage': np.sum(moderate_loss) / total_pixels * 100,
                'significant_loss_percentage': np.sum(significant_loss) / total_pixels * 100,
                'mean_ndvi_change': np.mean(ndvi_diff),
                'net_change': np.sum(ndvi_diff)
            }
            
            return {
                'classification': classification,
                'difference': ndvi_diff,
                'statistics': stats
            }
        
        elif method == 'gradient':
            # Gradient-based change detection
            grad_before = cv2.Sobel(ndvi_before, cv2.CV_64F, 1, 1, ksize=3)
            grad_after = cv2.Sobel(ndvi_after, cv2.CV_64F, 1, 1, ksize=3)
            
            grad_diff = np.abs(grad_after - grad_before)
            change_mask = (grad_diff > np.percentile(grad_diff, 90)).astype(np.uint8) * 255
            
            return {
                'gradient_difference': grad_diff,
                'binary': change_mask,
                'change_intensity': grad_diff / np.max(grad_diff) if np.max(grad_diff) > 0 else grad_diff
            }
    
    def analyze_temporal_changes(
        self,
        images: List[np.ndarray],
        timestamps: List[str]
    ) -> Dict:
        """
        Analyze changes over multiple time periods
        
        Args:
            images: List of images over time
            timestamps: Corresponding timestamps
            
        Returns:
            Dictionary with temporal analysis
        """
        if len(images) < 2:
            raise ValueError("Need at least 2 images for temporal analysis")
        
        all_changes = []
        trend_analysis = {
            'change_trend': [],
            'stability_score': 0,
            'volatility_index': 0,
            'change_pattern': None
        }
        
        # Analyze consecutive pairs
        for i in range(len(images) - 1):
            change_result = self.detect_changes(images[i], images[i + 1])
            
            change_info = {
                'period': f"{timestamps[i]}_to_{timestamps[i+1]}",
                'change_percentage': change_result['statistics']['change_area_percentage'],
                'mean_probability': change_result['statistics']['mean_change_probability'],
                'num_regions': change_result['statistics']['num_change_regions']
            }
            all_changes.append(change_info)
            
            trend_analysis['change_trend'].append(
                change_result['statistics']['change_area_percentage']
            )
        
        # Calculate trend metrics
        if len(trend_analysis['change_trend']) > 1:
            changes_array = np.array(trend_analysis['change_trend'])
            trend_analysis['stability_score'] = 100 - np.mean(changes_array)
            trend_analysis['volatility_index'] = np.std(changes_array)
            
            # Identify change pattern
            if np.all(changes_array < 5):
                trend_analysis['change_pattern'] = 'stable'
            elif np.all(np.diff(changes_array) > 0):
                trend_analysis['change_pattern'] = 'accelerating'
            elif np.all(np.diff(changes_array) < 0):
                trend_analysis['change_pattern'] = 'decelerating'
            else:
                trend_analysis['change_pattern'] = 'oscillating'
        
        return {
            'pairwise_changes': all_changes,
            'trend_analysis': trend_analysis,
            'overall_change': np.mean([c['change_percentage'] for c in all_changes])
        }
    
    def create_change_heatmap(
        self,
        change_prob: np.ndarray,
        colormap: str = 'RdYlGn_r'
    ) -> np.ndarray:
        """
        Create heatmap visualization of change probabilities
        
        Args:
            change_prob: Change probability map
            colormap: Matplotlib colormap name
            
        Returns:
            RGB heatmap image
        """
        import matplotlib.pyplot as plt
        
        # Normalize to [0, 1]
        normalized = (change_prob - change_prob.min()) / (change_prob.max() - change_prob.min() + 1e-8)
        
        # Apply colormap
        cmap = plt.cm.get_cmap(colormap)
        heatmap_rgba = cmap(normalized)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        
        return heatmap_rgb
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'backbone': self.backbone,
            'transform_config': self.transform
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.backbone = checkpoint['backbone']
        logger.info(f"Model loaded from {filepath}")


class SatelliteImageProcessor:
    """
    Processor for satellite imagery with pretrained models
    """
    
    def __init__(self, model_type: str = 'unet'):
        self.change_detector = PretrainedChangeDetector(model_name=model_type)
        self.ndvi_processor = NDVIProcessor()
    
    def process_satellite_pair(
        self,
        image1_path: str,
        image2_path: str,
        output_dir: str = './output'
    ) -> Dict:
        """
        Process satellite image pair for change detection
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            output_dir: Output directory
            
        Returns:
            Dictionary with all analysis results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Load images
        img1 = self._load_image(image1_path)
        img2 = self._load_image(image2_path)
        
        # Calculate NDVI if possible
        ndvi_results = {}
        if img1.shape[-1] >= 3 and img2.shape[-1] >= 3:
            ndvi1 = self.ndvi_processor.calculate_ndvi(img1)
            ndvi2 = self.ndvi_processor.calculate_ndvi(img2)
            
            ndvi_results = self.change_detector.detect_ndvi_changes(ndvi1, ndvi2)
            
            # Save NDVI results
            cv2.imwrite(f'{output_dir}/ndvi_before.png', (ndvi1 * 255).astype(np.uint8))
            cv2.imwrite(f'{output_dir}/ndvi_after.png', (ndvi2 * 255).astype(np.uint8))
            cv2.imwrite(f'{output_dir}/ndvi_change_classification.png', 
                       ndvi_results.get('classification', np.zeros_like(ndvi1)))
        
        # Detect general changes
        change_results = self.change_detector.detect_changes(img1, img2)
        
        # Create visualizations
        heatmap = self.change_detector.create_change_heatmap(
            change_results['probability']
        )
        
        # Save outputs
        cv2.imwrite(f'{output_dir}/change_probability.png', 
                   (change_results['probability'] * 255).astype(np.uint8))
        cv2.imwrite(f'{output_dir}/change_binary.png', change_results['binary'])
        cv2.imwrite(f'{output_dir}/change_heatmap.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        
        # Create summary
        summary = {
            'general_changes': change_results['statistics'],
            'vegetation_changes': ndvi_results.get('statistics', {}),
            'image_info': {
                'image1_shape': img1.shape,
                'image2_shape': img2.shape,
                'processing_date': datetime.now().isoformat()
            }
        }
        
        # Save summary as JSON
        import json
        with open(f'{output_dir}/change_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete. Results saved to {output_dir}")
        return summary
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image with proper handling"""
        if image_path.endswith('.tif') or image_path.endswith('.tiff'):
            # Load GeoTIFF
            with rasterio.open(image_path) as src:
                image = src.read()
                image = reshape_as_image(image)
        else:
            # Load regular image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image


class NDVIProcessor:
    """
    NDVI calculation and analysis
    """
    
    @staticmethod
    def calculate_ndvi(image: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI from RGB or multi-band image
        
        Args:
            image: Input image (assumes bands are in standard order)
            
        Returns:
            NDVI array normalized to [-1, 1]
        """
        if len(image.shape) == 2:
            # Already grayscale
            return image.astype(np.float32)
        
        # For RGB images, approximate NDVI using green and red channels
        # This is a simplified version - for real NDVI you need NIR band
        if image.shape[-1] == 3:
            # RGB image - use green and red as approximation
            green = image[:, :, 1].astype(np.float32)
            red = image[:, :, 0].astype(np.float32)
            
            # Avoid division by zero
            denominator = (green + red + 1e-8)
            ndvi = (green - red) / denominator
            
        elif image.shape[-1] >= 4:
            # Multi-band with NIR (typically band 4)
            nir = image[:, :, 3].astype(np.float32)
            red = image[:, :, 0].astype(np.float32)
            
            denominator = (nir + red + 1e-8)
            ndvi = (nir - red) / denominator
        
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        # Normalize to [0, 1] for visualization
        ndvi_normalized = (ndvi + 1) / 2  # [-1, 1] -> [0, 1]
        
        return ndvi_normalized
    
    @staticmethod
    def classify_vegetation(ndvi: np.ndarray) -> np.ndarray:
        """
        Classify vegetation based on NDVI values
        
        Returns:
            Classification map:
            0: Water/No vegetation (NDVI < 0)
            1: Bare soil (0 <= NDVI < 0.2)
            2: Sparse vegetation (0.2 <= NDVI < 0.5)
            3: Moderate vegetation (0.5 <= NDVI < 0.7)
            4: Dense vegetation (NDVI >= 0.7)
        """
        classification = np.zeros_like(ndvi, dtype=np.uint8)
        
        classification[(ndvi >= 0) & (ndvi < 0.2)] = 1      # Bare soil
        classification[(ndvi >= 0.2) & (ndvi < 0.5)] = 2    # Sparse vegetation
        classification[(ndvi >= 0.5) & (ndvi < 0.7)] = 3    # Moderate vegetation
        classification[ndvi >= 0.7] = 4                     # Dense vegetation
        
        return classification


# Example usage function
def example_usage():
    """Example of how to use the change detector"""
    print("Initializing Pretrained Change Detector...")
    
    # Initialize detector
    detector = PretrainedChangeDetector(model_name='unet', backbone='resnet18')
    
    # Create dummy test images
    img1 = np.random.rand(256, 256, 3).astype(np.float32) * 255
    img2 = img1.copy()
    # Add some changes
    img2[100:150, 100:150, :] = np.random.rand(50, 50, 3).astype(np.float32) * 255
    
    # Detect changes
    print("\nDetecting changes...")
    results = detector.detect_changes(img1, img2)
    
    print(f"Change area: {results['statistics']['change_area_percentage']:.2f}%")
    print(f"Number of change regions: {results['statistics']['num_change_regions']}")
    print(f"Mean change probability: {results['statistics']['mean_change_probability']:.3f}")
    
    # Create heatmap
    heatmap = detector.create_change_heatmap(results['probability'])
    cv2.imwrite('example_heatmap.png', cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print("Heatmap saved as 'example_heatmap.png'")
    
    # Process NDVI changes
    print("\nProcessing NDVI changes...")
    ndvi1 = np.random.rand(256, 256).astype(np.float32)
    ndvi2 = ndvi1 + np.random.randn(256, 256) * 0.1  # Add some changes
    
    ndvi_results = detector.detect_ndvi_changes(ndvi1, ndvi2, method='threshold')
    
    if 'statistics' in ndvi_results:
        print(f"Significant vegetation gain: {ndvi_results['statistics']['significant_gain_percentage']:.2f}%")
        print(f"Significant vegetation loss: {ndvi_results['statistics']['significant_loss_percentage']:.2f}%")
    
    print("\nExample complete!")


if __name__ == "__main__":
    example_usage()