import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
import timm

from utils.generate import generate_perlin_noise
from model.filter import GaussianBlur2d


class SimpleNetLoss(nn.Module):
    """
    Computes the combined loss for SimpleNet, including focal loss for anomaly map
    and score, and a truncated L1 loss for the anomaly map.
    """

    def __init__(self, truncation_term: float = 0.5) -> None:
        """
        Initializes the SimpleNetLoss module.

        Args:
            truncation_term (float): The truncation threshold for the L1 loss.
        """
        super().__init__()
        # Initialize focal loss with specified alpha and gamma, and mean reduction.
        self.focal_loss = partial(sigmoid_focal_loss, alpha=-1, gamma=4.0, reduction="mean")
        self.th = truncation_term  # Truncation term for L1 loss

    def trunc_l1_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the truncated L1 loss.
        This loss penalizes predictions that are too high for normal regions
        and too low for anomalous regions, up to a truncation threshold.

        Args:
            pred (torch.Tensor): Predicted anomaly scores/map.
            target (torch.Tensor): Ground truth mask/labels (0 for normal, >0 for anomalous).

        Returns:
            torch.Tensor: The calculated truncated L1 loss.
        """
        # Separate scores for normal and anomalous regions
        normal_scores = pred[target == 0]
        anomalous_scores = pred[target > 0]

        # Calculate true_loss: penalize normal scores if they exceed -self.th
        true_loss = torch.clip(normal_scores + self.th, min=0)
        # Calculate fake_loss: penalize anomalous scores if they fall below self.th
        fake_loss = torch.clip(-anomalous_scores + self.th, min=0)

        # Compute mean losses, handling empty tensors
        true_loss_mean = true_loss.mean() if true_loss.numel() > 0 else torch.tensor(0.0, device=pred.device)
        fake_loss_mean = fake_loss.mean() if fake_loss.numel() > 0 else torch.tensor(0.0, device=pred.device)

        return true_loss_mean + fake_loss_mean

    def forward(
        self,
        pred_map: torch.Tensor,
        pred_score: torch.Tensor,
        target_mask: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the loss calculation.

        Args:
            pred_map (torch.Tensor): Predicted anomaly map.
            pred_score (torch.Tensor): Predicted anomaly score (image-level).
            target_mask (torch.Tensor): Ground truth anomaly mask.
            target_label (torch.Tensor): Ground truth image-level label.

        Returns:
            torch.Tensor: The total calculated loss.
        """
        # Calculate focal loss for the anomaly map
        map_focal = self.focal_loss(pred_map, target_mask)
        # Calculate truncated L1 loss for the anomaly map
        map_trunc_l1 = self.trunc_l1_loss(pred_map, target_mask)
        # Calculate focal loss for the anomaly score
        pred_score = pred_score.squeeze(dim=-1)
        score_focal = self.focal_loss(pred_score, target_label)

        return map_focal + map_trunc_l1 + score_focal


class AnomalyMapGenerator(nn.Module):
    """
    Generates and refines the final anomaly map by interpolating
    and applying Gaussian smoothing.
    """

    def __init__(self, sigma: float) -> None:
        """
        Initializes the AnomalyMapGenerator module.

        Args:
            sigma (float): Standard deviation for the Gaussian blur kernel.
        """
        super().__init__()
        # Calculate kernel size based on sigma for Gaussian blur
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.smoothing = GaussianBlur2d(kernel_size=kernel_size, sigma=sigma)

    def forward(self, out_map: torch.Tensor, final_size: tuple[int, int]) -> torch.Tensor:
        """
        Forward pass for generating the anomaly map.

        Args:
            out_map (torch.Tensor): Raw anomaly map output from the network.
            final_size (tuple[int, int]): Desired final height and width of the anomaly map.

        Returns:
            torch.Tensor: Smoothed and resized anomaly map.
        """
        # Upsample the raw anomaly map to the desired final size
        anomaly_map = F.interpolate(out_map, size=final_size, mode="bilinear", align_corners=False)
        # Apply Gaussian smoothing to the anomaly map
        return self.smoothing(anomaly_map)


class AnomalyGenerator(nn.Module):
    """
    Generates synthetic anomalies by adding Perlin noise and Gaussian noise
    to features, modifying masks and labels accordingly.
    """

    def __init__(self, noise_mean: float, noise_std: float, threshold: float) -> None:
        """
        Initializes the AnomalyGenerator module.

        Args:
            noise_mean (float): Mean of the Gaussian noise.
            noise_std (float): Standard deviation of the Gaussian noise.
            threshold (float): Threshold for binarizing Perlin noise.
        """
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.threshold = threshold

    @staticmethod
    def next_power_2(num: int) -> int:
        """
        Calculates the next power of 2 greater than or equal to `num`.
        Used for Perlin noise generation dimensions.

        Args:
            num (int): Input number.

        Returns:
            int: The smallest power of 2 greater than or equal to `num`.
        """
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches: int, height: int, width: int) -> torch.Tensor:
        """
        Generates a batch of thresholded Perlin noise masks.

        Args:
            batches (int): Number of Perlin noise masks to generate.
            height (int): Desired height of the Perlin noise mask.
            width (int): Desired width of the Perlin noise mask.

        Returns:
            torch.Tensor: Stacked tensor of binarized Perlin noise masks.
        """
        perlin_masks = []
        for _ in range(batches):
            perlin_height = self.next_power_2(height)
            perlin_width = self.next_power_2(width)
            # Generate raw Perlin noise
            perlin_noise = generate_perlin_noise(height=perlin_height, width=perlin_width)
            # Resize Perlin noise to target dimensions
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, perlin_height, perlin_width),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            # Binarize Perlin noise based on threshold
            thresholded_perlin = torch.where(perlin_noise > self.threshold, 1.0, 0.0)

            # Randomly set some generated Perlin masks to all zeros (no anomaly)
            if torch.rand(1).item() > 0.5:
                thresholded_perlin = torch.zeros_like(thresholded_perlin)

            perlin_masks.append(thresholded_perlin)
        return torch.cat(perlin_masks)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for anomaly generation.
        Doubles the batch size, adds synthetic noise, and updates masks/labels.

        Args:
            features (torch.Tensor): Input feature maps.
            mask (torch.Tensor): Ground truth anomaly masks.
            labels (torch.Tensor): Ground truth image-level labels.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - perturbed (torch.Tensor): Feature maps with synthetic anomalies.
                - mask (torch.Tensor): Updated anomaly masks including synthetic anomalies.
                - labels (torch.Tensor): Updated image-level labels including synthetic anomalies.
        """
        b, c, h, w = features.shape

        # Duplicate inputs to create a batch for synthetic anomaly injection
        features = torch.cat((features, features))
        mask = torch.cat((mask, mask))
        labels = torch.cat((labels, labels))

        # Generate Gaussian noise
        noise = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=features.shape,
            device=features.device,
            requires_grad=False,
        )

        # Create a base noise mask (initially all ones)
        noise_mask = torch.ones(
            b * 2, 1, h, w, device=features.device, requires_grad=False
        )

        # Apply existing mask to noise_mask (prevent noise in original anomaly regions)
        noise_mask *= (1 - mask)  # Only apply noise where original mask is 0 (normal)

        # Generate Perlin noise masks
        perlin_mask = self.generate_perlin(b * 2, h, w).to(features.device)
        # Combine noise_mask with Perlin mask to define areas for synthetic anomalies
        noise_mask *= perlin_mask

        # Update the overall mask with newly generated noise regions
        mask = mask + noise_mask
        mask = torch.where(mask > 0, 1.0, 0.0) # Binarize the mask

        # Update labels: if any new anomaly was introduced, mark as anomalous
        # Reshape noise_mask to (batch_size * 2, -1) and check if any pixel is 1
        new_anomalous = noise_mask.reshape(b * 2, -1).any(dim=1).type(torch.float32)
        labels = labels + new_anomalous
        labels = torch.where(labels > 0, 1.0, 0.0) # Binarize the labels

        # Perturb features with generated noise in defined regions
        perturbed = features + noise * noise_mask

        return perturbed, mask, labels


class SuperSimpleNet(torch.nn.Module):
    """
    SuperSimpleNet for unsupervised anomaly detection.
    It utilizes a pre-trained backbone for feature extraction,
    generates synthetic anomalies during training, and predicts
    both an anomaly map and an image-level anomaly score.
    """

    def __init__(self, patch_size: int = 3, out_indices: tuple[int, ...] = (2, 3), sigma: float = 4.0) -> None:
        """
        Initializes the SuperSimpleNet model.

        Args:
            patch_size (int): Kernel size for the feature aggregation AvgPool2d.
            out_indices (tuple[int, ...]): Indices of feature map layers to extract from the backbone.
            sigma (float): Sigma for Gaussian smoothing in AnomalyMapGenerator.
        """
        super().__init__()
        # Initialize backbone using timm library
        self.backbone = timm.create_model(
            'wide_resnet50_2.tv_in1k',
            pretrained=True,
            features_only=True,
            exportable=True,
            out_indices=out_indices,
        )
        self.channels = self.backbone.feature_info.channels()
        self.scales = self.backbone.feature_info.reduction()

        # Feature aggregation using AvgPool2d
        self.feature_aggregator = nn.AvgPool2d(
            kernel_size=patch_size, padding=patch_size // 2, stride=1
        )
        in_channels = sum(self.channels)

        # Feature adapter (1x1 convolution)
        self.feature_adapter = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # Anomaly generator for synthetic anomaly creation during training
        self.generator = AnomalyGenerator(noise_mean=0.0, noise_std=0.015, threshold=0.2)

        # Segmentation head for anomaly map prediction
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1, kernel_size=1, padding=0, stride=1, bias=False),
        )

        # Adaptive pooling layers for classification head inputs
        self.map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dec_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Classification head for image-level anomaly score prediction
        # Input channels: aggregated features + 1 (from anomaly map)
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels + 1, 128, kernel_size=5, padding="same", stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # Fully connected layer for the final anomaly score
        # Input features: 2 (from map pools) + 2 (from decoder pools)
        self.cls_fc = nn.Linear(128 * 2 + 2, 1)

        # Anomaly map post-processing (smoothing and resizing)
        self.map_generator = AnomalyMapGenerator(sigma=sigma)
        # Loss function
        self.loss = SimpleNetLoss()

    @staticmethod
    def downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        masks = masks.type(torch.float32)
        masks = masks.unsqueeze(1)
        masks = F.interpolate(
            masks,
            size=(feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )
        # Binarize the interpolated mask (values < 0.5 become 0, otherwise 1)
        
        return torch.where(masks < 0.5, 0.0, 1.0)

    def forward(self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the SimpleNet model.

        Args:
            x (torch.Tensor): Input image tensor.
            masks (torch.Tensor | None): Ground truth anomaly masks (for training).
            labels (torch.Tensor | None): Ground truth image-level labels (for training).

        Returns:
            Union[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                tuple[torch.Tensor, torch.Tensor]
            ]:
                - During training: (total_loss, predicted_anomaly_map, predicted_anomaly_score)
                - During evaluation: (final_anomaly_map, predicted_anomaly_score)
        """
        # Store original image size for final anomaly map resizing
        image_size = x.shape[2:]

        # Extract features from the backbone (in eval mode, no_grad)
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)

        # Get spatial dimensions from the first feature map
        h0, w0 = features[0].shape[2:]
        
        # Upsample and concatenate features from different layers
        features = torch.cat(
            [F.interpolate(f, size=(h0 * 2, w0 * 2), mode='bilinear', align_corners=False) for f in features], dim=1
        )

        features = self.feature_aggregator(features)
        features = self.feature_adapter(features)
        anomaly_map = self.seg_head(features)

        if self.training:
            # Downsample ground truth masks to feature map resolution
            masks = self.downsample_mask(masks, *features.shape[2:])
            # Ensure labels are float32
            if labels is not None:
                labels = labels.type(torch.float32)

            # Generate synthetic anomalies and update features, masks, and labels
            features, masks, labels = self.generator(features, masks, labels)

            # Re-run segmentation head with perturbed features to get map for loss
            anomaly_map_for_loss = self.seg_head(features)
            map_dec_copy = anomaly_map_for_loss.detach()

            mask_cat = torch.cat((features, map_dec_copy), dim=1)
            dec_out = self.cls_head(mask_cat)

            dec_max = self.dec_max_pool(dec_out)
            dec_avg = self.dec_avg_pool(dec_out)

            map_max = self.map_max_pool(anomaly_map_for_loss).detach()
            map_avg = self.map_avg_pool(anomaly_map_for_loss).detach()

            dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(dim=-1).squeeze(dim=-1)
            ano_score = self.cls_fc(dec_cat)

            total_loss = self.loss(anomaly_map_for_loss, ano_score, masks, labels)
            return total_loss

        else:
            map_dec_copy = anomaly_map.detach()

            mask_cat = torch.cat((features, map_dec_copy), dim=1)
            dec_out = self.cls_head(mask_cat)
            dec_max = self.dec_max_pool(dec_out)
            dec_avg = self.dec_avg_pool(dec_out)

            map_max = self.map_max_pool(anomaly_map).detach()
            map_avg = self.map_avg_pool(anomaly_map).detach()
            dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(dim=-1).squeeze(dim=-1)
            ano_score = self.cls_fc(dec_cat)
            
            # Post-process the anomaly map: resize and smooth
            final_anomaly_map = self.map_generator(anomaly_map, image_size)
            
            return ano_score, final_anomaly_map