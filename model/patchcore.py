from abc import ABC

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import timm
from tqdm import tqdm

from sklearn.utils.random import sample_without_replacement

from model.filter import GaussianBlur2d

class TimmFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone: str | nn.Module,
        layers: list[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad
        if isinstance(backbone, str):
            self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            self.out_dims = self.feature_extractor.feature_info.channels()

        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self) -> list[int]:
        """Map layer names to their indices in the model's output.

        Returns:
            list[int]: Indices corresponding to the requested layer names.

        Note:
            If a requested layer is not found in the model, it is removed from
            ``self.layers`` and a warning is logged.
        """
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:
                msg = f"Layer {layer} not found in model {self.backbone}. Available layers: {layer_names}"
                self.layers.remove(layer)

        return idx

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.requires_grad:
            features = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(inputs)
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))
        return features

class AnomalyMapGenerator(nn.Module):
    def __init__(
        self,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]))
        return self.blur(anomaly_map)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        return self.compute_anomaly_map(patch_scores, image_size)


class SparseRandomProjection:

    def __init__(self, eps: float = 0.1, random_state: int | None = None) -> None:
        self.n_components: int
        self.sparse_random_matrix: torch.Tensor
        self.eps = eps
        self.random_state = random_state

    def _sparse_random_matrix(self, n_features: int) -> torch.Tensor:
        density = 1 / np.sqrt(n_features)

        if density == 1:
            # skip index generation if totally dense
            binomial = torch.distributions.Binomial(total_count=1, probs=0.5)
            components = binomial.sample((self.n_components, n_features)) * 2 - 1
            components = 1 / np.sqrt(self.n_components) * components

        else:
            # Sparse matrix is not being generated here as it is stored as dense anyways
            components = torch.zeros((self.n_components, n_features), dtype=torch.float32)
            for i in range(self.n_components):
                # find the indices of the non-zero components for row i
                nnz_idx = torch.distributions.Binomial(total_count=n_features, probs=density).sample()
                # get nnz_idx column indices
                # pylint: disable=not-callable
                c_idx = torch.tensor(
                    sample_without_replacement(
                        n_population=n_features,
                        n_samples=nnz_idx,
                        random_state=self.random_state,
                    ),
                    dtype=torch.int32,
                )
                data = torch.distributions.Binomial(total_count=1, probs=0.5).sample(sample_shape=c_idx.size()) * 2 - 1
                # assign data to only those columns
                components[i, c_idx] = data

            components *= np.sqrt(1 / density) / np.sqrt(self.n_components)

        return components

    @staticmethod
    def _johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.1) -> int | np.integer:
        denominator = (eps**2 / 2) - (eps**3 / 3)
        return (4 * np.log(n_samples) / denominator).astype(np.int64)

    def fit(self, embedding: torch.Tensor) -> "SparseRandomProjection":
        n_samples, n_features = embedding.shape
        device = embedding.device

        self.n_components = self._johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=self.eps)

        # Generate projection matrix
        # torch can't multiply directly on sparse matrix and moving sparse matrix to cuda throws error
        # (Could not run 'aten::empty_strided' with arguments from the 'SparseCsrCUDA' backend)
        # hence sparse matrix is stored as a dense matrix on the device
        self.sparse_random_matrix = self._sparse_random_matrix(n_features=n_features).to(device)

        return self

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.sparse_random_matrix is None:
            raise ValueError("SparseRandomProjection is not fitted yet. Call 'fit' before 'transform'.")

        return embedding @ self.sparse_random_matrix.T.float()

class KCenterGreedy:
    """k-center-greedy method for coreset selection.

    This class implements the k-center-greedy method to select a coreset from an
    embedding space. The method aims to minimize the maximum distance between any
    point and its nearest center.

    Args:
        embedding (torch.Tensor): Embedding tensor extracted from a CNN.
        sampling_ratio (float): Ratio to determine coreset size from embedding size.

    Attributes:
        embedding (torch.Tensor): Input embedding tensor.
        coreset_size (int): Size of the coreset to be selected.
        model (SparseRandomProjection): Dimensionality reduction model.
        features (torch.Tensor): Transformed features after dimensionality reduction.
        min_distances (torch.Tensor): Minimum distances to cluster centers.
        n_observations (int): Number of observations in the embedding.

    Example:
        >>> import torch
        >>> embedding = torch.randn(219520, 1536)
        >>> sampler = KCenterGreedy(embedding=embedding, sampling_ratio=0.001)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: torch.Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        self.model = SparseRandomProjection(eps=0.9)

        self.features: torch.Tensor
        self.min_distances: torch.Tensor = None
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances to None."""
        self.min_distances = None

    def update_distances(self, cluster_centers: list[int]) -> None:
        """Update minimum distances given cluster centers.

        Args:
            cluster_centers (list[int]): Indices of cluster centers.
        """
        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index of the next sample based on maximum minimum distance.

        Returns:
            int: Index of the selected sample.

        Raises:
            TypeError: If `self.min_distances` is not a torch.Tensor.
        """
        if isinstance(self.min_distances, torch.Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            msg = f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}"
            raise TypeError(msg)

        return idx

    def select_coreset_idxs(self, selected_idxs: list[int] | None = None) -> list[int]:
        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            self.model.fit(self.embedding)
            self.features = self.model.transform(self.embedding)
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: list[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        for _ in tqdm(range(self.coreset_size), desc="Selecting Coreset Indices."):
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                msg = "New indices should not be in selected indices."
                raise ValueError(msg)
            self.min_distances[idx] = 0
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: list[int] | None = None) -> torch.Tensor:
        idxs = self.select_coreset_idxs(selected_idxs)
        return self.embedding[idxs]
    
class DynamicBufferMixin(nn.Module, ABC):
    def get_tensor_attribute(self, attribute_name: str) -> torch.Tensor:
        attribute = getattr(self, attribute_name)
        if isinstance(attribute, torch.Tensor):
            return attribute

        msg = f"Attribute with name '{attribute_name}' is not a torch Tensor"
        raise ValueError(msg)

    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args) -> None:
        persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
        local_buffers = {k: v for k, v in persistent_buffers.items() if v is not None}

        for param in local_buffers:
            for key in state_dict:
                if (
                    key.startswith(prefix)
                    and key[len(prefix) :].split(".")[0] == param
                    and local_buffers[param].shape != state_dict[key].shape
                ):
                    attribute = self.get_tensor_attribute(param)
                    attribute.resize_(state_dict[key].shape)

        super()._load_from_state_dict(state_dict, prefix, *args)

class PatchCore(DynamicBufferMixin, nn.Module):
    def __init__(
        self,
        layers: list[str] = ["layer2", "layer3"],
        backbone: str = "wide_resnet50_2",
        pre_trained: bool = True,
        num_neighbors: int = 9,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = layers
        self.num_neighbors = num_neighbors

        self.feature_extractor = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=pre_trained,
            layers=self.layers,
        ).eval()
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.memory_bank: torch.Tensor
        self.register_buffer("memory_bank", torch.empty(0))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Process input tensor through the model.

        During training, returns embeddings extracted from the input. During
        inference, returns anomaly maps and scores computed by comparing input
        embeddings against the memory bank.

        Args:
            input_tensor (torch.Tensor): Input images of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor | InferenceBatch: During training, returns embeddings.
                During inference, returns ``InferenceBatch`` containing anomaly
                maps and scores.

        Example:
            >>> model = PatchcoreModel(layers=["layer1"])
            >>> input_tensor = torch.randn(32, 3, 224, 224)
            >>> output = model(input_tensor)
            >>> if model.training:
            ...     assert isinstance(output, torch.Tensor)
            ... else:
            ...     assert isinstance(output, InferenceBatch)
        """
        output_size = input_tensor.shape[-2:]

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = {layer: self.feature_pooler(feature) for layer, feature in features.items()}
        embedding = self.generate_embedding(features)

        batch_size, _, width, height = embedding.shape
        embedding = self.reshape_embedding(embedding)

        if self.training:
            if self.memory_bank.size(0) == 0:
                self.memory_bank = embedding
            else:
                new_bank = torch.cat((self.memory_bank, embedding), dim=0).to(self.memory_bank)
                self.memory_bank = new_bank
            return embedding

        # Ensure memory bank is not empty
        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot provide anomaly scores"
            raise ValueError(msg)

        # apply nearest neighbor search
        patch_scores, locations = self.nearest_neighbors(embedding=embedding, n_neighbors=1)
        # reshape to batch dimension
        patch_scores = patch_scores.reshape((batch_size, -1))
        locations = locations.reshape((batch_size, -1))
        # compute anomaly score
        pred_score = self.compute_anomaly_score(patch_scores, locations, embedding)
        # reshape to w, h
        patch_scores = patch_scores.reshape((batch_size, 1, width, height))
        # get anomaly map
        anomaly_map = self.anomaly_map_generator(patch_scores, output_size)

        return pred_score, anomaly_map

    def generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate embedding by concatenating multi-scale feature maps.

        Combines feature maps from different CNN layers by upsampling them to a
        common size and concatenating along the channel dimension.

        Args:
            features (dict[str, torch.Tensor]): Dictionary mapping layer names to
                feature tensors extracted from the backbone CNN.

        Returns:
            torch.Tensor: Concatenated feature embedding of shape
                ``(batch_size, num_features, height, width)``.

        Example:
            >>> features = {
            ...     "layer1": torch.randn(32, 64, 56, 56),
            ...     "layer2": torch.randn(32, 128, 28, 28)
            ... }
            >>> embedding = model.generate_embedding(features)
            >>> embedding.shape
            torch.Size([32, 192, 56, 56])
        """
        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="bilinear")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

    @staticmethod
    def reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """Reshape embedding tensor for patch-wise processing.

        Converts a 4D embedding tensor into a 2D matrix where each row represents
        a patch embedding vector.

        Args:
            embedding (torch.Tensor): Input embedding tensor of shape
                ``(batch_size, embedding_dim, height, width)``.

        Returns:
            torch.Tensor: Reshaped embedding tensor of shape
                ``(batch_size * height * width, embedding_dim)``.

        Example:
            >>> embedding = torch.randn(32, 512, 7, 7)
            >>> reshaped = PatchcoreModel.reshape_embedding(embedding)
            >>> reshaped.shape
            torch.Size([1568, 512])
        """
        embedding_size = embedding.size(1)
        return embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)

    def subsample_embedding(self, sampling_ratio: float, embeddings: torch.Tensor = None) -> None:
        """Subsample the memory_banks embeddings using coreset selection.

        Uses k-center-greedy coreset subsampling to select a representative
        subset of patch embeddings to store in the memory bank.

        Args:
            sampling_ratio (float): Fraction of embeddings to keep, in range (0,1].
            embeddings (torch.Tensor): **[DEPRECATED]**
            This argument is deprecated and will be removed in a future release.
            Use the default behavior (i.e., rely on `self.memory_bank`) instead.

        Example:
            >>> model.memory_bank = torch.randn(1000, 512)
            >>> model.subsample_embedding(sampling_ratio=0.1)
            >>> model.memory_bank.shape
            torch.Size([100, 512])
        """
        if embeddings is not None:
            del embeddings

        if self.memory_bank.size(0) == 0:
            msg = "Memory bank is empty. Cannot perform coreset selection."
            raise ValueError(msg)
        # Coreset Subsampling
        sampler = KCenterGreedy(embedding=self.memory_bank, sampling_ratio=sampling_ratio)
        coreset = sampler.sample_coreset().to(self.memory_bank)
        self.memory_bank = coreset

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances between two sets of vectors.

        Implements an efficient matrix computation of Euclidean distances between
        all pairs of vectors in ``x`` and ``y`` without using ``torch.cdist()``.

        Args:
            x (torch.Tensor): First tensor of shape ``(n, d)``.
            y (torch.Tensor): Second tensor of shape ``(m, d)``.

        Returns:
            torch.Tensor: Distance matrix of shape ``(n, m)`` where element
                ``(i,j)`` is the distance between row ``i`` of ``x`` and row
                ``j`` of ``y``.

        Example:
            >>> x = torch.randn(100, 512)
            >>> y = torch.randn(50, 512)
            >>> distances = PatchcoreModel.euclidean_dist(x, y)
            >>> distances.shape
            torch.Size([100, 50])

        Note:
            This implementation avoids using ``torch.cdist()`` for better
            compatibility with ONNX export and OpenVINO conversion.
        """
        x_norm = x.pow(2).sum(dim=-1, keepdim=True)  # |x|
        y_norm = y.pow(2).sum(dim=-1, keepdim=True)  # |y|
        # row distance can be rewritten as sqrt(|x| - 2 * x @ y.T + |y|.T)
        res = x_norm - 2 * torch.matmul(x, y.transpose(-2, -1)) + y_norm.transpose(-2, -1)
        return res.clamp_min_(0).sqrt_()

    def nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Find nearest neighbors in memory bank for input embeddings.

        Uses brute force search with Euclidean distance to find the closest
        matches in the memory bank for each input embedding.

        Args:
            embedding (torch.Tensor): Query embeddings to find neighbors for.
            n_neighbors (int): Number of nearest neighbors to return.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Distances to nearest neighbors (shape: ``(n, k)``)
                - Indices of nearest neighbors (shape: ``(n, k)``)
                where ``n`` is number of query embeddings and ``k`` is
                ``n_neighbors``.

        Example:
            >>> embedding = torch.randn(100, 512)
            >>> # Assuming memory_bank is already populated
            >>> scores, locations = model.nearest_neighbors(embedding, n_neighbors=5)
            >>> scores.shape, locations.shape
            (torch.Size([100, 5]), torch.Size([100, 5]))
        """
        distances = self.euclidean_dist(embedding, self.memory_bank)
        if n_neighbors == 1:
            # when n_neighbors is 1, speed up computation by using min instead of topk
            patch_scores, locations = distances.min(1)
        else:
            patch_scores, locations = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores, locations

    def compute_anomaly_score(
        self,
        patch_scores: torch.Tensor,
        locations: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute image-level anomaly scores.

        Implements the paper's weighted scoring mechanism that considers both
        the distance to nearest neighbors and the local neighborhood structure
        in the memory bank.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores.
            locations (torch.Tensor): Memory bank indices of nearest neighbors.
            embedding (torch.Tensor): Input embeddings that generated the scores.

        Returns:
            torch.Tensor: Image-level anomaly scores.

        Example:
            >>> patch_scores = torch.randn(32, 49)  # 7x7 patches
            >>> locations = torch.randint(0, 1000, (32, 49))
            >>> embedding = torch.randn(32 * 49, 512)
            >>> scores = model.compute_anomaly_score(patch_scores, locations,
            ...                                     embedding)
            >>> scores.shape
            torch.Size([32])

        Note:
            When ``num_neighbors=1``, returns the maximum patch score directly.
            Otherwise, computes weighted scores using neighborhood information.
        """
        # Don't need to compute weights if num_neighbors is 1
        if self.num_neighbors == 1:
            return patch_scores.amax(1)
        batch_size, num_patches = patch_scores.shape
        # 1. Find the patch with the largest distance to it's nearest neighbor in each image
        max_patches = torch.argmax(patch_scores, dim=1)  # indices of m^test,* in the paper
        # m^test,* in the paper
        max_patches_features = embedding.reshape(batch_size, num_patches, -1)[torch.arange(batch_size), max_patches]
        # 2. Find the distance of the patch to it's nearest neighbor, and the location of the nn in the membank
        score = patch_scores[torch.arange(batch_size), max_patches]  # s^* in the paper
        nn_index = locations[torch.arange(batch_size), max_patches]  # indices of m^* in the paper
        # 3. Find the support samples of the nearest neighbor in the membank
        nn_sample = self.memory_bank[nn_index, :]  # m^* in the paper
        # indices of N_b(m^*) in the paper
        memory_bank_effective_size = self.memory_bank.shape[0]  # edge case when memory bank is too small
        _, support_samples = self.nearest_neighbors(
            nn_sample,
            n_neighbors=min(self.num_neighbors, memory_bank_effective_size),
        )
        # 4. Find the distance of the patch features to each of the support samples
        distances = self.euclidean_dist(max_patches_features.unsqueeze(1), self.memory_bank[support_samples])
        # 5. Apply softmax to find the weights
        weights = (1 - F.softmax(distances.squeeze(1), 1))[..., 0]
        # 6. Apply the weight factor to the score
        return weights * score  # s in the paper