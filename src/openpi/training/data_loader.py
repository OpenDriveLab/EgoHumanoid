from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raw_item = self._dataset[index]
        # Preserve episode_index for debugging
        episode_index = raw_item.get("episode_index") if isinstance(raw_item, dict) else None
        transformed = self._transform(raw_item)
        if episode_index is not None and isinstance(transformed, dict):
            transformed["episode_index"] = episode_index
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return  self._dataset.num_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return self._dataset.num_episodes


def remap_episode_indices(dataset, episodes: list[int]) -> None:
    """Remap episode_index column in hf_dataset to consecutive indices starting from 0.

    Fixes a LeRobot indexing issue when using the episodes param:
    episode_data_index uses 0..len(episodes)-1, but hf_dataset retains original values.

    Args:
        dataset: LeRobotDataset instance.
        episodes: Original episode index list.
    """
    if not hasattr(dataset, 'hf_dataset'):
        logging.warning("Dataset has no hf_dataset attribute, cannot remap episode indices")
        return
    
    # Build mapping: original index -> new index (0, 1, 2, ...)
    sorted_episodes = sorted(episodes)
    episode_map = {orig: new for new, orig in enumerate(sorted_episodes)}
    
    logging.info(f"Remapping episode_index: {len(episodes)} episodes, "
                f"original range [{min(episodes)}, {max(episodes)}] -> [0, {len(episodes)-1}]")
    
    hf_dataset = dataset.hf_dataset
    
    # Vectorized numpy mapping for efficiency
    import numpy as np
    
    # Get original episode_index array
    original_indices = np.array(hf_dataset['episode_index'])
    
    # Apply mapping via np.vectorize
    map_func = np.vectorize(lambda x: episode_map.get(x, x))
    new_indices = map_func(original_indices).tolist()
    
    # Replace column via remove + add
    new_hf_dataset = hf_dataset.remove_columns(['episode_index'])
    new_hf_dataset = new_hf_dataset.add_column('episode_index', new_indices)
    
    dataset.hf_dataset = new_hf_dataset
    
    logging.info(f"episode_index remapping complete")


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class WeightedMultiDatasetSampler(torch.utils.data.Sampler):
    """Sampler that weights samples from different datasets in a MultiLeRobotDataset.
    
    This sampler ensures that EACH BATCH contains samples from different datasets 
    according to the specified weights. For example, with weights [0.5, 0.3, 0.2] and 
    batch_size=100, each batch will have approximately 50, 30, 20 samples from each dataset.
    
    Args:
        dataset: A MultiLeRobotDataset with multiple sub-datasets.
        weights: A sequence of weights for each sub-dataset. The weights control
            the proportion of samples from each dataset in each batch.
        batch_size: The batch size, used to compute per-batch sample counts.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        dataset: lerobot_dataset.MultiLeRobotDataset,
        weights: Sequence[float],
        batch_size: int,
        seed: int = 0,
    ):
        self._dataset = dataset
        self._seed = seed
        self._epoch = 0
        self._batch_size = batch_size
        
        # Get dataset sizes and compute cumulative lengths
        dataset_sizes = [d.num_frames for d in dataset._datasets]
        cumulative_lengths = []
        cumsum = 0
        for size in dataset_sizes:
            cumsum += size
            cumulative_lengths.append(cumsum)
        
        num_datasets = len(cumulative_lengths)
        
        if len(weights) != num_datasets:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of datasets ({num_datasets})"
            )
        
        # Normalize weights
        total_weight = sum(weights)
        self._normalized_weights = [w / total_weight for w in weights]
        
        # Calculate samples per dataset per batch
        # Use floor for most, then add remaining to largest weight dataset
        samples_per_dataset = []
        remaining = batch_size
        for i, w in enumerate(self._normalized_weights):
            if i == len(self._normalized_weights) - 1:
                # Last dataset gets the remainder to ensure batch_size is exact
                samples_per_dataset.append(remaining)
            else:
                count = int(batch_size * w)
                samples_per_dataset.append(count)
                remaining -= count
        
        self._samples_per_dataset = samples_per_dataset
        
        # Store dataset ranges (start_idx, end_idx) for each dataset
        self._dataset_ranges = []
        prev_cumulative = 0
        for cumulative in cumulative_lengths:
            self._dataset_ranges.append((prev_cumulative, cumulative))
            prev_cumulative = cumulative
        
        # Total samples is the full dataset length
        self._total_samples = len(dataset)
        # Number of complete batches we can generate
        self._num_batches = self._total_samples // batch_size
        
        logging.info(
            f"WeightedMultiDatasetSampler initialized: "
            f"dataset_sizes={dataset_sizes}, "
            f"weights={list(weights)} -> normalized={self._normalized_weights}, "
            f"samples_per_dataset={self._samples_per_dataset}, "
            f"total_samples={self._total_samples}, num_batches={self._num_batches}"
        )
    
    def __iter__(self):
        # Create a new generator each iteration with epoch-based seed for reproducibility
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        
        # Generate indices batch by batch, ensuring each batch has the correct proportions
        all_indices = []
        
        for _ in range(self._num_batches):
            batch_indices = []
            for dataset_idx, (start_idx, end_idx) in enumerate(self._dataset_ranges):
                num_samples = self._samples_per_dataset[dataset_idx]
                dataset_size = end_idx - start_idx
                
                # Randomly sample from this dataset's range
                random_indices = torch.randint(
                    0, dataset_size, (num_samples,), generator=g
                )
                
                # Convert to global indices
                global_indices = random_indices + start_idx
                batch_indices.extend(global_indices.tolist())
            
            # Shuffle within the batch so samples from different datasets are interleaved
            perm = torch.randperm(len(batch_indices), generator=g)
            batch_indices = [batch_indices[i] for i in perm.tolist()]
            
            all_indices.extend(batch_indices)
        
        yield from all_indices
    
    def __len__(self) -> int:
        return self._num_batches * self._batch_size
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.
        
        This ensures different random orderings for each epoch.
        
        Args:
            epoch: Epoch number.
        """
        self._epoch = epoch


class WeightedDistributedMultiDatasetSampler(torch.utils.data.Sampler):
    """Distributed sampler that weights samples from different datasets in a MultiLeRobotDataset.
    
    This sampler ensures each rank independently samples from datasets according to 
    the specified weights. Unlike strided sampling which breaks weight ratios,
    each rank samples its own weighted batch independently.
    
    Key features:
    - Each rank gets LOCAL batches with correct weight proportions
    - Different ranks sample different data (via rank-based seed offset)
    - Supports epoch-based reseeding for reproducibility
    
    Args:
        dataset: A MultiLeRobotDataset with multiple sub-datasets.
        weights: A sequence of weights for each sub-dataset.
        batch_size: The LOCAL batch size per rank.
        num_replicas: Number of distributed processes (world size).
        rank: Current process rank.
        seed: Random seed for reproducibility.
        drop_last: Whether to drop the last incomplete batch.
    """
    
    def __init__(
        self,
        dataset: lerobot_dataset.MultiLeRobotDataset,
        weights: Sequence[float],
        batch_size: int,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        
        self._dataset = dataset
        self._num_replicas = num_replicas
        self._rank = rank
        self._epoch = 0
        self._seed = seed
        self._drop_last = drop_last
        self._batch_size = batch_size
        
        # Get dataset sizes and compute cumulative lengths
        dataset_sizes = [d.num_frames for d in dataset._datasets]
        cumulative_lengths = []
        cumsum = 0
        for size in dataset_sizes:
            cumsum += size
            cumulative_lengths.append(cumsum)
        
        num_datasets = len(cumulative_lengths)
        
        if len(weights) != num_datasets:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of datasets ({num_datasets})"
            )
        
        # Normalize weights
        total_weight = sum(weights)
        self._normalized_weights = [w / total_weight for w in weights]
        
        # Calculate samples per dataset per LOCAL batch (this rank only)
        # This ensures each rank's batch has correct weight proportions
        samples_per_dataset = []
        remaining = batch_size
        for i, w in enumerate(self._normalized_weights):
            if i == len(self._normalized_weights) - 1:
                # Last dataset gets the remainder to ensure batch_size is exact
                samples_per_dataset.append(remaining)
            else:
                count = int(batch_size * w)
                samples_per_dataset.append(count)
                remaining -= count
        
        self._samples_per_dataset = samples_per_dataset
        
        # Store dataset ranges (start_idx, end_idx) for each dataset
        self._dataset_ranges = []
        prev_cumulative = 0
        for cumulative in cumulative_lengths:
            self._dataset_ranges.append((prev_cumulative, cumulative))
            prev_cumulative = cumulative
        
        # Calculate number of batches this rank will generate
        # Each rank samples independently, so we base on total dataset size / (batch_size * num_replicas)
        total_samples = len(dataset)
        global_batch_size = batch_size * num_replicas
        self._num_batches = total_samples // global_batch_size
        
        # Total samples this rank will return
        self._num_samples = self._num_batches * batch_size
        
        logging.info(
            f"WeightedDistributedMultiDatasetSampler[rank={rank}]: "
            f"dataset_sizes={dataset_sizes}, "
            f"weights={list(weights)} -> normalized={self._normalized_weights}, "
            f"local_batch_size={batch_size}, "
            f"samples_per_dataset_per_batch={self._samples_per_dataset}, "
            f"num_batches={self._num_batches}, num_samples={self._num_samples}"
        )
    
    def __iter__(self):
        # Create generator with seed based on epoch AND rank
        # Different ranks use different seeds to sample different data
        # Same epoch across ranks ensures reproducibility
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch * self._num_replicas + self._rank)
        
        # Each rank independently generates its own weighted samples
        # This ensures each local batch has correct weight proportions
        all_indices = []
        
        for batch_idx in range(self._num_batches):
            batch_indices = []
            for dataset_idx, (start_idx, end_idx) in enumerate(self._dataset_ranges):
                num_samples = self._samples_per_dataset[dataset_idx]
                dataset_size = end_idx - start_idx
                
                # Randomly sample from this dataset's range
                random_indices = torch.randint(
                    0, dataset_size, (num_samples,), generator=g
                )
                
                # Convert to global indices
                global_indices = random_indices + start_idx
                batch_indices.extend(global_indices.tolist())
            
            # Shuffle within the batch so samples from different datasets are interleaved
            perm = torch.randperm(len(batch_indices), generator=g)
            batch_indices = [batch_indices[i] for i in perm.tolist()]
            
            all_indices.extend(batch_indices)
        
        assert len(all_indices) == self._num_samples, (
            f"Number of generated indices ({len(all_indices)}) does not match "
            f"num_samples ({self._num_samples})"
        )
        
        yield from all_indices
    
    def __len__(self) -> int:
        return self._num_samples
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler.
        
        This ensures different random orderings for each epoch while maintaining
        reproducibility. All ranks should call this with the same epoch value.
        
        Args:
            epoch: Epoch number.
        """
        self._epoch = epoch


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if isinstance(repo_id, list):
        # If repo_id is a list, create a dataset for each repo_id and concatenate them.
        dataset_metas = [
            lerobot_dataset.LeRobotDatasetMetadata(r) for r in repo_id
        ]
        dataset = lerobot_dataset.MultiLeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
                for dataset_meta in dataset_metas
                for key in data_config.action_sequence_keys
            },
            episodes=data_config.episodes,
        )
        # Apply per-dataset transforms
        # Note: LeRobot's MultiLeRobotDataset automatically adds "dataset_index" field
        for n, d in enumerate(dataset._datasets):
            transforms_list = []
            
            # Add prompt transform if enabled
            if data_config.prompt_from_task:
                transforms_list.append(_transforms.PromptFromLeRobotTask(dataset_metas[n].tasks))
            
            # Add dataset index marker if per-dataset action keys are configured
            if data_config.per_dataset_action_keys is not None:
                transforms_list.append(_transforms.AddDatasetIndex(dataset_idx=n))
            
            if transforms_list:
                dataset._datasets[n] = TransformedDataset(d, transforms_list)
    else:
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
            episodes=data_config.episodes,
        )
        
        # Remap episode indices in hf_dataset when episodes param is used
        if data_config.episodes is not None:
            remap_episode_indices(dataset, data_config.episodes)
        
        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = list(data_config.repack_transforms.inputs)
    
    # Add per-dataset ConcatActions if configured (must be after repack_transforms)
    if data_config.per_dataset_action_keys is not None:
        transforms_list.append(
            _transforms.PerDatasetConcatActions(
                per_dataset_action_keys=tuple(data_config.per_dataset_action_keys),
                output_key="actions",
                create_mask=data_config.use_action_mask,
            )
        )
    
    transforms_list.extend([
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    return TransformedDataset(dataset, transforms_list)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = list(data_config.repack_transforms.inputs)
    
    # Add per-dataset ConcatActions if configured (must be after repack_transforms)
    if data_config.per_dataset_action_keys is not None:
        transforms_list.append(
            _transforms.PerDatasetConcatActions(
                per_dataset_action_keys=tuple(data_config.per_dataset_action_keys),
                output_key="actions",
                create_mask=data_config.use_action_mask,
            )
        )
    
    transforms_list.extend([
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ])

    return IterableTransformedDataset(dataset, transforms_list, is_batched=is_batched)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    # Create raw dataset first (before transforms) to access MultiLeRobotDataset properties
    raw_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    
    # First, compute local_batch_size (needed for weighted sampler)
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
    
    sampler = None
    use_shuffle = shuffle
    
    # Check if weighted sampling is configured for multi-dataset
    if (
        data_config.dataset_weights is not None
        and isinstance(data_config.repo_id, list)
        and len(data_config.repo_id) > 1
    ):
        # Get the underlying MultiLeRobotDataset
        # raw_dataset may be wrapped in TransformedDataset, need to get the original
        underlying_dataset = raw_dataset
        while hasattr(underlying_dataset, '_dataset'):
            underlying_dataset = underlying_dataset._dataset
        
        if isinstance(underlying_dataset, lerobot_dataset.MultiLeRobotDataset):
            sampler = WeightedMultiDatasetSampler(
                underlying_dataset,
                weights=data_config.dataset_weights,
                batch_size=local_batch_size,
                seed=seed,
            )
            use_shuffle = False  # Don't shuffle when using sampler
            logging.info(
                f"Using weighted sampling with weights: {list(data_config.dataset_weights)} "
                f"for {len(data_config.repo_id)} datasets, "
                f"per-batch samples: {sampler._samples_per_dataset}"
            )
        else:
            logging.warning(
                "dataset_weights specified but dataset is not MultiLeRobotDataset, "
                "falling back to uniform sampling"
            )
    
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            if sampler is not None:
                # We have a weighted sampler, need to replace with distributed version
                logging.info(
                    "Replacing WeightedMultiDatasetSampler with WeightedDistributedMultiDatasetSampler "
                    "for distributed training"
                )
                # Get the underlying MultiLeRobotDataset
                underlying_dataset = raw_dataset
                while hasattr(underlying_dataset, '_dataset'):
                    underlying_dataset = underlying_dataset._dataset
                
                sampler = WeightedDistributedMultiDatasetSampler(
                    underlying_dataset,
                    weights=data_config.dataset_weights,
                    batch_size=local_batch_size,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    seed=seed,
                    drop_last=True,
                )
                logging.info(
                    f"Using weighted distributed sampling with weights: {list(data_config.dataset_weights)} "
                    f"for {len(data_config.repo_id)} datasets, "
                    f"per-batch samples per rank: {sampler._samples_per_dataset}"
                )
            else:
                # No weighted sampling, use standard DistributedSampler
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=True,
                )
            use_shuffle = False

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and use_shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._framework = framework

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._sampler = sampler
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader
    
    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler (for distributed training)."""
        if self._sampler is not None and hasattr(self._sampler, 'set_epoch'):
            self._sampler.set_epoch(epoch)

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    # batch = jax.tree.map(torch.as_tensor, batch)
                    # # For PyTorch, ensure images are in CHW format
                    # if self._framework == "pytorch" and "image" in batch:
                    #     def convert_image_format(img):
                    #         # If image is in HWC format [..., H, W, C], convert to CHW [..., C, H, W]
                    #         if img.dim() >= 3 and img.shape[-1] <= 4:  # Assume last dim is channels
                    #             # Permute from [..., H, W, C] to [..., C, H, W]
                    #             return img.permute(*range(img.dim() - 3), img.dim() - 1, img.dim() - 3, img.dim() - 2)
                    #         return img
                    #     batch["image"] = {k: convert_image_format(v) for k, v in batch["image"].items()}
                    # yield batch
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(
        self, 
        data_config: _config.DataConfig, 
        data_loader: TorchDataLoader | RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for the sampler (for distributed training)."""
        if hasattr(self._data_loader, 'set_epoch'):
            self._data_loader.set_epoch(epoch)

    def __iter__(self):
        for batch in self._data_loader:
            observation = _model.Observation.from_dict(batch)
            actions = batch["actions"]
            action_mask = batch.get("action_mask")  # May be None if not using per-dataset masking
            episode_index = batch.get("episode_index")  # Preserve episode_index for debugging
            yield observation, actions, action_mask, episode_index
