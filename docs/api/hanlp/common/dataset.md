# dataset

This module provides base definition for datasets, dataloaders and samplers.

## datasets

```{eval-rst}
.. currentmodule:: hanlp.common

.. autoclass:: hanlp.common.dataset.Transformable
	:members:

.. autoclass:: hanlp.common.dataset.TransformableDataset
	:members:
	:special-members:
	:exclude-members: __init__, __repr__
```

## dataloaders

```{eval-rst}
.. currentmodule:: hanlp.common

.. autoclass:: hanlp.common.dataset.PadSequenceDataLoader
	:members:
	:special-members:
	:exclude-members: __init__, __repr__

.. autoclass:: hanlp.common.dataset.PrefetchDataLoader
	:members:
	:special-members:
	:exclude-members: __init__, __repr__
```

## samplers

```{eval-rst}
.. currentmodule:: hanlp.common

.. autoclass:: hanlp.common.dataset.BucketSampler
	:members:

.. autoclass:: hanlp.common.dataset.KMeansSampler
	:members:

.. autoclass:: hanlp.common.dataset.SortingSampler
	:members:
```

## sampler builders

```{eval-rst}
.. currentmodule:: hanlp.common

.. autoclass:: hanlp.common.dataset.SamplerBuilder
	:members:

.. autoclass:: hanlp.common.dataset.SortingSamplerBuilder
	:members:

.. autoclass:: hanlp.common.dataset.KMeansSamplerBuilder
	:members:

```