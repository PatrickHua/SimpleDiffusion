"""Utils for the MAE implementation."""

import jax
import jax.numpy as jnp

PRNGKey = Any
import einops
from typing import Optional

def shuffle_and_partition(n_batch, n_tokens, n_masked, rng):
  """Implements random shuffling and partitioning necessary for MAE."""

  # Returns two arrays, the first one will contain tokens to encode, which
  # represent a random subset of the tokens in addition to the class token,
  # always at index 0. The second one returns the other part of `[0, n_tokens]`
  # array. This is repeated `n_batch` times.
  ids = jnp.tile(jnp.arange(n_tokens), n_batch).reshape((n_batch, n_tokens))
  n_other = n_tokens - n_masked
  rng_keys = jax.random.split(rng, n_batch)
  ids = jax.vmap(
      lambda seq, rng: jax.random.permutation(rng, seq, independent=True))(
          ids, rng_keys)
  masked = jax.lax.dynamic_slice(ids, (0, 0,), (n_batch, n_masked,))
  others = jax.lax.dynamic_slice(ids, (0, n_masked,), (n_batch, n_other,))
  masked = jnp.add(masked, 1)
  others = jnp.add(others, 1)
  others = jnp.concatenate(
      [jnp.tile(jnp.array([0]), [n_batch, 1]), others], axis=1)
  return others, masked

def serialize_tokens(grid: jnp.ndarray, raster: bool = False, m1: int = 2, m2: Optional[int] = None, rng: Optional[PRNGKey] = None):
  if not m2:
    m2 = m1
  n_batch, w_tokens, h_tokens, n_dim = grid.shape
  n_tokens = w_tokens * h_tokens
  n_segments = n_tokens // (m1 * m2)

  seg = einops.rearrange(
      grid, 'b (h m1) (w m2) d -> b (h w) m1 m2 d', m1=m1, m2=m2)
  assert seg.shape[1] == n_segments


  ids = jnp.tile(jnp.arange(n_tokens), n_batch).reshape((n_batch, n_tokens))

  permute_fn = jax.vmap(
    lambda seq, rng: jax.random.permutation(rng, seq, independent=True)
  )

  permuted_ids = permute_fn(ids)


  # if raster:
  #   rng = self.make_rng('permute') if rng is None else rng
  #   seg = jax.random.permutation(rng, seg, axis=-4, independent=False)

  # seg = seg.reshape((grid.shape[0], -1, grid.shape[-1]))  # flatten








