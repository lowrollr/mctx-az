# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp


T = TypeVar("T")


@chex.dataclass(frozen=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action.
  children_prior_logits: `[B, N, Anum_actions` the action prior logits of each
    node.
  children_visits: `[B, N, num_actions]` the visit counts for children for
    each action.
  children_rewards: `[B, N, num_actions]` the immediate reward for each action.
  children_discounts: `[B, N, num_actions]` the discount between the
    `children_rewards` and the `children_values`.
  children_values: `[B, N, num_actions]` the value of the next node after the
    action.
  next_node_index: `[B]` the next free index where a new node can be inserted.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  node_visits: chex.Array  # [B, N]
  raw_values: chex.Array  # [B, N]
  node_values: chex.Array  # [B, N]
  parents: chex.Array  # [B, N]
  action_from_parent: chex.Array  # [B, N]
  children_index: chex.Array  # [B, N, num_actions]
  children_prior_logits: chex.Array  # [B, N, num_actions]
  children_visits: chex.Array  # [B, N, num_actions]
  children_rewards: chex.Array  # [B, N, num_actions]
  children_discounts: chex.Array  # [B, N, num_actions]
  children_values: chex.Array  # [B, N, num_actions]
  next_node_index: chex.Array  # [B]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: chex.Array  # [B, num_actions]
  extra_data: T  # [B, ...]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(indices).shape:
      return jax.vmap(_unbatched_qvalues)(self, indices)
    else:
      return _unbatched_qvalues(self, indices)
    # pytype: enable=wrong-arg-types

  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.node_values, 2)
    value = self.node_values[:, Tree.ROOT_INDEX]
    batch_size, = value.shape
    root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(value.dtype)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  qvalues: chex.Array


def _unbatched_qvalues(tree: Tree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_rewards[index]
      + tree.children_discounts[index] * tree.children_values[index]
  )


def _get_translation(
    tree: Tree,
    child_index: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
  subtrees = jnp.arange(tree.num_simulations+1)

  def propagate_fun(_, subtrees):
    parents_subtrees = jnp.where(
        tree.parents != tree.NO_PARENT,
        subtrees[tree.parents],
        0
    )
    return jnp.where(
        jnp.greater(parents_subtrees, 0),
        parents_subtrees,
        subtrees
    )

  subtrees = jax.lax.fori_loop(0, tree.num_simulations, propagate_fun, subtrees)
  slots_aranged = jnp.arange(tree.num_simulations+1)
  subtree_master_idx = tree.children_index[tree.ROOT_INDEX, child_index]
  nodes_to_retain = subtrees == subtree_master_idx
  old_subtree_idxs = nodes_to_retain * slots_aranged
  cumsum = jnp.cumsum(nodes_to_retain)
  new_next_node_index = cumsum[-1]

  translation = jnp.where(
      nodes_to_retain,
      nodes_to_retain * (cumsum-1),
      tree.UNVISITED
  )
  erase_idxs = slots_aranged >= new_next_node_index

  return old_subtree_idxs, translation, erase_idxs


@jax.vmap
def get_subtree(
  tree: Tree,
  child_index: chex.Array
) -> Tree:
  """Extracts subtrees rooted at child indices of the root node, 
  across a batch of trees. Converts node index mappings and collapses
  node data so that populated nodes are contiguous and start at index 0.

  Assumes `tree` elements and `child_index` have a batch dimension.

  Args:
    tree: the tree to extract subtrees from
    child_index: `[B]` the index of the child (from the root) to extract each
      subtree from
  """
  # get mapping from old node indices to new node indices
  # and a mask of which nodes indices to erase
  old_subtree_idxs, translation, erase_idxs = _get_translation(
                                              tree, child_index)
  new_next_node_index = translation.max(axis=-1) + 1

  def translate(x, null_value=0):
    return jnp.where(
        erase_idxs.reshape((-1,) + (1,) * (x.ndim - 1)),
        jnp.full_like(x, null_value),
        # cases where translation == -1 will set last index
        # but since we are at least removing the root node
        # (and making one of its children the new root)
        # the last index will always be freed
        # and overwritten with zeros
        x.at[translation].set(x[old_subtree_idxs]),
    )

  def translate_idx(x, null_value=tree.UNVISITED):
    return jnp.where(
        erase_idxs.reshape((-1,) + (1,) * (x.ndim - 1)),
        jnp.full_like(x, null_value),
        # in this case we need to explicitly check for index
        # mappings to UNVISITED, since otherwise thsese will
        # map to the value of the last index of the translation
        x.at[translation].set(jnp.where(
            x == null_value,
            null_value,
            translation[x])))

  def translate_pytree(x, null_value=0):
    return jax.tree_map(
        lambda t: translate(t, null_value=null_value), x)

  return tree.replace(
      node_visits=translate(tree.node_visits),
      raw_values=translate(tree.raw_values),
      node_values=translate(tree.node_values),
      parents=translate_idx(tree.parents),
      action_from_parent=translate(
        tree.action_from_parent,
        null_value=tree.NO_PARENT).at[tree.ROOT_INDEX].set(tree.NO_PARENT),
      children_index=translate_idx(tree.children_index),
      children_prior_logits=translate(tree.children_prior_logits),
      children_visits=translate(tree.children_visits),
      children_rewards=translate(tree.children_rewards),
      children_discounts=translate(tree.children_discounts),
      children_values=translate(tree.children_values),
      next_node_index=new_next_node_index,
      root_invalid_actions=jnp.zeros_like(tree.root_invalid_actions),
      embeddings=translate_pytree(tree.embeddings)
  )


def reset_search_tree(
    tree: Tree,
    select_batch: Optional[chex.Array] = None) -> Tree:
  """Fills search tree with default values for selected batches.

  Useful for resetting the search tree after a terminated episode.

  Args:
    tree: the tree to reset
    select_batch: `[B]` a boolean mask to select which batch elements to reset.
      If `None`, all batch elements are reset.
  """
  if select_batch is None:
    select_batch = jnp.ones(tree.node_visits.shape[0], dtype=bool)

  return tree.replace(
      node_visits=tree.node_visits.at[select_batch].set(0),
      raw_values=tree.raw_values.at[select_batch].set(0),
      node_values=tree.node_values.at[select_batch].set(0),
      parents=tree.parents.at[select_batch].set(tree.NO_PARENT),
      action_from_parent=tree.action_from_parent.at[select_batch].set(
        tree.NO_PARENT),
      children_index=tree.children_index.at[select_batch].set(tree.UNVISITED),
      children_prior_logits=tree.children_prior_logits.at[select_batch].set(0),
      children_values=tree.children_values.at[select_batch].set(0),
      children_visits=tree.children_visits.at[select_batch].set(0),
      children_rewards=tree.children_rewards.at[select_batch].set(0),
      children_discounts=tree.children_discounts.at[select_batch].set(0),
      next_node_index=tree.next_node_index.at[select_batch].set(1),
      embeddings=jax.tree_util.tree_map(
          lambda t: t.at[select_batch].set(0),
          tree.embeddings),
      root_invalid_actions=tree.root_invalid_actions.at[select_batch].set(0)
      # extra_data is always overwritten by a call to search()
  )
