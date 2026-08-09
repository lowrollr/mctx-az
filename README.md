# mctx-az
This fork of [google-deepmind/mctx](https://github.com/google-deepmind/mctx) introduces a new feature used in AlphaZero: continuing search from
a subtree of a previous state's search output, or _subtree persistence_.

This allows Monte Carlo Tree Search to continue from an already-initialized, partially populated search tree. This lets work done in a previous 
call to Monte Carlo Tree Search persist to the next call, avoiding lots of repeated work!

## Quickstart
mctx-az introduces a new policy: `alphazero_policy` which allows the user to pass a pre-initialized `Tree` to continue the search with.

Then, `get_subtree` can be used to extract the subtree rooted at a particular child node of the root, corresponding to a taken action.

In cases where the search tree should not be saved, such as an episdoe terminating, `reset_search_tree` can be used to clear the tree.

In order to initialize a new tree, pass `tree=None`, to `alphazero_policy`, along with `max_nodes` to specify the capacity of the tree, which in most cases
should be >= `num_simulations`.

`alphazero_policy` otherwise functions exactly the same as `muzero_policy`.

#### Initializing a new Tree:
```python
policy_output = mctx.alphazero_policy(params, rng_key, root, recurrent_fn,
                                      num_simulations=32, tree=None, max_nodes=48)
tree = policy_output.search_tree
```
#### Extracting the subtree and continuing search
```python
# get chosen action from policy output
action = policy_output.action

# extract the subtree corresponding to the chosen action
tree = mctx.get_subtree(tree, action)

# go to next environment state
env_state = env.step(env_state, action)

# reset the search tree where the environment has terminated
tree = mctx.reset_search_tree(tree, env_state.terminated)

# new search with subtree
# (max_nodes has no effect when a tree is passed) 
policy_ouput = mctx.alphazero_policy(params, rng_key, root, recurrent_fn,
                                     num_simulations=32, tree=tree)
```
#### Note on out-of-bounds expansions:
A call to any mctx policy will expand `num_simulations` nodes (assuming `max_depth` is not breached).

Given that `alphazero_policy` accepts a pre-populated `Tree`, it is possible that there will not be enough 
room left for `num_simulations` new nodes.

In the case where a tree is full, values and visit counts are still propagated backwards to all nodes along the visit path
as they would if the expansion was in bounds. However, a new node is not created and stored in the search tree, only its 
in-bounds predecessors are updated.

## Examples
The mctx readme links to a simple Connect4 example: https://github.com/Carbon225/mctx-classic

I modified this example to demonstrate the use of `alphazero_policy` and `get_subtree`. You can see it [here](https://github.com/jcbmrshll/mctx-az/blob/main/connect4.ipynb)

## Issues
If you run into problems or need help, please create an Issue and I will do my best to assist you promptly.
