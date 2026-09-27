# Worklist

Current user-visible defects:

- `rpg refs` misses Ruby calls written without parentheses. See D27.
- Two fresh builds of one unchanged tree can rank near-ties differently:
  the same chunk text gets vectors differing at cosine ~0.9996 depending on
  which texts share its embedding batch, and neither matches encoding the
  text alone. Seen with the static backend on toolshed (`rpg search Store`,
  positions 4–5).
