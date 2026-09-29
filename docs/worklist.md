# Worklist

Current user-visible defects:

- `rpg refs` misses Ruby calls written without parentheses. See D27.

Open decisions:

- Whether to keep the Homebrew formula. Homebrew has no pip extras, so the
  formula installs only the core (static embeddings); `onnx` and `webgpu`
  need a PyPI install. Each release also needs a Homebrew install test and
  a formula update: the sdist `url` and `sha256`, plus every dependency
  `resource` whose version moved, all edited by hand in
  `nitkrar/homebrew-tap`. If the formula stays, `brew bump-formula-pr` and
  `brew update-python-resources`, run by hand or by the release workflow,
  would replace the hand edits.
