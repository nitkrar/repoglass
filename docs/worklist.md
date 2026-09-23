# Worklist

Current user-visible defects:

- Ruby loses paren-less method calls. Distinguishing one from a local
  read needs the scope tracking `(#is-not? local)` assumes, so the
  reference query keeps only what the grammar already calls a call.
