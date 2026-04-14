# Local Patch Set

This directory tracks the local Cognee patch set that was normalized into
feature-scoped commits on branch `panasee/v1.0.0-upstream-sync`.

Base context:
- local branch started from `v1.0.0`
- upstream sync merge landed before the patch commits
- patch files below are generated from the actual git commits via `git format-patch`

## Patch Index

1. `01-remember-http-text-inputs.patch`
   Commit: `a0e578b57`
   Purpose: allow `POST /api/v1/remember` to accept a single `text` field or repeated `texts` form fields without changing the existing file-upload path.

2. `02-search-used-graph-element-ids.patch`
   Commit: `bd996f118`
   Purpose: expose retriever-attributed `used_graph_element_ids` in verbose graph-completion search responses for downstream feedback/memify flows.

3. `03-dataset-graph-feedback-and-edge-details.patch`
   Commit: `dfb1c15f2`
   Purpose: add dataset graph edge metadata to graph reads and a bulk feedback-weight update endpoint for nodes and edges.

4. `04-hard-delete-missing-document-subgraph.patch`
   Commit: `be11ed85b`
   Purpose: let hard deletes continue when legacy document subgraphs are already missing, while preserving soft-delete behavior.

5. `05-local-runtime-falkor-auth-compat.patch`
   Commit: `5e365ed4c`
   Purpose: install and patch the local Falkor runtime path and keep optional HTTP auth decoupled from backend access control.

## Verification

- `ruff check` on all touched code and test files
- targeted unit test suite: 49 passed

## Notes

- `.codex/` and `cognee/modules/graph/methods/hard_delete_degree_one_nodes.py` remain local-only and are intentionally excluded from the committed patch set.
- `.omx/` is ignored in the repo root `.gitignore` so local OMX state does not leak into version control.
