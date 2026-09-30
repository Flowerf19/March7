# scripts/ — operational helpers

Runtime services live under `services/` and `twin/`. These helpers are explicit
owner-run operations, not an agent host-execution fallback.

- **`bootstrap_system_gateway.py`** — provision the native host gateway and
  private credentials outside the repository. Linux installation requires
  root; macOS uses a user LaunchAgent; Windows runs in the foreground with
  manual service setup. Review listener/firewall and key-mount requirements in
  [System Gateway](../services/system_gateway/README.md) before running it.
- **`pull_harrier_model.py`** — download and checksum-verify the local Harrier
  q4 ONNX model. Run one pull process at a time: downloads share a resumable
  `.part` file. No embedding API is used.
- **`migrate_t2_harrier.py`** — re-embed legacy T2 vectors at 640 dimensions.
  Without `--apply` or `--rollback`, it defaults to dry-run. Writes require
  `--apply --backup PATH`; use the checkpoint for resume and
  `--rollback --backup PATH` to restore backed-up embeddings.
  `--recreate-index` additionally requires `--apply` and a completed vector
  validation; it preserves HASH data but interrupts searches during index
  recreation. Retry after a partial index-recreation failure needs owner
  inspection. Never pad/truncate vectors, delete T2 HASHes, use `DD`, or flush
  Redis. No production migration has been performed by this remediation.
- **`calibrate_t2.py`** — evaluate recall/merge gates with the local model.
  `--offline` uses labeled examples without live Redis; results are evidence
  for that dataset, not a universal quality guarantee.
- **`migrate_t3_5to8.py`** — preview the older T3 Markdown schema migration;
  `--apply` writes changes and a `.bak5` backup. Review affected files first.

Use disposable Redis for integration tests. Production credentials, backups,
and data must never be committed or mounted into test containers.
