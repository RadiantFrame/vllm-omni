# MiniMax-H3 scripts

```
scripts/MiniMax-H3/
├── generate/     # hardware-independent clients: generate*.sh (HTTP request senders)
│                 #   + 2k.sh (hosted-API client, no local GPUs)
└── deploy/       # hardware-specific server configs, one folder per host profile
    ├── h800/4h800/     # 4x H800 (80 GiB): tier0-5 optimization ladder, FP8 candidates
    └── rtx5090/
        ├── 2rtx5090/  # 2-card 5090 (32 GiB): single-service tier0/1 + N-service fan-out (tier0_2/3/4)
        └── 4rtx5090/  # 4-card 5090 (32 GiB): single TP4 service; see its README for findings
```

## Why the split

`generate*.sh` are pure HTTP clients against `/v1/videos/sync` — they do not
care what hardware serves the request, so there is exactly **one copy per
request shape** (not per hardware dir). Differences between them are only the
request: default resolution / duration / single-port vs multi-port fan-out.

`deploy*.sh` are the hardware-coupled server configs and stay under
`deploy/<hardware>/`. Each experiment folder should keep a README recording
script status (✅/❌ + exact failure signature), measured peaks, and pitfalls —
see `deploy/rtx5090/4rtx5090/README.md` for the format.

## Usage

```bash
# start a service (pick a hardware dir), then from anywhere:
bash scripts/MiniMax-H3/generate/generate_480p_5s.sh              # -> localhost:8000
BASE_URL=http://localhost:8001 bash scripts/MiniMax-H3/generate/generate.sh
PORTS="8000 8001" bash scripts/MiniMax-H3/generate/generate_480p_5s_n.sh  # fan-out
```

All generate scripts accept env overrides (`BASE_URL`, `PORTS`, `SEED`,
`DURATION`, `FIRST_FRAME`, `OUTPUT`/`OUT_DIR`).
