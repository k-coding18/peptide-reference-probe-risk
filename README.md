# Pre-PEFT Diagnostics

Code for studying whether uncertainty from reference probes on frozen ESM2 embeddings can rank peptide candidates by later LoRA classification errors. Experiments cover 22 peptide tasks, neighbour-distance controls and sequence-separated evaluation.

## Installation

Python 3.11 or 3.12. Run commands from the repository root.

```bash
python -m pip install -e '.[full]'
```

## Inputs

Place the 22 processed task CSVs, `paper_task_contract.toml` and `SHA256SUMS` in `data/paper_inputs/`. Required CSV columns are `sample_id`, `sequence`, `length`, `y` and `task`.

Provide a local ESM2-t6 checkpoint matching `configs/protocol.toml`. Data and checkpoints are not included in this repository. `configs/input_contract.toml` checks that the input sequences, labels and class counts match the paper.

## Run

```bash
python scripts/reproduce_results.py \
  --input-root data/paper_inputs \
  --model-root /path/to/esm2-t6 \
  --run-root runs/reproduction
```

## Outputs

Each run saves logs, predictions and summary CSVs under `runs/reproduction/run-01/`, `run-02/` or `run-03/`. Combined summaries are saved in `runs/reproduction/three_run_summary/`.

## License

[MIT](LICENSE).
