from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np

from drs.config import load_config
from drs.io import from_wav
from drs.preprocessing import preprocess_signal
from drs.analysis import compute_drs
from drs.core import cosine_similarity


def get_config():
    parser = argparse.ArgumentParser(
        prog="poles_dancing.py",
        description="Just compute the DRS and save the poles to the disk as an .npz file. It's pole dancing, really!",
        epilog="This is useful for cosine similarity calculations. One can just persist the signal to disk and reuse it, which saves tonnes of time."
        )
    parser.add_argument("--config", required=True, help="Path to the YAML config file, e.g. configs/experiments/piano.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    return config


def create_run(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = config.get("run_name", "run")
    run_dir = Path(config["paths"]["output_dir"]) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=2)
    return run_dir


def main():
    config = get_config()

    print(f'=== Configuration: {config["run_name"]} ===')
    signal_raw, sample_rate = from_wav(config["paths"]["raw_data"])
    print(f'Samples: {len(signal_raw)}' )
    signal_preprocessed = preprocess_signal(signal_raw, **config["preprocessing"])
    spectrogram = compute_drs(signal_preprocessed, **config["analysis"])
    print(f"Spectrogram length (expecting 1): {len(spectrogram)}")
    dzs0, dzs_rev0, offset0, window_size0 = spectrogram[0]
    np.savez(f'data/output/{config["run_name"]}_drs.npz', dzs=dzs0, dzs_rev=dzs_rev0, offset=offset0, window_size=window_size0)
    print(f"Saved poles to data/output/{config['run_name']}_drs.npz")
    print("=== Terminated ===")

if __name__ == "__main__":
    main()
