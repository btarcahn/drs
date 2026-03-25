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
    parser = argparse.ArgumentParser(description="Compute the cosine similarity between two poles of two DRS.", epilog="I hope that it works.")
    parser.add_argument("--npz0", required=True, help="Path to the first .npz file containing the poles, e.g. data/output/piano-2s_drs.npz")
    parser.add_argument("--npz1", required=True, help="Path to the second .npz file containing the poles, e.g. data/output/piano-2s_drs.npz")
    return parser.parse_args()


def create_run(config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = config.get("run_name", "run")
    run_dir = Path(config["paths"]["output_dir"]) / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.json", "w") as f:
        json.dump(config, f, indent=2)
    return run_dir


def main():
    args = get_config()

    print(f"Loading poles from {args.npz0}...")
    data0 = np.load(args.npz0)
    dzs0 = data0["dzs"]
    dzs_rev0 = data0["dzs_rev"]
    print(f"Loading poles from {args.npz1}...")
    data1 = np.load(args.npz1)
    dzs1 = data1["dzs"]
    dzs_rev1 = data1["dzs_rev"]
    print('Assuming sample rate of 44100 Hz!!!')
    print('Starting cosine similarity calculation...')
    print(f"cos sim: {cosine_similarity(dzs0, dzs_rev0, dzs1, dzs_rev1, 44100)}")
    # spectrograms = []
    # for config in two_configs:    
    #     # print(f'Starting experiment: {config["run_name"]}')
    #     signal_raw, sample_rate = from_wav(config["paths"]["raw_data"])
    #     # print(f'Samples: {len(signal_raw)}' )
    #     signal_preprocessed = preprocess_signal(signal_raw, **config["preprocessing"])
    #     spectrogram = compute_drs(signal_preprocessed, **config["analysis"])
    #     # for dzs, dzs_rev, offset, window_size in spectrogram:...
    #     spectrograms.append(spectrogram)
    #     # print(f'Spectrogram length: {len(spectrogram)}')
    
    # s0, s1 = spectrograms
    # v0, v1 = s0[0], s1[0]
    # dzs0, dzs_rev0, offset0, window_size0 = v0
    # dzs1, dzs_rev1, offset1, window_size1 = v1
    

    # print('Starting cosine similarity calculation...')
    # print(f"cos sim: {cosine_similarity(dzs0, dzs_rev0, dzs1, dzs_rev1, 44100)}")

if __name__ == "__main__":
    main()
