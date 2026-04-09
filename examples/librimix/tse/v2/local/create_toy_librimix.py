import argparse
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf


def collect_utts(subset_dir: Path):
    by_spk = defaultdict(list)
    for flac in subset_dir.rglob("*.flac"):
        uid = flac.stem
        spk = uid.split("-")[0]
        by_spk[spk].append(flac)
    return {spk: sorted(paths) for spk, paths in by_spk.items() if paths}


def load_min_mix(path1: Path, path2: Path):
    wav1, sr1 = sf.read(path1, dtype="float32")
    wav2, sr2 = sf.read(path2, dtype="float32")
    if sr1 != 16000 or sr2 != 16000:
        raise RuntimeError(f"Expected 16k audio, got {sr1} and {sr2}")
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]
    mix = 0.5 * (wav1 + wav2)
    return wav1, wav2, mix, sr1


def write_example(out_dir: Path, name: str, wav1, wav2, mix, sr: int):
    sf.write(out_dir / "s1" / f"{name}.wav", wav1, sr)
    sf.write(out_dir / "s2" / f"{name}.wav", wav2, sr)
    sf.write(out_dir / "mix_clean" / f"{name}.wav", mix, sr)


def build_split(subset_dir: Path, out_dir: Path, num_mix: int, seed: int):
    rng = random.Random(seed)
    spk2utts = collect_utts(subset_dir)
    if len(spk2utts) < 2:
        raise RuntimeError(f"Not enough speakers found in {subset_dir}")

    for sub in ("s1", "s2", "mix_clean"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    speakers = list(spk2utts.keys())
    for idx in range(num_mix):
        # Keep trying until the pair forms a valid, non-identical mixture.
        for _ in range(100):
            spk1, spk2 = rng.sample(speakers, 2)
            utt1 = rng.choice(spk2utts[spk1])
            utt2 = rng.choice(spk2utts[spk2])
            if utt1.stem != utt2.stem:
                break
        else:
            raise RuntimeError("Failed to sample a valid speaker pair.")

        mix_name = f"{utt1.stem}_{utt2.stem}"
        wav1, wav2, mix, sr = load_min_mix(utt1, utt2)
        write_example(out_dir, mix_name, wav1, wav2, mix, sr)


def main():
    parser = argparse.ArgumentParser(
        description="Create a tiny Libri2Mix-style toy dataset from LibriSpeech."
    )
    parser.add_argument("librispeech_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--train_num", type=int, default=200)
    parser.add_argument("--dev_num", type=int, default=20)
    parser.add_argument("--test_num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_map = {
        "train-100": ("train-clean-100", args.train_num, args.seed),
        "dev": ("dev-clean", args.dev_num, args.seed + 1),
        "test": ("test-clean", args.test_num, args.seed + 2),
    }

    base_out = args.output_root / "wav16k" / "min"
    for out_split, (src_split, num_mix, split_seed) in split_map.items():
        src_dir = args.librispeech_root / src_split
        if not src_dir.exists():
            raise FileNotFoundError(f"Missing source subset: {src_dir}")
        build_split(src_dir, base_out / out_split, num_mix, split_seed)


if __name__ == "__main__":
    main()
