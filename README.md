# MOP-ABE: research prototype

Python prototype of MOP-ABE, a module-LWE key-policy attribute-based encryption (ABE) scheme in which the cloud transforms a ciphertext with a blinded transformation key and returns a short packed ciphertext (2,400 bytes at module rank 7) that the user decrypts with a 448-byte key. The raw measurements behind the tables and figures of the accompanying paper are in `data/`.

> **Research code.** It is written for measurement, is not constant-time, and must not be used to protect real data. Timings come from unoptimized Python on a shared two-core virtual machine, so ratios, not absolute milliseconds, are the meaningful quantity.

## Layout

| Path | Content |
|---|---|
| `src/mopabe/ring.py` | Ring R = Z[X]/(X^256+1) with an exact residue-number-system (RNS) layer and number-theoretic transform (NTT) |
| `src/mopabe/sampling.py` | Gadget matrix, trapdoor generation, perturbed preimage sampler, discrete Gaussians |
| `src/mopabe/mop.py` | Scheme: `setup`, `user_keygen`, `tk_gen`, `encrypt`, `transform`, `user_decrypt` |
| `src/mopabe/params.py`, `estimator.py` | Parameter selection and our own core-SVP estimator (validated against Kyber 118/182/255) |
| `scripts/bench.py` | Benchmark driver, one stage per argument |
| `tests/` | End-to-end smoke test and statistical checks |
| `data/` | CSV results and logs produced by `bench.py` |

## Requirements

Python 3.10 or newer and NumPy 2.x (`pip install -r requirements.txt`). Nothing else.

## Quick start

```bash
python3 tests/smoke_end_to_end.py 7 4      # module rank k=7, 4 attributes; prints timings and "ok True"
python3 tests/check_noise_ratio.py         # measured vs predicted noise
```

## Reproducing the data

```bash
python3 scripts/bench.py params      # data/params.csv   (security, noise, correctness per rank)
python3 scripts/bench.py scale       # data/scale.csv    (sizes and timings versus attributes and rank)
python3 scripts/bench.py psweep      # data/psweep.csv   (clause size sweep)
python3 scripts/bench.py negative    # data/negative.csv (unauthorized attempts)
python3 scripts/bench.py compress    # data/compress.csv (compression widths)
```

Running a stage overwrites the matching CSV. Stages take from minutes to about an hour on a small VM; the setup step at k=7 takes roughly 30 s. Timings vary between machines.

## Citation

See `CITATION.cff`. Repository URL and Zenodo DOI: [to be set].

## License

MIT, see `LICENSE`.
