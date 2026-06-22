# pydcm benchmarks

Run against an **installed wheel** (not the dev `src/` build — C++ changes like
GIL release only take effect in a recompiled `_core`):

```bash
pip install dist/pydcm-*.whl numpy pydicom   # pydicom supplies the bundled sample file
python bench/bench_single.py pydcm
python bench/bench_touch_all.py pydcm     # lazy read
python bench/bench_threads.py             # GIL-release multithread scaling
```

Reference numbers (Apple M-series, py3.12, 0.1.0), median ms:

| workload | pydcm |
|---|---|
| MR uncompressed read+pixels | **0.160** |
| RLE read+pixels | **0.174** |
| JPEG 2000 read+pixels | **0.615** |
| JPEG-LS read+pixels | **0.247** |
| CT read + touch all 258 values | **0.934** |

pydcm decodes every transfer syntax built in (JPEG 2000, JPEG-LS, …) with no
separate codec plugins to install.

Threaded decode (64 J2K, GIL released): 2t **1.95x**, 4t **3.54x**, 8t **5.93x**.
