import time, statistics, numpy as np
from concurrent.futures import ThreadPoolExecutor
import pydcm
from pydicom.data import get_testdata_file
# build a realistic-size batch: 64 J2K files (decode-bound, where GIL release matters)
src = get_testdata_file("MR_small_jp2klossless.dcm")
N = 64
def decode_one(_): return pydcm.decode(src)
def run(workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(decode_one, range(N)))
def bench(workers, n=7):
    for _ in range(2): run(workers)
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); run(workers); ts.append(time.perf_counter()-t0)
    return statistics.median(ts)
t1 = bench(1)
for w in (2, 4, 8):
    tw = bench(w)
    print(f"  {w} threads: {tw*1000:6.1f} ms   speedup {t1/tw:.2f}x")
print(f"  1 thread : {t1*1000:6.1f} ms (baseline, {N} J2K decodes)")
