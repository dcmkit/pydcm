import sys, time, statistics
def bench(fn, n=30, warmup=3):
    for _ in range(warmup): fn()
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)
import pydicom
from pydicom.data import get_testdata_file
path = get_testdata_file("CT_small.dcm")
which = sys.argv[1]
M = __import__(which)
ds0 = M.dcmread(path)
n_elem = len(ds0)
def read_touch_all():
    ds = M.dcmread(path)
    for el in ds: _ = el.value          # force value conversion (defeats lazy reads)
print(f"{which}: elements={n_elem}  dcmread={bench(lambda: M.dcmread(path)):.3f} ms  read+touch-all={bench(read_touch_all):.3f} ms")
