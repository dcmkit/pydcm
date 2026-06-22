import sys, time, statistics

def bench(fn, n=30, warmup=3):
    for _ in range(warmup): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts)

import pydicom
from pydicom.data import get_testdata_file
files = {
    "CT 16-bit uncompressed": get_testdata_file("CT_small.dcm"),
    "MR uncompressed":        get_testdata_file("MR_small.dcm"),
    "RLE lossless":           get_testdata_file("MR_small_RLE.dcm"),
    "JPEG2000 lossless":      get_testdata_file("MR_small_jp2klossless.dcm"),
    "JPEG-LS lossless":       get_testdata_file("MR_small_jpeg_ls_lossless.dcm"),
}
which = sys.argv[1]
if which == "pydicom":
    import pydicom as M
else:
    import pydcm as M

print(f"== {which} (py{sys.version_info[0]}.{sys.version_info[1]}) ==")
for name, path in files.items():
    try:
        t_read = bench(lambda: M.dcmread(path))
        def full():
            ds = M.dcmread(path); ds.pixel_array
        t_full = bench(full)
        print(f"{name:26s} dcmread {t_read:7.3f} ms   read+pixels {t_full:7.3f} ms")
    except Exception as e:
        print(f"{name:26s} FAILED: {type(e).__name__}: {str(e)[:60]}")
