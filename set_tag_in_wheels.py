import os
import sys

cuda_ver = sys.argv[1]
wheels = sys.argv[2:]
for wheel in wheels:
    dirname = os.path.dirname(wheel)
    basename = os.path.basename(wheel)
    parts = basename.split('-')
    if len(parts) != 5:
        continue
    version = parts[1].split('+')[0]
    parts[1] = f'{version}{cuda_ver}'
    basename = '-'.join(parts)
    os.rename(wheel, f'{dirname}/{basename}')
