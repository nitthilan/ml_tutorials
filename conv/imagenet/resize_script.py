from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing
cpus = multiprocessing.cpu_count()
cpus = min(48,cpus)

base_home = Path('/mnt/additional/nitthilan/data/ml_tutorial/')
PATH = base_home/'imagenet'
#DEST = Path('/mnt/ram')
DEST = base_home/'imagenet-sz'

#szs = (int(128*1.25), int(256*1.25))
szs = (224,)

def resize_img(p, im, fn, sz):
    w,h = im.size
    ratio = min(h/sz,w/sz)
    im = im.resize((int(w/ratio), int(h/ratio)), resample=Image.BICUBIC)
    #import pdb; pdb.set_trace()
    new_fn = DEST/str(sz)/fn.relative_to(PATH)
    new_fn.parent.mkdir(exist_ok=True)
    im.save(new_fn)

def resizes(p, fn):
    im = Image.open(fn)
    for sz in szs: resize_img(p, im, fn, sz)

def resize_imgs(p):
    files = p.glob('*/*.JPEG')
    # print(sum(1 for x in files[int(0.7*1281166)]))
    n = 0
    for x in files: 
        print(x) 
        n+=1
        if(n == int(0.6*1281166)): break
    #list(map(partial(resizes, p), files))
    with ProcessPoolExecutor(cpus) as e: e.map(partial(resizes, p), files)


for sz in szs:
    ssz=str(sz)
    (DEST/ssz).mkdir(exist_ok=True)
    for ds in ('val','train'): (DEST/ssz/ds).mkdir(exist_ok=True)

# for ds in ('val','train'): resize_imgs(PATH/ds)

for ds in ['train']: resize_imgs(PATH/ds)
