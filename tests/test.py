"""Round-trip and error-handling tests. Run: make test"""
import array, os, random, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOT, VOCAB = os.path.join(ROOT, 'fot'), os.path.join(ROOT, 'vocab')
tmp = tempfile.mkdtemp()
random.seed(0)

cases = {
    'empty': b'', 'one': b'x', 'allbytes': bytes(range(256)) * 3,
    'random': bytes(random.getrandbits(8) for _ in range(20000)),
    'text': b'The quick brown fox jumps over the lazy dog. ' * 500,
    'spaces': b'a' + b' ' * 300 + b'b\n\n\n  c',
    'unicode': ('Grüße 世界 مرحبا 🎉 ' * 300).encode(),
    'longtoken': b'-' * 1000 + b'=' * 1000,
    'code': b'def f(x):\n    return [i * x for i in range(10) if i % 2]\n' * 400,
    'chinese': '压缩是将数据编码为更少位的过程。'.encode() * 500,
}
opts = ['', '-a greedy', '-e leb -m adaptive', '-e prefix -t 200 -m static', '-K 16', '-B 777', '-j 1', '-j 4 -B 4096',
        '-a greedy -B 5000', '-t 64', '-c zstd:3', '-c xz:1', '-c xze:1', '-c zlib:6', '-c bz2', '-c brotli:5',
        '-c brotli:5 -w 26 -a greedy']
fails = total = 0

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True)

def check(ok, what):
    global fails, total
    total += 1
    if not ok:
        fails += 1
        print('FAIL', what)

for name, data in cases.items():
    src = f'{tmp}/{name}.in'
    open(src, 'wb').write(data)
    for o in opts:
        dec_opts = o.split('-c')[0]
        r = run(f'{FOT} enc -q {o} {src} {tmp}/x.fot && {FOT} dec -q {dec_opts} {tmp}/x.fot {tmp}/x.out')
        check(r.returncode == 0 and open(f'{tmp}/x.out', 'rb').read() == data, f'{name} [{o}] {r.stderr.decode().strip()}')

o200k = f'-v {VOCAB}/o200k_base.vocab'
check(run(f'{FOT} enc -q {o200k} {tmp}/unicode.in {tmp}/x.fot && {FOT} dec -q {o200k} {tmp}/x.fot {tmp}/x.out '
          f'&& cmp -s {tmp}/unicode.in {tmp}/x.out').returncode == 0, 'o200k round trip')
check(run(f'{FOT} enc -q -c zstd:1 {tmp}/text.in {tmp}/x.fot && {FOT} dec -q {o200k} {tmp}/x.fot {tmp}/x.out').returncode != 0,
      'vocab mismatch not detected')

# precomputed ids (-I): single-byte token ids spell the input exactly
vb, p, tid, byte_id = open(f'{VOCAB}/cl100k_base.vocab', 'rb').read(), 8, 0, {}
while p < len(vb):
    if vb[p] == 1: byte_id[vb[p + 1]] = tid
    p, tid = p + 1 + vb[p], tid + 1
open(f'{tmp}/abc.in', 'wb').write(b'abc' * 2000)
array.array('I', [byte_id[c] for c in b'abc' * 2000]).tofile(open(f'{tmp}/ids.u32', 'wb'))
check(run(f'{FOT} enc -q -B 5 -I {tmp}/ids.u32 {tmp}/abc.in {tmp}/x.fot && {FOT} dec -q {tmp}/x.fot {tmp}/x.out '
          f'&& cmp -s {tmp}/abc.in {tmp}/x.out').returncode == 0, 'precomputed ids')

# truncated files must fail with an error, not crash
run(f'{FOT} enc -q -c zstd:1 {tmp}/text.in {tmp}/x.fot')
good = open(f'{tmp}/x.fot', 'rb').read()
for cut in (1, 2, 7):
    open(f'{tmp}/bad.fot', 'wb').write(good[:-cut])
    check(run(f'{FOT} dec -q {tmp}/bad.fot {tmp}/x.out').returncode == 1, f'truncated by {cut}')

print(f'{total - fails} passed, {fails} failed')
sys.exit(1 if fails else 0)
