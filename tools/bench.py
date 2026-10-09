"""Benchmark fot transform modes against raw codecs. Usage:
  python tools/bench.py --data f1 f2 --codecs zstd:22 xz:6 zlib:9 --modes raw dp greedy --out res.json [--threads 1] [--ids-dir DIR]
Modes: raw (baseline), dp, greedy, bpe (exact cl100k BPE ids via tiktoken, cached in --ids-dir), or name=<fot args>.
"""
import argparse, json, os, subprocess, sys, tempfile

MODES = {'dp': '-a dp', 'greedy': '-a greedy'}
FOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fot")

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode: sys.exit(f'FAILED: {cmd}\n{r.stderr}')
    return r.stdout.strip().split('\t'), r.stderr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--codecs', nargs='+', default=['zstd:22', 'xz:6', 'zlib:9'])
    ap.add_argument('--modes', nargs='+', default=['raw', 'greedy']); ap.add_argument('--out', default=None); ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--extra', default=''); ap.add_argument('--ids-dir', default=''); ap.add_argument('--no-dec', action='store_true'); a = ap.parse_args()
    tmp = tempfile.mkdtemp(); j = f'-j {a.threads}' if a.threads else ''; res = []
    for f in a.data:
        n = os.path.getsize(f); base = os.path.basename(f)
        for mode in a.modes:
            if mode == 'raw': args = None
            elif mode == 'bpe':
                ids = f'{a.ids_dir or tmp}/{base}.u32'
                if not os.path.exists(ids):
                    import tiktoken, array
                    array.array('I', tiktoken.get_encoding('cl100k_base').encode_ordinary(open(f, 'rb').read().decode('utf-8'))).tofile(open(ids, 'wb'))
                args = f'-I {ids}'
            elif '=' in mode: args = mode.split('=', 1)[1]
            else: args = MODES[mode]
            for codec in a.codecs:
                out = f'{tmp}/x.fot'; cmd = 'raw' if args is None else 'enc'
                fields, err = run(f'{FOT} {cmd} -q {j} {args or ""} {a.extra} -c {codec} {f} {out}')
                r = {'data': base, 'bytes': n, 'mode': mode, 'codec': codec, 'size': int(fields[4]), 'ratio': float(fields[5]), 't_transform': float(fields[6]), 't_codec': float(fields[7])}
                r['t_total'] = r['t_transform'] + r['t_codec']
                if not a.no_dec:
                    t = (args or '').split(); dargs = ' '.join(f'{t[i]} {t[i + 1]}' for i in range(len(t) - 1) if t[i] in ('-v', '-T'))
                    d, _ = run(f'{FOT} {"unraw" if args is None else "dec"} -q {j} {dargs} {out} {tmp}/x.out')
                    r['t_decodec'] = float(d[3]); r['t_untransform'] = float(d[4]); r['t_dec_total'] = r['t_decodec'] + r['t_untransform']
                    if subprocess.run(['cmp', '-s', f, f'{tmp}/x.out']).returncode: sys.exit(f'ROUNDTRIP MISMATCH {f} {mode} {codec}')
                res.append(r)
                print(f"{base:16s} {mode:10s} {codec:10s} {r['ratio']:7.3f}%  enc {r['t_total']:7.2f}s (tr {r['t_transform']:.2f})  dec {r.get('t_dec_total', 0):6.3f}s", flush=True)
    if a.out: json.dump(res, open(a.out, 'w'), indent=1)

if __name__ == '__main__': main()
