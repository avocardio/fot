"""Export a tiktoken vocabulary and build its static rank table.

    python tools/make_vocab.py cl100k_base enwik9 vocab/

Writes <name>.vocab (token byte strings in id order) and <name>.table (token ids
sorted by their count in enwik9 bytes [100 MiB, 200 MiB), ties by id), the
files shipped in vocab/. Requires tiktoken and numpy.
"""
import struct, sys
import numpy as np, tiktoken

if len(sys.argv) != 4:
    sys.exit(__doc__)
name, corpus, out = sys.argv[1:4]
enc = tiktoken.get_encoding(name)
tokens = sorted(enc._mergeable_ranks.items(), key=lambda kv: kv[1])
V = len(tokens)
assert [r for _, r in tokens] == list(range(V))

with open(f'{out}/{name}.vocab', 'wb') as f:
    f.write(b'FOTV' + struct.pack('<I', V))
    for tok, _ in tokens:
        f.write(bytes([len(tok)]) + tok)

with open(corpus, 'rb') as f:
    f.seek(100 << 20)
    text = f.read(100 << 20).decode('utf-8', errors='replace')
counts = np.bincount(enc.encode_ordinary(text), minlength=enc.n_vocab)[:V]
order = np.lexsort((np.arange(V), -counts)).astype('<u4')
with open(f'{out}/{name}.table', 'wb') as f:
    f.write(b'FOTT' + struct.pack('<I', V) + order.tobytes())
