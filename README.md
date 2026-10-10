# fot: frequency-ordered tokenization

A text transform that makes standard compressors both smaller and faster. `fot` segments text into the
vocabulary of a pretrained BPE tokenizer (OpenAI's cl100k by default), replaces each token by its frequency rank
from a fixed table, and writes the ranks with a byte-oriented prefix code (ranks below 128 take one byte). The
result is about 40% of the text's size and is then compressed with zstd, xz, brotli, zlib or bzip2.

enwik8 (100 MB), one core, times include the transform:

| compressor | raw    | fot        | raw time | fot time   |
|------------|--------|------------|----------|------------|
| zlib-9     | 36.48% | **28.52%**   | 4.1 s    | **2.6 s**    |
| xz-6       | 26.38% | **24.14%**   | 58.8 s   | **18.2 s**   |
| zstd-22    | 25.33% | **23.89%**   | 86.8 s   | **17.2 s**   |
| brotli-11  | 25.74% | **23.78%**   | 170.8 s  | **94.5 s**   |

On enwik9, `fot` sets a new state of the art among LZ-based compressors on the
[Large Text Compression Benchmark](https://mattmahoney.net/dc/text.html): followed by xz
(`--lzma2=preset=9e,lc=4,pb=0`, dictionary at least the stream size) it gives 189,382,412 bytes, or 189,923,156
including our decompressor, vocabulary and table. The static table was learned from part
of enwik9; the benchmark allows this because the table counts toward the decompressor size. To reproduce:

```
./fot enc -c none enwik9 enwik9.fot
xz -k -T1 --lzma2=preset=9e,dict=512MiB,lc=4,pb=0 enwik9.fot
```

## Build

Needs a C11 compiler with OpenMP and the zstd, xz, zlib, bzip2 and brotli libraries
(Debian/Ubuntu: `apt install libzstd-dev liblzma-dev zlib1g-dev libbz2-dev libbrotli-dev`).

```
make
make test
```

## Use

```
./fot enc -c zstd:19 input.txt input.fot
./fot dec input.fot output.txt
```

`fot enc -c none` writes the transformed stream for use with any external compressor. `./fot` without
arguments lists all options: segmentation (`-a dp|greedy`), rank map, rank code, threads and block size.
The output is identical for any number of threads.

## How it works

- **Segmentation.** Any split of the text into vocabulary tokens is lossless, because the vocabulary contains
  every single byte. `fot` picks the split with the smallest estimated code length, sum of log2(rank + 2), by
  dynamic programming over a trie of the vocabulary. This is both better and faster than exact BPE.
- **Ranks.** The table in `vocab/` orders the vocabulary by token frequency in enwik9 bytes [100 MiB, 200 MiB),
  which do not overlap enwik8. For each file the encoder may also send a short list of its most frequent tokens,
  chosen automatically (128 tokens, 0.3 KB, on enwik8).
- **Code.** Ranks 0–127 take one byte, the next 32,256 two bytes, the rest three bytes.

`tools/make_vocab.py` regenerates the files in `vocab/` from tiktoken; `tools/bench.py` runs the benchmarks.

## Paper

M. Kalcher, *Frequency-Ordered Tokenization for Better Text Compression*,
[arXiv:2602.22958](https://arxiv.org/abs/2602.22958).

```bibtex
@misc{kalcher2026fot,
  title         = {Frequency-Ordered Tokenization for Better Text Compression},
  author        = {Kalcher, Maximilian},
  year          = {2026},
  eprint        = {2602.22958},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT}
}
```

## License

MIT. The vocabularies in `vocab/` are derived from OpenAI's [tiktoken](https://github.com/openai/tiktoken) (MIT).
