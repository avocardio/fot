// fot: frequency-ordered tokenization, a text transform for better and faster compression.
// Text is segmented into the tokens of a pretrained BPE vocabulary, each token is replaced
// by its frequency rank, and the ranks are written with a byte-oriented prefix code.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t n, maxlen;     // number of tokens, longest token in bytes
    uint8_t *lens;          // token byte lengths
    uint32_t *offs;         // token offsets into bytes
    uint8_t *bytes;         // concatenated token byte strings (16 bytes of slack)
    int32_t *dat;           // double-array trie, 3 ints per state: check, base, token id (-1 if none)
    uint32_t dat_n;         // number of trie states
    int32_t *tab2;          // trie state after the first two bytes (b0 << 8 | b1), -1 if none
    uint32_t byte_tok[256]; // id of each single-byte token
    uint32_t hash;          // identifies the vocabulary in encoded streams
} fot_vocab;

typedef struct {
    uint32_t n;
    uint32_t *order;        // order[rank] = token id, most frequent first
    uint32_t hash;
} fot_table;

enum { FOT_DP = 0, FOT_GREEDY = 1 };   // segmentation: min-cost DP or greedy longest match
enum { FOT_MAP_AUTO = 0, FOT_MAP_STATIC = 1, FOT_MAP_HYBRID = 2, FOT_MAP_ADAPTIVE = 3 };
enum { FOT_VAR_PREFIX = 0, FOT_VAR_LEB128 = 1 };

typedef struct {
    int algo, map, varint, threads;
    int T;                  // prefix code: ranks below T take one byte (default 128)
    uint32_t K;             // explicit top-K list size for FOT_MAP_HYBRID
    uint32_t block;         // block size in bytes; blocks are coded in parallel (default 1 MiB)
    const uint32_t *ids;    // optional precomputed token ids (skips segmentation)
    size_t nids;
} fot_params;

typedef struct { uint64_t in_bytes, tokens, stream_bytes, map_bytes; int T, E, K; } fot_stats;

fot_vocab *fot_vocab_load(const char *path);
void fot_vocab_free(fot_vocab *v);
fot_table *fot_table_load(const char *path);
void fot_table_free(fot_table *t);
fot_params fot_default_params(void);

// Segment in[0..n) into token ids (out needs n entries); returns the token count.
size_t fot_tokenize(const fot_vocab *v, const fot_table *t, const fot_params *p,
                    const uint8_t *in, size_t n, uint32_t *out);
// Write a rank table ordering the vocabulary by counts (descending).
int fot_table_write(const char *path, const fot_vocab *v, const uint64_t *counts);
// Forward and inverse transform; *out is malloc'd. Corrupt input exits with an error.
int fot_encode(const fot_vocab *v, const fot_table *t, const fot_params *p,
               const uint8_t *in, size_t n, uint8_t **out, size_t *out_n, fot_stats *st);
int fot_decode(const fot_vocab *v, const fot_table *t, int threads,
               const uint8_t *in, size_t n, uint8_t **out, size_t *out_n);
