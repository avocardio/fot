#include "fot.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ROOT 1
#define DIE(...) do { fprintf(stderr, "fot: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

static uint32_t fnv(const uint8_t *p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 16777619u;
    return h;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(sz + 16);
    if (fread(b, 1, sz, f) != (size_t)sz) DIE("short read %s", path);
    fclose(f);
    *n = sz;
    return b;
}

// LEB128 varints, used in the container header
static uint8_t *put_varint(uint8_t *p, uint64_t v) {
    while (v >= 0x80) { *p++ = (uint8_t)(v | 0x80); v >>= 7; }
    *p++ = (uint8_t)v;
    return p;
}
static uint64_t get_varint(const uint8_t **pp, const uint8_t *end) {
    const uint8_t *p = *pp;
    uint64_t v = 0;
    int s = 0;
    do {
        if (p >= end || s > 63) DIE("corrupt varint");
        v |= (uint64_t)(*p & 0x7f) << s;
        s += 7;
    } while (*p++ & 0x80);
    *pp = p;
    return v;
}
static uint32_t varint_len(uint32_t r) { return r < 128 ? 1 : r < 16384 ? 2 : r < 2097152 ? 3 : 4; }

// ---------------- vocabulary and double-array trie ----------------

typedef struct { uint64_t key; uint32_t val; } hent;
typedef struct { hent *e; uint64_t mask; } hmap;

// open-addressing map from (node << 8 | byte) to child node; inserts UINT32_MAX for new keys
static uint32_t *hmap_get(hmap *h, uint64_t key) {
    uint64_t k = key;
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    for (uint64_t i = k & h->mask;; i = (i + 1) & h->mask) {
        if (h->e[i].key == key) return &h->e[i].val;
        if (h->e[i].key == UINT64_MAX) { h->e[i].key = key; h->e[i].val = UINT32_MAX; return &h->e[i].val; }
    }
}

static void build_trie(fot_vocab *v) {
    // 1. explicit trie: child map, token at each node, child bitmap per node
    uint64_t cap = 1;
    while (cap < (uint64_t)v->n * 16) cap <<= 1;
    hmap h = { malloc(cap * sizeof(hent)), cap - 1 };
    memset(h.e, 0xff, cap * sizeof(hent));
    uint32_t nn = 1, ncap = v->n * 8;
    int32_t *term = malloc(ncap * 4);
    uint8_t *kids = calloc(ncap * 32, 1);
    term[0] = -1;
    for (uint32_t i = 0; i < v->n; i++) {
        uint32_t s = 0;
        const uint8_t *b = v->bytes + v->offs[i];
        for (uint32_t k = 0; k < v->lens[i]; k++) {
            uint32_t *c = hmap_get(&h, ((uint64_t)s << 8) | b[k]);
            if (*c == UINT32_MAX) {
                if (nn == ncap) {
                    ncap *= 2;
                    term = realloc(term, ncap * 4);
                    kids = realloc(kids, ncap * 32);
                    memset(kids + nn * 32, 0, (ncap - nn) * 32);
                }
                *c = nn;
                term[nn] = -1;
                kids[s * 32 + b[k] / 8] |= 1 << (b[k] % 8);
                nn++;
            }
            s = *c;
        }
        term[s] = (int32_t)i;
    }
    // 2. double array: place each node's children (BFS order) at base + byte in the first free slots
    uint32_t dn = 1u << 22;
    int32_t *base = malloc(dn * 4), *check = malloc(dn * 4), *tok = malloc(dn * 4);
    for (uint32_t i = 0; i < dn; i++) { base[i] = 0; check[i] = -1; tok[i] = -1; }
    uint32_t *slot = malloc(nn * 4), *queue = malloc(nn * 4), qh = 0, qt = 0, first_free = 2;
    int32_t maxused = ROOT;
    check[ROOT] = 0;
    slot[0] = ROOT;
    queue[qt++] = 0;
    while (qh < qt) {
        uint32_t s = queue[qh++], ds = slot[s];
        uint8_t c[256];
        int nc = 0;
        tok[ds] = term[s];
        for (int x = 0; x < 256; x++)
            if (kids[s * 32 + x / 8] & (1 << (x % 8))) c[nc++] = (uint8_t)x;
        if (!nc) { base[ds] = 0; continue; }
        while (check[first_free] >= 0) first_free++;
        int64_t b = -1;
        for (uint32_t f = first_free;; f++) {
            if (f + 256 >= dn) {
                uint32_t od = dn;
                dn *= 2;
                base = realloc(base, dn * 4); check = realloc(check, dn * 4); tok = realloc(tok, dn * 4);
                for (uint32_t i = od; i < dn; i++) { base[i] = 0; check[i] = -1; tok[i] = -1; }
            }
            if (check[f] >= 0 || (int64_t)f - c[0] < 1) continue;
            int64_t cand = (int64_t)f - c[0];
            int ok = 1;
            for (int k = 1; k < nc && ok; k++) ok = check[cand + c[k]] < 0;
            if (ok) { b = cand; break; }
        }
        base[ds] = (int32_t)b;
        for (int k = 0; k < nc; k++) {
            uint32_t t = (uint32_t)(b + c[k]);
            check[t] = (int32_t)ds;
            if ((int32_t)t > maxused) maxused = (int32_t)t;
            uint32_t child = *hmap_get(&h, ((uint64_t)s << 8) | c[k]);
            slot[child] = t;
            queue[qt++] = child;
        }
    }
    v->dat_n = (uint32_t)maxused + 257;
    v->dat = malloc((size_t)v->dat_n * 12);
    for (uint32_t i = 0; i < v->dat_n; i++) {
        v->dat[3 * i] = check[i];
        v->dat[3 * i + 1] = base[i];
        v->dat[3 * i + 2] = tok[i];
    }
    free(base); free(check); free(tok); free(h.e); free(term); free(kids); free(slot); free(queue);
}

// the trie build takes ~0.2 s, so it is cached next to the vocab file (best effort, keyed by vocab hash)
static void load_or_build_trie(fot_vocab *v, const char *path) {
    char cpath[4096], tpath[4200];
    snprintf(cpath, sizeof cpath, "%s.dat", path);
    size_t cn;
    uint8_t *c = read_file(cpath, &cn);
    if (c && cn >= 12 && !memcmp(c, "FOTD", 4) && !memcmp(c + 4, &v->hash, 4)) {
        memcpy(&v->dat_n, c + 8, 4);
        if (cn == 12 + (size_t)v->dat_n * 12) {
            v->dat = malloc((size_t)v->dat_n * 12);
            memcpy(v->dat, c + 12, (size_t)v->dat_n * 12);
        }
    }
    free(c);
    if (v->dat) return;
    build_trie(v);
    snprintf(tpath, sizeof tpath, "%s.%d.tmp", cpath, (int)getpid());
    FILE *f = fopen(tpath, "wb");
    if (!f) return;
    fwrite("FOTD", 1, 4, f);
    fwrite(&v->hash, 4, 1, f);
    fwrite(&v->dat_n, 4, 1, f);
    fwrite(v->dat, 12, v->dat_n, f);
    if (fclose(f) || rename(tpath, cpath)) remove(tpath);
}

// vocab file: "FOTV", u32 n, then n tokens as (u8 length, bytes) in id order
fot_vocab *fot_vocab_load(const char *path) {
    size_t n;
    uint8_t *b = read_file(path, &n);
    if (!b) DIE("cannot open vocab %s", path);
    if (n < 8 || memcmp(b, "FOTV", 4)) DIE("bad vocab file %s", path);
    fot_vocab *v = calloc(1, sizeof *v);
    memcpy(&v->n, b + 4, 4);
    v->lens = malloc(v->n);
    v->offs = malloc(v->n * 4);
    v->bytes = malloc(n + 16);
    size_t p = 8, o = 0;
    int seen[256] = {0};
    for (uint32_t i = 0; i < v->n; i++) {
        if (p >= n) DIE("truncated vocab");
        uint8_t l = b[p++];
        if (!l || p + l > n) DIE("bad token length");
        v->lens[i] = l;
        v->offs[i] = (uint32_t)o;
        memcpy(v->bytes + o, b + p, l);
        if (l > v->maxlen) v->maxlen = l;
        if (l == 1) { v->byte_tok[b[p]] = i; seen[b[p]] = 1; }
        o += l;
        p += l;
    }
    memset(v->bytes + o, 0, 16);
    for (int c = 0; c < 256; c++)
        if (!seen[c]) DIE("vocab lacks single-byte token %d", c);
    v->hash = fnv(b, n);
    free(b);
    load_or_build_trie(v, path);
    const int32_t *d = v->dat;
    v->tab2 = malloc(65536 * 4);
    for (int a = 0; a < 256; a++) {
        int32_t s1 = d[3 * ROOT + 1] + a;
        for (int c = 0; c < 256; c++) {
            int32_t t = d[3 * s1 + 1] + c;
            v->tab2[a << 8 | c] = (t >= 0 && (uint32_t)t < v->dat_n && d[3 * t] == s1) ? t : -1;
        }
    }
    return v;
}

void fot_vocab_free(fot_vocab *v) {
    if (!v) return;
    free(v->lens); free(v->offs); free(v->bytes); free(v->dat); free(v->tab2); free(v);
}

// table file: "FOTT", u32 n, then n token ids, most frequent first
fot_table *fot_table_load(const char *path) {
    size_t n;
    uint8_t *b = read_file(path, &n);
    if (!b) DIE("cannot open table %s", path);
    if (n < 8 || memcmp(b, "FOTT", 4)) DIE("bad table file %s", path);
    fot_table *t = calloc(1, sizeof *t);
    memcpy(&t->n, b + 4, 4);
    if (n != 8 + (size_t)t->n * 4) DIE("bad table size");
    t->order = malloc(t->n * 4);
    memcpy(t->order, b + 8, t->n * 4);
    t->hash = fnv(b, n);
    free(b);
    uint8_t *seen = calloc(t->n, 1);
    for (uint32_t i = 0; i < t->n; i++) {
        if (t->order[i] >= t->n || seen[t->order[i]]) DIE("table %s is not a permutation", path);
        seen[t->order[i]] = 1;
    }
    free(seen);
    return t;
}

void fot_table_free(fot_table *t) {
    if (t) { free(t->order); free(t); }
}

typedef struct { uint64_t c; uint32_t id; } count_id;
static int by_count(const void *a, const void *b) {   // count descending, then id ascending
    const count_id *x = a, *y = b;
    if (x->c != y->c) return x->c > y->c ? -1 : 1;
    return x->id < y->id ? -1 : x->id > y->id;
}
static void sort_by_count(uint32_t *ids, uint32_t n, const uint64_t *counts) {
    count_id *a = malloc((size_t)n * sizeof(count_id) + 1);
    for (uint32_t i = 0; i < n; i++) a[i] = (count_id){ counts[ids[i]], ids[i] };
    qsort(a, n, sizeof(count_id), by_count);
    for (uint32_t i = 0; i < n; i++) ids[i] = a[i].id;
    free(a);
}

int fot_table_write(const char *path, const fot_vocab *v, const uint64_t *counts) {
    uint32_t *ids = malloc(v->n * 4);
    for (uint32_t i = 0; i < v->n; i++) ids[i] = i;
    sort_by_count(ids, v->n, counts);
    FILE *f = fopen(path, "wb");
    if (!f) { free(ids); return -1; }
    fwrite("FOTT", 1, 4, f);
    fwrite(&v->n, 4, 1, f);
    fwrite(ids, 4, v->n, f);
    fclose(f);
    free(ids);
    return 0;
}

fot_params fot_default_params(void) {
    fot_params p = { FOT_DP, FOT_MAP_AUTO, FOT_VAR_PREFIX, 1, 128, 0, 1u << 20, NULL, 0 };
#ifdef _OPENMP
    p.threads = omp_get_max_threads();
#endif
    return p;
}

// ---------------- segmentation ----------------

// greedy longest match
static size_t tok_greedy(const fot_vocab *v, const uint8_t *p, size_t n, uint32_t *out) {
    const int32_t *d = v->dat;
    size_t i = 0, m = 0;
    while (i < n) {
        int32_t s = ROOT;
        uint32_t best = v->byte_tok[p[i]];
        size_t bl = 1;
        for (size_t k = 0; k < n - i; k++) {
            int32_t t = d[3 * s + 1] + p[i + k];
            if (d[3 * t] != s) break;
            s = t;
            if (d[3 * s + 2] >= 0) { best = (uint32_t)d[3 * s + 2]; bl = k + 1; }
        }
        out[m++] = best;
        i += bl;
    }
    return m;
}

// greedy on the two halves of the buffer with interleaved cursors, which hides trie-walk memory latency
static size_t tok_greedy2(const fot_vocab *v, const uint8_t *p, size_t n, uint32_t *out) {
    if (n < 4096) return tok_greedy(v, p, n, out);
    const int32_t *d = v->dat;
    size_t m = n / 2, iA = 0, iB = m, oA = 0, oB = 0, kA = 0, kB = 0, blA = 1, blB = 1;
    uint32_t *outB = malloc((n - m) * 4 + 4), bA = v->byte_tok[p[0]], bB = v->byte_tok[p[m]];
    int32_t sA = ROOT, sB = ROOT;
    while (iA < m && iB < n) {
        int32_t tA = d[3 * sA + 1] + p[iA + kA], tB = d[3 * sB + 1] + p[iB + kB];
        int goA = d[3 * tA] == sA && iA + kA < m, goB = d[3 * tB] == sB && iB + kB < n;
        if (goA) {
            sA = tA; kA++;
            if (d[3 * sA + 2] >= 0) { bA = (uint32_t)d[3 * sA + 2]; blA = kA; }
        } else {
            out[oA++] = bA; iA += blA; sA = ROOT; kA = 0; blA = 1; bA = v->byte_tok[p[iA]];
        }
        if (goB) {
            sB = tB; kB++;
            if (d[3 * sB + 2] >= 0) { bB = (uint32_t)d[3 * sB + 2]; blB = kB; }
        } else {
            outB[oB++] = bB; iB += blB; sB = ROOT; kB = 0; blB = 1; bB = iB < n ? v->byte_tok[p[iB]] : 0;
        }
    }
    if (iA < m) oA += tok_greedy(v, p + iA, m - iA, out + oA);
    if (iB < n) oB += tok_greedy(v, p + iB, n - iB, outB + oB);
    memcpy(out + oA, outB, oB * 4);
    free(outB);
    return oA + oB;
}

// minimum-cost segmentation: Viterbi over the trie, cost[id] in 1/16 bits
typedef struct { uint32_t *c, *from; uint8_t *len; } dp_scratch;

static inline void relax(dp_scratch *sc, size_t j, uint32_t c, uint32_t id, size_t len) {
    if (c < sc->c[j]) { sc->c[j] = c; sc->from[j] = id; sc->len[j] = (uint8_t)len; }
}

static size_t tok_dp(const fot_vocab *v, const uint16_t *cost, const uint8_t *p, size_t n, uint32_t *out, dp_scratch *sc) {
    const int32_t *d = v->dat;
    if (!n) return 0;
    for (size_t i = 1; i <= n; i++) sc->c[i] = UINT32_MAX;
    sc->c[0] = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t ci = sc->c[i];
        int32_t s = d[3 * ROOT + 1] + p[i];   // every single byte is a token
        relax(sc, i + 1, ci + cost[d[3 * s + 2]], (uint32_t)d[3 * s + 2], 1);
        if (i + 1 >= n || (s = v->tab2[p[i] << 8 | p[i + 1]]) < 0) continue;
        if (d[3 * s + 2] >= 0) relax(sc, i + 2, ci + cost[d[3 * s + 2]], (uint32_t)d[3 * s + 2], 2);
        for (size_t k = 2; k < n - i; k++) {
            int32_t t = d[3 * s + 1] + p[i + k];
            if (d[3 * t] != s) break;
            s = t;
            if (d[3 * s + 2] >= 0) relax(sc, i + k + 1, ci + cost[d[3 * s + 2]], (uint32_t)d[3 * s + 2], k + 1);
        }
    }
    size_t m = 0;
    for (size_t i = n; i > 0; i -= sc->len[i]) out[m++] = sc->from[i];
    for (size_t a = 0, b = m - 1; a < b; a++, b--) { uint32_t x = out[a]; out[a] = out[b]; out[b] = x; }
    return m;
}

// DP per word chunk (a chunk ends before a space that follows a non-space), with a per-thread cache
// of short words. Chunk boundaries barely change the result but make each word's segmentation cacheable.
#define WC_BITS 17
#define WC_MAXLEN 24
typedef struct { uint64_t key[3]; uint8_t len, ntok; uint32_t tok[8]; } __attribute__((aligned(64))) wc_ent;

static size_t tok_dp_cached(const fot_vocab *v, const uint16_t *cost, const uint8_t *p, size_t n,
                            uint32_t *out, dp_scratch *sc, wc_ent *wc) {
    size_t i = 0, m = 0;
    while (i < n) {
        const uint8_t *q = memchr(p + i + 1, ' ', n - i - 1);
        while (q && q[-1] == ' ') q = (size_t)(q + 1 - p) < n ? memchr(q + 1, ' ', n - (size_t)(q + 1 - p)) : NULL;
        size_t j = q ? (size_t)(q - p) : n, L = j - i, k;
        if (L <= WC_MAXLEN) {
            uint64_t key[3] = { 0, 0, 0 };
            memcpy(key, p + i, L);
            uint64_t h = (key[0] * 0x9E3779B97F4A7C15ULL) ^ (key[1] * 0xC2B2AE3D27D4EB4FULL) ^ (key[2] * 0x165667B19E3779F9ULL) ^ L;
            h ^= h >> 31; h *= 0xD6E8FEB86659FD93ULL; h ^= h >> 32;
            wc_ent *e = &wc[h & ((1u << WC_BITS) - 1)];
            if (e->len == L && e->key[0] == key[0] && e->key[1] == key[1] && e->key[2] == key[2]) {
                memcpy(out + m, e->tok, e->ntok * 4);
                m += e->ntok;
                i = j;
                continue;
            }
            k = tok_dp(v, cost, p + i, L, out + m, sc);
            if (k <= 8) {
                memcpy(e->key, key, 24);
                e->len = (uint8_t)L;
                e->ntok = (uint8_t)k;
                memcpy(e->tok, out + m, k * 4);
            }
        } else {
            k = tok_dp(v, cost, p + i, L, out + m, sc);
        }
        m += k;
        i = j;
    }
    return m;
}

// token cost = code length under Zipf's law given the static rank: log2(rank + 2), in 1/16 bits
static uint16_t *make_cost(const fot_vocab *v, const fot_table *t) {
    uint16_t *cost = malloc(v->n * 2);
    for (uint32_t r = 0; r < v->n; r++) cost[t ? t->order[r] : r] = (uint16_t)(16.0 * log2((double)r + 2.0));
    return cost;
}

typedef struct { size_t off, n; uint32_t *tok; size_t ntok; } block;

static block *make_blocks(size_t n, uint32_t bs, size_t *nb) {
    *nb = n ? (n + bs - 1) / bs : 0;
    block *B = calloc(*nb + 1, sizeof(block));
    for (size_t b = 0; b < *nb; b++) {
        B[b].off = b * (size_t)bs;
        B[b].n = b + 1 == *nb ? n - B[b].off : bs;
    }
    return B;
}

static void check_table(const fot_vocab *v, const fot_table *t) {
    if (t && t->n != v->n) DIE("table has %u entries but vocab has %u", t->n, v->n);
}

static void tokenize_blocks(const fot_vocab *v, const fot_table *t, const fot_params *p, const uint8_t *in, block *B, size_t nb) {
    check_table(v, t);
    int nt = p->threads > 0 ? p->threads : 1, dp = p->algo == FOT_DP;
    uint16_t *cost = dp ? make_cost(v, t) : NULL;
    wc_ent **caches = calloc(nt, sizeof(wc_ent *));
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (size_t b = 0; b < nb; b++) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        B[b].tok = malloc(B[b].n * 4 + 4);
        if (!dp) { B[b].ntok = tok_greedy2(v, in + B[b].off, B[b].n, B[b].tok); continue; }
        if (!caches[tid]) {
            caches[tid] = aligned_alloc(64, (1u << WC_BITS) * sizeof(wc_ent));
            memset(caches[tid], 0, (1u << WC_BITS) * sizeof(wc_ent));
        }
        dp_scratch sc = { malloc((B[b].n + 1) * 4), malloc((B[b].n + 1) * 4), malloc(B[b].n + 1) };
        B[b].ntok = tok_dp_cached(v, cost, in + B[b].off, B[b].n, B[b].tok, &sc, caches[tid]);
        free(sc.c); free(sc.from); free(sc.len);
    }
    for (int i = 0; i < nt; i++) free(caches[i]);
    free(caches);
    free(cost);
}

size_t fot_tokenize(const fot_vocab *v, const fot_table *t, const fot_params *p, const uint8_t *in, size_t n, uint32_t *out) {
    size_t nb, m = 0;
    block *B = make_blocks(n, p->block, &nb);
    tokenize_blocks(v, t, p, in, B, nb);
    for (size_t b = 0; b < nb; b++) {
        memcpy(out + m, B[b].tok, B[b].ntok * 4);
        m += B[b].ntok;
        free(B[b].tok);
    }
    free(B);
    return m;
}

// ---------------- ranks and prefix code ----------------
// Ranks below T take 1 byte; lead bytes T..255-E start 2-byte codes, the last E lead bytes 3-byte codes.

static int min_E(uint32_t V, int T) {
    for (int E = 1; E < 256 - T; E++)
        if ((uint64_t)T + (uint64_t)(256 - T - E) * 256 + (uint64_t)E * 65536 >= V) return E;
    return -1;
}

// stream length for rank counts with prefix sums cum (T = 0: LEB128)
static uint64_t stream_bytes(const uint64_t *cum, uint32_t V, int T, int E) {
    uint32_t b1, b2;
    if (T == 0) { b1 = V < 128 ? V : 128; b2 = V < 16384 ? V : 16384; }
    else {
        uint32_t n2 = (uint32_t)(256 - T - E) * 256;
        b1 = V < (uint32_t)T ? V : (uint32_t)T;
        b2 = V < T + n2 ? V : T + n2;
    }
    return cum[b1] + 2 * (cum[b2] - cum[b1]) + 3 * (cum[V] - cum[b2]);
}

// rank order: the explicit list first, then the static order without the listed ids
static void build_rank(const fot_table *t, const uint32_t *listed, uint32_t K, uint32_t *rank, uint32_t V) {
    uint8_t *seen = calloc(V, 1);
    uint32_t r = 0;
    for (uint32_t i = 0; i < K; i++) { rank[listed[i]] = r++; seen[listed[i]] = 1; }
    for (uint32_t i = 0; i < V; i++) {
        uint32_t id = t ? t->order[i] : i;
        if (!seen[id]) rank[id] = r++;
    }
    free(seen);
}

// Choose the explicit list size K. Proxy fitted on zstd, xz, zlib and brotli: compressors realize
// about a tenth of the stream bytes saved by reordering, and the id list compresses to ~85%.
static uint32_t choose_K(const fot_table *t, const fot_params *p, const uint64_t *cnt, const uint32_t *sorted,
                         uint32_t nseen, uint32_t V, int T, int E, uint32_t *rank) {
    uint32_t Ks[40], nK = 0, best_K = 0;
    if (p->map == FOT_MAP_STATIC) Ks[nK++] = 0;
    else if (p->map == FOT_MAP_ADAPTIVE) Ks[nK++] = nseen;
    else if (p->map == FOT_MAP_HYBRID) Ks[nK++] = p->K < nseen ? p->K : nseen;
    else {
        Ks[nK++] = 0;
        for (uint32_t k = 32; k < nseen; k *= 2) Ks[nK++] = k;
        Ks[nK++] = nseen;
    }
    uint64_t *cum = malloc((V + 1) * 8), *by_rank = malloc(V * 8), best = UINT64_MAX;
    for (uint32_t ki = 0; ki < nK; ki++) {
        uint32_t K = Ks[ki];
        build_rank(t, sorted, K, rank, V);
        for (uint32_t i = 0; i < V; i++) by_rank[rank[i]] = cnt[i];
        cum[0] = 0;
        for (uint32_t r = 0; r < V; r++) cum[r + 1] = cum[r] + by_rank[r];
        uint64_t list_bytes = varint_len(K);
        for (uint32_t i = 0; i < K; i++) list_bytes += varint_len(sorted[i]);
        uint64_t proxy = stream_bytes(cum, V, T, E) / 10 + list_bytes * 85 / 100;
        if (proxy < best) { best = proxy; best_K = K; }
    }
    free(cum);
    free(by_rank);
    build_rank(t, sorted, best_K, rank, V);
    return best_K;
}

static uint8_t *write_ranks(const uint32_t *tok, size_t ntok, const uint32_t *rank, int T, int E, size_t *len) {
    uint32_t n2 = T ? (uint32_t)(256 - T - E) * 256 : 0;
    uint8_t *o = malloc(ntok * 4 + 16), *q = o;
    for (size_t i = 0; i < ntok; i++) {
        uint32_t r = rank[tok[i]];
        if (!T) q = put_varint(q, r);
        else if (r < (uint32_t)T) *q++ = (uint8_t)r;
        else if (r < T + n2) { r -= T; *q++ = (uint8_t)(T + (r >> 8)); *q++ = (uint8_t)r; }
        else { r -= T + n2; *q++ = (uint8_t)(256 - E + (r >> 16)); *q++ = (uint8_t)(r >> 8); *q++ = (uint8_t)r; }
    }
    *len = (size_t)(q - o);
    return o;
}

// precomputed ids: split into blocks of about p->block bytes
static block *blocks_from_ids(const fot_vocab *v, const fot_params *p, size_t n, size_t *nb) {
    block *B = calloc(p->nids + 1, sizeof(block));
    size_t off = 0, i = 0;
    *nb = 0;
    while (i < p->nids) {
        size_t j = i, bytes = 0;
        while (j < p->nids && bytes < p->block) {
            if (p->ids[j] >= v->n) DIE("id out of range");
            bytes += v->lens[p->ids[j++]];
        }
        block *b = &B[(*nb)++];
        b->off = off; b->n = bytes; b->ntok = j - i;
        b->tok = malloc((j - i) * 4 + 4);
        memcpy(b->tok, p->ids + i, (j - i) * 4);
        off += bytes;
        i = j;
    }
    if (off != n) DIE("ids do not reproduce input length (%zu vs %zu)", off, n);
    return B;
}

// Container: "FOT1", vocab hash, table hash, T, E, then varints: input bytes, block count, K, the K listed ids,
// and per block (tokens, stream bytes, input bytes); then the block streams.
int fot_encode(const fot_vocab *v, const fot_table *t, const fot_params *p, const uint8_t *in, size_t n,
               uint8_t **out, size_t *out_n, fot_stats *st) {
    uint32_t V = v->n;
    size_t nb;
    check_table(v, t);
    block *B;
    if (p->ids) B = blocks_from_ids(v, p, n, &nb);
    else { B = make_blocks(n, p->block, &nb); tokenize_blocks(v, t, p, in, B, nb); }

    uint64_t *cnt = calloc(V, 8), ntok = 0;
    for (size_t b = 0; b < nb; b++) {
        for (size_t i = 0; i < B[b].ntok; i++) cnt[B[b].tok[i]]++;
        ntok += B[b].ntok;
    }
    uint32_t *sorted = malloc(V * 4), nseen = 0;
    for (uint32_t i = 0; i < V; i++) if (cnt[i]) sorted[nseen++] = i;
    sort_by_count(sorted, nseen, cnt);

    int T = p->varint == FOT_VAR_LEB128 ? 0 : p->T, E = T ? min_E(V, T) : 0;
    if (T && (E < 0 || T < 1)) DIE("bad prefix threshold T=%d", T);
    uint32_t *rank = malloc(V * 4);
    uint32_t K = choose_K(t, p, cnt, sorted, nseen, V, T, E, rank);

    uint8_t **bufs = malloc(nb * sizeof(uint8_t *));
    size_t *blen = malloc(nb * sizeof(size_t));
    int nt = p->threads > 0 ? p->threads : 1;
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (size_t b = 0; b < nb; b++) {
        bufs[b] = write_ranks(B[b].tok, B[b].ntok, rank, T, E, &blen[b]);
        free(B[b].tok);
    }

    size_t total = 64 + (size_t)K * 5 + nb * 30;
    for (size_t b = 0; b < nb; b++) total += blen[b];
    uint8_t *o = malloc(total), *q = o;
    uint32_t th = t ? t->hash : 0;
    memcpy(q, "FOT1", 4); q += 4;
    memcpy(q, &v->hash, 4); q += 4;
    memcpy(q, &th, 4); q += 4;
    *q++ = (uint8_t)T;
    *q++ = (uint8_t)E;
    q = put_varint(q, n);
    q = put_varint(q, nb);
    q = put_varint(q, K);
    uint64_t list_bytes = 0;
    for (uint32_t i = 0; i < K; i++) { q = put_varint(q, sorted[i]); list_bytes += varint_len(sorted[i]); }
    for (size_t b = 0; b < nb; b++) { q = put_varint(q, B[b].ntok); q = put_varint(q, blen[b]); q = put_varint(q, B[b].n); }
    size_t hlen = (size_t)(q - o);
    for (size_t b = 0; b < nb; b++) { memcpy(q, bufs[b], blen[b]); q += blen[b]; free(bufs[b]); }
    *out = o;
    *out_n = (size_t)(q - o);
    if (st) *st = (fot_stats){ n, ntok, *out_n - hlen, list_bytes, T, E, (int)K };
    free(bufs); free(blen); free(B); free(cnt); free(sorted); free(rank);
    return 0;
}

int fot_decode(const fot_vocab *v, const fot_table *t, int threads, const uint8_t *in, size_t n, uint8_t **out, size_t *out_n) {
    const uint8_t *p = in, *end = in + n;
    uint32_t V = v->n, vh, th;
    if (n < 14 || memcmp(p, "FOT1", 4)) DIE("not a FOT1 stream");
    memcpy(&vh, p + 4, 4);
    memcpy(&th, p + 8, 4);
    p += 12;
    if (vh != v->hash) DIE("vocab mismatch (stream was encoded with a different vocabulary)");
    if (th && (!t || th != t->hash)) DIE("static table mismatch");
    if (!th) t = NULL;
    check_table(v, t);
    uint32_t T = *p++, E = *p++, n2 = T ? (256 - T - E) * 256 : 0;
    uint64_t total = get_varint(&p, end), nb = get_varint(&p, end), K = get_varint(&p, end);
    if (K > V || (T && (E < 1 || T + E > 255))) DIE("corrupt header");

    uint32_t *listed = malloc(K * 4 + 4), *rank = malloc(V * 4), *r2id = malloc(V * 4);
    for (uint64_t i = 0; i < K; i++)
        if ((listed[i] = (uint32_t)get_varint(&p, end)) >= V) DIE("corrupt map");
    build_rank(t, listed, (uint32_t)K, rank, V);
    for (uint32_t i = 0; i < V; i++) r2id[rank[i]] = i;
    free(rank);
    free(listed);

    block *B = calloc(nb + 1, sizeof(block));
    size_t *soff = malloc((nb + 1) * sizeof(size_t));
    uint64_t off = 0, so = 0;
    for (uint64_t b = 0; b < nb; b++) {
        B[b].ntok = get_varint(&p, end);
        soff[b] = get_varint(&p, end);
        B[b].n = get_varint(&p, end);
        B[b].off = off;
        off += B[b].n;
    }
    if (off != total) DIE("corrupt block table");
    for (uint64_t b = 0; b < nb; b++) { size_t l = soff[b]; soff[b] = so; so += l; }
    if ((size_t)(p - in) + so != n) DIE("corrupt stream length");

    uint8_t *o = malloc(total + 16);
    int err = 0, nt = threads > 0 ? threads : 1;
    #pragma omp parallel for schedule(dynamic) num_threads(nt)
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *s = p + soff[b], *se = b + 1 < nb ? p + soff[b + 1] : end;
        uint8_t *q = o + B[b].off, *qe = q + B[b].n;
        for (size_t i = 0; i < B[b].ntok; i++) {
            uint32_t r = 0;
            if (s >= se) { err = 1; break; }
            if (!T) {
                for (int sh = 0;; sh += 7) {
                    if (s >= se || sh > 21) { err = 1; break; }
                    uint32_t c = *s++;
                    r |= (c & 0x7f) << sh;
                    if (!(c & 0x80)) break;
                }
                if (err) break;
            } else {
                uint32_t c = *s++, extra = c < T ? 0 : c < 256 - E ? 1 : 2;
                if ((size_t)(se - s) < extra) { err = 1; break; }
                if (extra == 0) r = c;
                else if (extra == 1) r = T + ((c - T) << 8 | *s++);
                else { r = T + n2 + (((c - (256 - E)) << 16) | ((uint32_t)s[0] << 8) | s[1]); s += 2; }
            }
            if (r >= V) { err = 1; break; }
            uint32_t id = r2id[r], l = v->lens[id];
            const uint8_t *tp = v->bytes + v->offs[id];
            if (q + l > qe) { err = 1; break; }
            if (l <= 16 && q + 16 <= qe) memcpy(q, tp, 16);   // fixed-size copy is faster; bytes has slack
            else memcpy(q, tp, l);
            q += l;
        }
        if (q != qe || s != se) err = 1;
    }
    free(B); free(soff); free(r2id);
    if (err) { free(o); DIE("corrupt stream"); }
    *out = o;
    *out_n = total;
    return 0;
}
