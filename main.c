// fot command line: the FOT transform followed by a standard compressor.
#include "fot.h"
#include <brotli/decode.h>
#include <brotli/encode.h>
#include <bzlib.h>
#include <lzma.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>
#include <zstd.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static void die(const char *msg) {
    fprintf(stderr, "fot: %s\n", msg);
    exit(1);
}

static uint8_t *read_all(const char *path, size_t *n) {
    FILE *f = strcmp(path, "-") ? fopen(path, "rb") : stdin;
    if (!f) die("cannot open input");
    size_t cap = 1 << 20, len = 0, r;
    uint8_t *b = malloc(cap);
    while ((r = fread(b + len, 1, cap - len, f)) > 0)
        if ((len += r) == cap) b = realloc(b, cap *= 2);
    if (f != stdin) fclose(f);
    *n = len;
    return b;
}

static void write_all(const char *path, const uint8_t *b, size_t n) {
    FILE *f = strcmp(path, "-") ? fopen(path, "wb") : stdout;
    if (!f) die("cannot open output");
    if (fwrite(b, 1, n, f) != n) die("write failed");
    if (f != stdout) fclose(f);
}

// ---------------- compressors ----------------

enum { C_NONE, C_ZSTD, C_XZ, C_XZE, C_BROTLI, C_ZLIB, C_BZ2, C_COUNT };
static const char *codec_names[C_COUNT] = { "none", "zstd", "xz", "xze", "brotli", "zlib", "bz2" };

static int parse_codec(const char *s, int *level) {
    const char *colon = strchr(s, ':');
    size_t len = colon ? (size_t)(colon - s) : strlen(s);
    *level = colon ? atoi(colon + 1) : -1;
    for (int i = 0; i < C_COUNT; i++)
        if (strlen(codec_names[i]) == len && !strncmp(s, codec_names[i], len)) return i;
    die("unknown codec (zstd|xz|xze|brotli|zlib|bz2|none)");
    return 0;
}

// wbits: window log for zstd and brotli (0 = codec default)
static uint8_t *codec_compress(int codec, int level, int wbits, int threads, const uint8_t *in, size_t n, size_t *out_n) {
    uint8_t *o = NULL;
    switch (codec) {
    case C_NONE:
        o = malloc(n + 1);
        memcpy(o, in, n);
        *out_n = n;
        break;
    case C_ZSTD: {
        size_t cap = ZSTD_compressBound(n);
        o = malloc(cap);
        ZSTD_CCtx *c = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, level < 0 ? 3 : level);
        if (threads > 1) ZSTD_CCtx_setParameter(c, ZSTD_c_nbWorkers, threads);
        if (wbits) ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, wbits);
        size_t r = ZSTD_compress2(c, o, cap, in, n);
        if (ZSTD_isError(r)) die(ZSTD_getErrorName(r));
        ZSTD_freeCCtx(c);
        *out_n = r;
        break;
    }
    case C_XZ:
    case C_XZE: {
        uint32_t preset = (uint32_t)(level < 0 ? 6 : level) | (codec == C_XZE ? LZMA_PRESET_EXTREME : 0);
        size_t cap = lzma_stream_buffer_bound(n), pos = 0;
        o = malloc(cap);
        if (lzma_easy_buffer_encode(preset, LZMA_CHECK_CRC64, NULL, in, n, o, &pos, cap) != LZMA_OK) die("xz failed");
        *out_n = pos;
        break;
    }
    case C_BROTLI: {
        if (!wbits) wbits = 24;
        size_t cap = BrotliEncoderMaxCompressedSize(n) + 64, avail_in = n, avail_out = cap;
        o = malloc(cap);
        const uint8_t *ip = in;
        uint8_t *op = o;
        BrotliEncoderState *s = BrotliEncoderCreateInstance(NULL, NULL, NULL);
        BrotliEncoderSetParameter(s, BROTLI_PARAM_QUALITY, (uint32_t)(level < 0 ? 11 : level));
        if (wbits > 24) BrotliEncoderSetParameter(s, BROTLI_PARAM_LARGE_WINDOW, 1);
        BrotliEncoderSetParameter(s, BROTLI_PARAM_LGWIN, (uint32_t)wbits);
        BrotliEncoderSetParameter(s, BROTLI_PARAM_SIZE_HINT, (uint32_t)(n < 0xffffffffu ? n : 0xffffffffu));
        while (!BrotliEncoderIsFinished(s))
            if (!BrotliEncoderCompressStream(s, BROTLI_OPERATION_FINISH, &avail_in, &ip, &avail_out, &op, NULL)) die("brotli failed");
        BrotliEncoderDestroyInstance(s);
        *out_n = cap - avail_out;
        break;
    }
    case C_ZLIB: {
        uLongf len = compressBound(n);
        o = malloc(len);
        if (compress2(o, &len, in, n, level < 0 ? 6 : level) != Z_OK) die("zlib failed");
        *out_n = len;
        break;
    }
    case C_BZ2: {
        unsigned len = (unsigned)(n * 1.01 + 600);
        o = malloc(len);
        if (BZ2_bzBuffToBuffCompress((char *)o, &len, (char *)in, (unsigned)n, level < 0 ? 9 : level, 0, 30) != BZ_OK) die("bz2 failed");
        *out_n = len;
        break;
    }
    }
    return o;
}

static uint8_t *codec_decompress(int codec, int wbits, const uint8_t *in, size_t n, size_t raw) {
    uint8_t *o = malloc(raw + 1);
    switch (codec) {
    case C_NONE:
        if (n != raw) die("corrupt file");
        memcpy(o, in, n);
        break;
    case C_ZSTD: {
        size_t r = ZSTD_decompress(o, raw, in, n);
        if (ZSTD_isError(r) || r != raw) die("zstd decompression failed");
        break;
    }
    case C_XZ:
    case C_XZE: {
        uint64_t memlimit = UINT64_MAX;
        size_t ip = 0, op = 0;
        if (lzma_stream_buffer_decode(&memlimit, 0, NULL, in, &ip, n, o, &op, raw) != LZMA_OK || op != raw) die("xz decompression failed");
        break;
    }
    case C_BROTLI: {
        size_t avail_in = n, avail_out = raw;
        const uint8_t *ip = in;
        uint8_t *op = o;
        BrotliDecoderState *s = BrotliDecoderCreateInstance(NULL, NULL, NULL);
        if (wbits > 24) BrotliDecoderSetParameter(s, BROTLI_DECODER_PARAM_LARGE_WINDOW, 1);
        if (BrotliDecoderDecompressStream(s, &avail_in, &ip, &avail_out, &op, NULL) != BROTLI_DECODER_RESULT_SUCCESS || avail_out)
            die("brotli decompression failed");
        BrotliDecoderDestroyInstance(s);
        break;
    }
    case C_ZLIB: {
        uLongf len = raw;
        if (uncompress(o, &len, in, n) != Z_OK || len != raw) die("zlib decompression failed");
        break;
    }
    case C_BZ2: {
        unsigned len = (unsigned)raw;
        if (BZ2_bzBuffToBuffDecompress((char *)o, &len, (char *)in, (unsigned)n, 0, 0) != BZ_OK || len != raw) die("bz2 decompression failed");
        break;
    }
    default:
        die("unknown codec in file");
    }
    return o;
}

// ---------------- command line ----------------

static void usage(void) {
    fprintf(stderr,
        "usage: fot enc|dec|raw|unraw|table [options] in out        ('-' = stdin/stdout)\n"
        "  enc, dec      FOT transform + compressor, and inverse\n"
        "  raw, unraw    compressor only (baseline)\n"
        "  table         build a rank table from the input files, written to -T\n"
        "options:\n"
        "  -c codec[:level]  zstd|xz|xze|brotli|zlib|bz2|none (default none); -w window log\n"
        "  -v vocab          vocabulary (default vocab/cl100k_base.vocab next to the binary)\n"
        "  -T table          rank table (default: the vocab path with .table)\n"
        "  -a dp|greedy      segmentation: minimum cost (default) or greedy longest match\n"
        "  -m auto|static|hybrid|adaptive   rank map; -K n sends the n most frequent ids (hybrid)\n"
        "  -e prefix|leb     rank code (default prefix; -t T one-byte codes, default 128)\n"
        "  -j threads  -B block bytes (default 1048576)  -I ids.u32 (precomputed token ids)  -q quiet\n");
    exit(2);
}

static int pick(const char *s, const char *const *names, int n) {
    for (int i = 0; i < n; i++)
        if (!strcmp(s, names[i])) return i;
    usage();
    return 0;
}

static void default_paths(const char **vocab, const char **table) {
    static char vpath[4096], tpath[4096];
    if (!*vocab) {
        ssize_t l = readlink("/proc/self/exe", vpath, sizeof vpath - 64);
        char *slash = l > 0 ? (vpath[l] = 0, strrchr(vpath, '/')) : NULL;
        if (slash) strcpy(slash, "/vocab/cl100k_base.vocab");
        else strcpy(vpath, "vocab/cl100k_base.vocab");
        *vocab = vpath;
    }
    size_t vl = strlen(*vocab);
    if (!*table && vl > 6 && vl < sizeof tpath && !strcmp(*vocab + vl - 6, ".vocab")) {
        memcpy(tpath, *vocab, vl - 6);
        strcpy(tpath + vl - 6, ".table");
        *table = tpath;
    }
}

static fot_table *load_table(const char *path) { return path && access(path, R_OK) == 0 ? fot_table_load(path) : NULL; }

// file: "FOTC", codec, window log, transformed flag, varint length of the compressor input, compressed data
static int encode(const char *cmd, const char *src, const char *dst, const char *vocab, const char *table,
                  fot_params *p, int codec, int level, int wbits, int quiet) {
    size_t n, trn, cn;
    double t0 = now();
    uint8_t *in = read_all(src, &n), *tr = in;
    double t_read = now() - t0, t_tr = 0;
    trn = n;
    if (!strcmp(cmd, "enc")) {
        t0 = now();
        fot_vocab *v = fot_vocab_load(vocab);
        fot_table *t = load_table(table);
        double t_load = now() - t0;
        fot_stats st;
        t0 = now();
        fot_encode(v, t, p, in, n, &tr, &trn, &st);
        t_tr = now() - t0;
        if (!quiet)
            fprintf(stderr, "load %.3fs | transform %.3fs (%.1f MB/s, %d threads) | %llu tokens (%.2f B/token) | "
                    "stream %llu B (%.2f%%) | list K=%d (%llu B)\n", t_load, t_tr, n / 1e6 / t_tr, p->threads,
                    (unsigned long long)st.tokens, (double)n / st.tokens, (unsigned long long)st.stream_bytes,
                    100.0 * st.stream_bytes / n, st.K, (unsigned long long)st.map_bytes);
        fot_vocab_free(v);
        fot_table_free(t);
    }
    t0 = now();
    uint8_t *c = codec_compress(codec, level, wbits, p->threads, tr, trn, &cn);
    double t_c = now() - t0;
    uint8_t hdr[32], *q = hdr;
    memcpy(q, "FOTC", 4);
    q += 4;
    *q++ = (uint8_t)codec;
    *q++ = (uint8_t)wbits;
    *q++ = (uint8_t)(cmd[0] == 'e');
    for (uint64_t x = trn;; x >>= 7) {
        *q++ = (uint8_t)((x & 0x7f) | (x >= 0x80 ? 0x80 : 0));
        if (x < 0x80) break;
    }
    FILE *f = strcmp(dst, "-") ? fopen(dst, "wb") : stdout;
    if (!f) die("cannot open output");
    if (fwrite(hdr, 1, q - hdr, f) != (size_t)(q - hdr) || fwrite(c, 1, cn, f) != cn) die("write failed");
    if (f != stdout) fclose(f);
    size_t out_n = cn + (q - hdr);
    if (!quiet)
        fprintf(stderr, "%s:%d %.3fs | total %.3fs (%.1f MB/s) | %zu -> %zu bytes = %.4f%%\n", codec_names[codec], level,
                t_c, t_tr + t_c, n / 1e6 / (t_tr + t_c), n, out_n, 100.0 * out_n / n);
    if (f != stdout)   // machine-readable summary for tools/bench.py
        printf("%s\t%s\t%d\t%zu\t%zu\t%.6f\t%.4f\t%.4f\t%.4f\n", cmd, codec_names[codec], level, n, out_n,
               100.0 * out_n / n, t_tr, t_c, t_read);
    free(c);
    if (tr != in) free(tr);
    free(in);
    return 0;
}

static int decode(const char *cmd, const char *src, const char *dst, const char *vocab, const char *table, int threads, int quiet) {
    size_t n, out_n;
    uint8_t *in = read_all(src, &n);
    if (n < 8 || memcmp(in, "FOTC", 4)) die("not a fot file");
    int codec = in[4], wbits = in[5], transformed = in[6];
    const uint8_t *q = in + 7, *end = in + n;
    uint64_t raw = 0;
    for (int s = 0;; s += 7) {
        if (q >= end || s > 63) die("corrupt file");
        raw |= (uint64_t)(*q & 0x7f) << s;
        if (!(*q++ & 0x80)) break;
    }
    double t0 = now();
    uint8_t *d = codec_decompress(codec, wbits, q, (size_t)(end - q), raw), *out = d;
    double t_d = now() - t0, t_tr = 0;
    out_n = raw;
    if (transformed) {
        fot_vocab *v = fot_vocab_load(vocab);
        fot_table *t = load_table(table);
        t0 = now();
        fot_decode(v, t, threads, d, raw, &out, &out_n);
        t_tr = now() - t0;
        free(d);
        fot_vocab_free(v);
        fot_table_free(t);
    }
    write_all(dst, out, out_n);
    if (!quiet)
        fprintf(stderr, "%s decompress %.3fs | inverse transform %.3fs | %zu bytes\n", codec_names[codec], t_d, t_tr, out_n);
    if (strcmp(dst, "-")) printf("%s\t%s\t%zu\t%.4f\t%.4f\n", cmd, codec_names[codec], out_n, t_d, t_tr);
    free(out);
    free(in);
    return 0;
}

// count tokens over the inputs (segmented with the current table, if any) and write the table to -T
static int build_table(int argc, char **argv, const char *vocab, const char *table, fot_params *p, int quiet) {
    if (argc < 1 || !table) usage();
    fot_vocab *v = fot_vocab_load(vocab);
    fot_table *t = p->algo == FOT_DP ? load_table(table) : NULL;
    uint64_t *cnt = calloc(v->n, 8);
    for (int i = 0; i < argc; i++) {
        size_t n;
        uint8_t *in = read_all(argv[i], &n);
        uint32_t *tok = malloc(n * 4 + 4);
        size_t m = fot_tokenize(v, t, p, in, n, tok);
        for (size_t j = 0; j < m; j++) cnt[tok[j]]++;
        free(tok);
        free(in);
    }
    if (fot_table_write(table, v, cnt)) die("cannot write table");
    if (!quiet) fprintf(stderr, "wrote %s\n", table);
    return 0;
}

int main(int argc, char **argv) {
    static const char *const algos[] = { "dp", "greedy" }, *const maps[] = { "auto", "static", "hybrid", "adaptive" },
                             *const codes[] = { "prefix", "leb" };
    if (argc < 2) usage();
    const char *cmd = argv[1], *vocab = NULL, *table = NULL, *codec_arg = "none";
    int quiet = 0, wbits = 0, opt;
    fot_params p = fot_default_params();
    while ((opt = getopt(argc - 1, argv + 1, "v:T:a:m:K:e:t:B:c:w:j:qI:")) != -1) {
        switch (opt) {
        case 'v': vocab = optarg; break;
        case 'T': table = optarg; break;
        case 'a': p.algo = pick(optarg, algos, 2); break;
        case 'm': p.map = pick(optarg, maps, 4); break;
        case 'K': p.K = (uint32_t)atoi(optarg); p.map = FOT_MAP_HYBRID; break;
        case 'e': p.varint = pick(optarg, codes, 2); break;
        case 't': p.T = atoi(optarg); break;
        case 'B': p.block = (uint32_t)atoi(optarg); break;
        case 'c': codec_arg = optarg; break;
        case 'w': wbits = atoi(optarg); break;
        case 'j': p.threads = atoi(optarg); break;
        case 'q': quiet = 1; break;
        case 'I': { size_t n; p.ids = (const uint32_t *)read_all(optarg, &n); p.nids = n / 4; break; }
        default: usage();
        }
    }
    argv += optind + 1;
    argc -= optind + 1;
    default_paths(&vocab, &table);
    int level, codec = parse_codec(codec_arg, &level);
    if (!strcmp(cmd, "table")) return build_table(argc, argv, vocab, table, &p, quiet);
    if (argc != 2) usage();
    if (!strcmp(cmd, "enc") || !strcmp(cmd, "raw")) return encode(cmd, argv[0], argv[1], vocab, table, &p, codec, level, wbits, quiet);
    if (!strcmp(cmd, "dec") || !strcmp(cmd, "unraw")) return decode(cmd, argv[0], argv[1], vocab, table, p.threads, quiet);
    usage();
    return 0;
}
