CC ?= cc
CFLAGS ?= -O3 -march=native -Wall -Wextra
override CFLAGS += -std=gnu11 -fopenmp
LDLIBS = -lzstd -llzma -lz -lbz2 -lbrotlienc -lbrotlidec -lm

fot: fot.c main.c fot.h
	$(CC) $(CFLAGS) -o $@ fot.c main.c $(LDLIBS)

test: fot
	python3 tests/test.py

clean:
	rm -f fot vocab/*.dat

.PHONY: test clean
