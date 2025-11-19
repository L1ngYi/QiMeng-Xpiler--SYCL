/* utils/fake_libc_include/simd_bang.h */
#ifndef SIMD_BANG_H
#define SIMD_BANG_H

/* 1) half */
/* Use a 16‑bit integer to stand in for __fp16 */
typedef unsigned short half;

/* 2) BANGC built‑ins */
#define clusterId 0
#define coreId    1

#endif /* SIMD_BANG_H