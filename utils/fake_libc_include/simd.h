/* stub out the SIMD type and intrinsics */
typedef struct { long long v[2]; } __m128i;

__m128i _mm_setzero_si128(void);
__m128i _mm_loadu_si128(const __m128i *p);
__m128i _mm_dpbusds_epi32(__m128i a, __m128i b, __m128i c);
void     _mm_storeu_si128(__m128i *p, __m128i v);

typedef struct { long long v[2]; } __m512i;

__m512i _mm_setzero_si512(void);
__m512i _mm_loadu_si512(const __m512i *p);
__m512i _mm_dpbusds_epi32(__m512i a, __m512i b, __m512i c);
void     _mm_storeu_si512(__m512i *p, __m512i v);