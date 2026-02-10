
#include <bits/time.h>
#include <emmintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <xmmintrin.h>

// std == 24
#define LOOPS ((uint64_t)1 << 30)
#define OUTLPS (1 << 5)

#define BASEMASK 0x07FFFFFF
#define YVALMASK 0x0FFFFFFF

#include "mersene.c"

const int tab32[32] = {
     0,  9,  1, 10, 13, 21,  2, 29,
    11, 14, 16, 18, 22, 25,  3, 30,
     8, 12, 20, 28, 15, 17, 24,  7,
    19, 27, 23,  6, 26,  5,  4, 31};

uint32_t* bmem;
uint32_t* qmem;

void __attribute__((noinline)) no_op ( int32_t* y, int32_t q1, int32_t q2, int32_t q3, int32_t q0 ) {
  y[1] = q1;
  y[2] = q2;
  y[3] = q3;
  y[0] = q0;
}

void __attribute__((noinline)) baseline ( int32_t* y, int32_t x ) {
  y[1] = y[1] % x;
  y[2] = y[2] % x;
  y[3] = y[3] % x;
  y[0] = y[0] % x;
}

const int intpow = 3;
const uint32_t cc = 1 << intpow;

// faster than baseline
void __attribute__((noinline)) simd1 ( int32_t* y, int32_t x ) {
  // c == 2^pow
  float q = (float)cc / (float)x;
  __m128 yy = _mm_set_ps( y[3], y[2], y[1], y[0] );
  __m128 qq = _mm_set1_ps( q );
  // int32_t div = (unsigned)( (float)y * q );
  qq = _mm_mul_ps( qq, yy );
  // "integer divide" by shifting right pow
  __m128i qqi = _mm_cvtps_epi32( qq );
  qqi = _mm_srli_epi32( qqi, intpow );
  qq = _mm_cvtepi32_ps( qqi );
  // int32_t ml = div * x;
  __m128 xx = _mm_set1_ps( x );
  __m128 ml = _mm_mul_ps( qq, xx );
  // y -= ml; // + x*( (y-ml>=x) - (ml>y) );
  yy = _mm_sub_ps( yy, ml );
  __m128 mask = _mm_setzero_ps();
  // y-ml < 0 => y < ml
  mask = _mm_cmplt_ps( yy, mask );
  // convert -1 (0xFFFFFFFF) to float
  mask = _mm_cvtepi32_ps( _mm_castps_si128( mask ) );
  // (y-ml >= x)
  __m128 mask2 = _mm_cmpge_ps( yy, xx );
  // 0xFFFFFFFF >> 31 == 1 (integer, conv to float)
  __m128i tmp = _mm_srli_epi32( _mm_castps_si128( mask2 ), 31 );
  mask2 = _mm_cvtepi32_ps( tmp );
  mask = _mm_add_ps( mask, mask2 );
  // x*( (y-ml>=x) - (ml>y) )
  mask = _mm_mul_ps( mask, xx );
  yy = _mm_sub_ps( yy, mask );
  // load the vals in the mem
  __m128i yyi = _mm_cvtps_epi32( yy );
  _mm_storeu_si128( (__m128i_u*)y, yyi );
}

// more optimized version of simd1
void __attribute__((noinline)) simd2 ( int32_t* y, int32_t x ) {
  // c == 2^pow
  float q = (float)cc / (float)x;
  __m128 yy = _mm_set_ps( y[3], y[2], y[1], y[0] );
  __m128 qq = _mm_set1_ps( q );
  // int32_t div = (unsigned)( (float)y * q );
  qq = _mm_mul_ps( qq, yy );
  // "integer divide" by shifting right pow
  __m128i qqi = _mm_cvtps_epi32( qq );
  qqi = _mm_srli_epi32( qqi, intpow );
  qq = _mm_cvtepi32_ps( qqi );
  // int32_t ml = div * x;
  __m128 xx = _mm_set1_ps( x );
  __m128 ml = _mm_mul_ps( qq, xx );
  // y -= ml; // + x*( (y-ml>=x) - (ml>y) );
  yy = _mm_sub_ps( yy, ml );
  __m128 mask = _mm_setzero_ps();
  // y-ml < 0 => y < ml
  mask = _mm_cmplt_ps( yy, mask );
  // if y-ml < 0 then set to x otherwise 0
  mask = _mm_and_ps( mask, xx );
  // (y-ml >= x)
  __m128 mask2 = _mm_cmpge_ps( yy, xx );
  mask2 = _mm_and_ps( mask2, xx );
  // x*( -(y-ml>=x) + (ml>y) )
  mask = _mm_sub_ps( mask, mask2 );
  yy = _mm_add_ps( yy, mask );
  // load the vals in the mem
  __m128i yyi = _mm_cvtps_epi32( yy );
  _mm_storeu_si128( (__m128i_u*)y, yyi );
}

// made some of the fp ops into integer ops
void __attribute__((noinline)) simd3 ( int32_t* y, int32_t x ) {
  // c == 2^pow
  float q = (float)cc / (float)x;
  __m128i yyi = _mm_set_epi32( y[3], y[2], y[1], y[0] );
  __m128 yy = _mm_cvtepi32_ps( yyi );
  __m128 qq = _mm_set1_ps( q );
  // int32_t div = (unsigned)( (float)y * q );
  qq = _mm_mul_ps( qq, yy );
  // "integer divide" by shifting right pow
  __m128i qqi = _mm_cvtps_epi32( qq );
  qqi = _mm_srli_epi32( qqi, intpow );
  qq = _mm_cvtepi32_ps( qqi );
  // int32_t ml = div * x;
  __m128i xxi = _mm_set1_epi32( x );
  __m128 xx = _mm_cvtepi32_ps( xxi );
  __m128 ml = _mm_mul_ps( qq, xx );
  __m128i mli = _mm_cvtps_epi32( ml );
  // y -= ml; // + x*( (y-ml>=x) - (ml>y) );
  yyi = _mm_sub_epi32( yyi, mli );
  __m128i mask = _mm_setzero_si128();
  // y-ml < 0 => y < ml
  mask = _mm_cmplt_epi32( yyi, mask );
  // if y-ml < 0 then set to x otherwise 0
  mask = _mm_and_si128( mask, xxi );
  // (y-ml >= x)
  __m128i mask2 = _mm_cmplt_epi32( yyi, xxi );
  mask2 = _mm_andnot_si128( mask2, xxi );
  // x*( -(y-ml>=x) + (ml>y) )
  mask = _mm_sub_epi32( mask, mask2 );
  yyi = _mm_add_epi32( yyi, mask );
  // load the vals in the mem
  _mm_storeu_si128( (__m128i_u*)y, yyi );
}

// wait this actually just worked the whole time and just beats everything
// but it's still inaccurate
void __attribute__((noinline)) simd4 ( int32_t* y, int32_t x ) {
  __m128 yy = _mm_set_ps( y[3], y[2], y[1], y[0] );
  // it turns out this just works!
  __m128 qq = _mm_set1_ps( 1.0f / x );
  // "integer divide" y by x
  qq = _mm_mul_ps( qq, yy );
  __m128i qqi = _mm_cvtps_epi32( qq );
  qq = _mm_cvtepi32_ps( qqi );

  __m128 xx = _mm_set1_ps( x );
  __m128 ml = _mm_mul_ps( qq, xx );
  yy = _mm_sub_ps( yy, ml );
  __m128i yyi = _mm_cvtps_epi32( yy );
  _mm_storeu_si128( (__m128i_u*)y, yyi );

//  __m128  _k = _mm_set1_ps(1.0f / x);
//  __m128  _p = _mm_set1_ps(x);
//  __m128i _a = _mm_loadu_si128((__m128i*)(y));
//  __m128  _e = _mm_mul_ps(_mm_cvtepi32_ps(_a), _k); // e = int(float(d)/float(p));
//  __m128i _s = _mm_cvtps_epi32(_mm_mul_ps(_e, _p));
//  __m128i _c = _mm_sub_epi32(_a, _s );
//  _mm_storeu_si128((__m128i*)(y), _c);
}

// This is not completely, absolutely accurate (but is much, much more
// accurate than the previous iterations), AND IT'S SLOWER.
#define rv_elms (const int)((2 << 6) | (0 << 4) | (3 << 2) | (1 << 0))
void __attribute__((noinline)) simd5 ( int32_t* y, int32_t x ) {
  __m128i yyi = _mm_set_epi32( y[3], y[2], y[1], y[0] );
  __m128i xxi = _mm_set1_epi32( x );
  __m128 xx = _mm_set1_ps( 1.0f / x );
  // this function has a tendency to overestimate the modulus,
  // but we can do it twice.
  for ( int i = 0; i < 2; i++ ) {
    __m128 yy = _mm_cvtepi32_ps( yyi );
    // divide y by x
    __m128 qq = _mm_mul_ps( yy, xx );
    // round
    __m128i qqi = _mm_cvtps_epi32( qq );
    // round(y/x) * x
    __m128i tmp = _mm_mul_epu32( qqi, xxi );
    qqi = _mm_srli_si128( qqi, 4 );
    __m128i mli = _mm_srli_si128( xxi, 4 );
    mli = _mm_mul_epu32( qqi, mli );
    mli = _mm_slli_epi64( mli, 32 );
    mli = _mm_add_epi32( tmp, mli );
    // y = y - round(y/x)*x
    yyi = _mm_sub_epi32( yyi, mli );
  }
  __m128i mask = _mm_cmplt_epi32( yyi, xxi );
  xxi = _mm_andnot_si128( mask, xxi );
  yyi = _mm_sub_epi32( yyi, xxi );
  _mm_storeu_si128( (__m128i_u*)y, yyi );
}

int main () {
//  for ( int pow = 0; pow < 24; pow++ ) {
  struct timespec beg, end;
  int samples = 0;
  double avgt = 0, maxt = 0, mint = 1e200;

  bmem = malloc( LOOPS * 5 * sizeof( uint32_t ) );
  qmem = malloc( LOOPS * 4 * sizeof( uint32_t ) );

  _MM_SET_ROUNDING_MODE( _MM_ROUND_DOWN );

  mt_state random_state;
  initialize_state( &random_state, 0xBEEFCACE );
  //clock_gettime( CLOCK_REALTIME, &beg );
  //initialize_state( &random_state, beg.tv_nsec );
  uint64_t k = 0;
  for ( ; k<<3 < LOOPS*5; k++ ) {
    const uint64_t kk = k << 3;
    bmem[ kk + 0 ] = random_uint32( &random_state );
    bmem[ kk + 1 ] = random_uint32( &random_state );
    bmem[ kk + 2 ] = random_uint32( &random_state );
    bmem[ kk + 3 ] = random_uint32( &random_state );
    bmem[ kk + 4 ] = random_uint32( &random_state );
    bmem[ kk + 5 ] = random_uint32( &random_state );
    bmem[ kk + 6 ] = random_uint32( &random_state );
    bmem[ kk + 7 ] = random_uint32( &random_state );
  }
  for ( k = k<<3; k < LOOPS*5; k++ )
    bmem[ k ] = random_uint32( &random_state );
  

  for ( int j = 0; j < OUTLPS; j++ ) {
    int wrong = 0, maxdist = 0, log = 1;
    double avgdist = 0;
    int32_t y[4] __attribute__((aligned(16)));
    int32_t x;
    for ( uint64_t i = 0; i < LOOPS; i++ ) {
      x = ( bmem[5*i+0] & BASEMASK ) + 1;
      y[1] = bmem[5*i+1] & YVALMASK;
      y[2] = bmem[5*i+2] & YVALMASK;
      y[3] = bmem[5*i+3] & YVALMASK;
      y[0] = bmem[5*i+4] & YVALMASK;
      qmem[ 4*i + 1 ] = y[1] % x;
      qmem[ 4*i + 2 ] = y[2] % x;
      qmem[ 4*i + 3 ] = y[3] % x;
      qmem[ 4*i + 0 ] = y[0] % x;
    }
    clock_gettime( CLOCK_MONOTONIC, &beg );
    // printf( "POW == %i\n", pow );
    for ( uint64_t i = 0; i < LOOPS; i++ ) {
      x = ( bmem[5*i+0] & BASEMASK ) + 1;
      y[1] = bmem[5*i+1] & YVALMASK;
      y[2] = bmem[5*i+2] & YVALMASK;
      y[3] = bmem[5*i+3] & YVALMASK;
      y[0] = bmem[5*i+4] & YVALMASK;

      simd5( y, x );
      //baseline( y, x );
      //no_op( y, qmem[4*i+1], qmem[4*i+2], qmem[4*i+3], qmem[4*i+0] );

//      /* OLD STUFF */
//      uint32_t c = x;
//      c |= c >> 1;
//      c |= c >> 2;
//      c |= c >> 4;
//      c |= c >> 8;
//      c |= c >> 16;
//      c = ( c + 1 ) << 6;
//      uint32_t q = c / x;
//      y = ( y * q ) >> tab32[ (uint32_t)((c-1)*0x07C4ACDD) >> 27 ];
      
      if ( qmem[4*i+1] == y[1] && qmem[4*i+2] == y[2] && 
           qmem[4*i+3] == y[3] && qmem[4*i+0] == y[0] )
        continue;
      else {
        if ( qmem[4*i+1] != y[1] ) {
          if ( log )
            printf( "OH NO! (0x%08X != 0x%08X) (x=0x%08X) (y=0x%08X) "
              "(i=%lu)\n", qmem[4*i+1], y[1], x, bmem[5*i+1] & YVALMASK, i );
          wrong += 1;
          int dist = abs( (int32_t)qmem[4*i+1] - y[1] );
          maxdist = dist > maxdist ? dist : maxdist;
          avgdist += dist;
        }
        if ( qmem[4*i+2] != y[2] ) {
          if ( log )
            printf( "OH NO! (0x%08X != 0x%08X) (x=0x%08X) (y=0x%08X) "
              "(i=%lu)\n", qmem[4*i+2], y[2], x, bmem[5*i+2] & YVALMASK, i );
          wrong += 1;
          int dist = abs( (int32_t)qmem[4*i+2] - y[2] );
          maxdist = dist > maxdist ? dist : maxdist;
          avgdist += dist;
        }
        if ( qmem[4*i+3] != y[3] ) {
          if ( log )
            printf( "OH NO! (0x%08X != 0x%08X) (x=0x%08X) (y=0x%08X) "
              "(i=%lu)\n", qmem[4*i+3], y[3], x, bmem[5*i+3] & YVALMASK, i );
          wrong += 1;
          int dist = abs( (int32_t)qmem[4*i+3] - y[3] );
          maxdist = dist > maxdist ? dist : maxdist;
          avgdist += dist;
        }
        if ( qmem[4*i+0] != y[0] ) {
          if ( log )
            printf( "OH NO! (0x%08X != 0x%08X) (x=0x%08X) (y=0x%08X) "
              "(i=%lu)\n", qmem[4*i+0], y[0], x, bmem[5*i+0] & YVALMASK, i );
          wrong += 1;
          int dist = abs( (int32_t)qmem[4*i+0] - y[0] );
          maxdist = dist > maxdist ? dist : maxdist;
          avgdist += dist;
        }
        log = wrong < 32;
      }
    }
    clock_gettime( CLOCK_MONOTONIC, &end );
    if ( wrong == 0 ) {
      double sec = (double)( end.tv_sec - beg.tv_sec ) * 1.0e3 + 
                   (double)( end.tv_nsec - beg.tv_nsec ) / 1.0e6;
      printf( "TIME: %f\n", sec );
      if ( maxt < sec ) maxt = sec;
      avgt += sec;
      if ( mint > sec ) mint = sec;
      samples += 1;
    } else {
      avgdist /= LOOPS;
      printf( "WRONG TOTAL: %i\nAVG DIST: %f\nMAX DIST: %i\n",
          wrong, avgdist, maxdist );
      break;
    }
  }
  if ( samples > 0 ) {
    avgt /= samples;
    printf( "\n=========================\n" );
    printf( "TMAX: %f\n", maxt );
    printf( "TAVG: %f\n", avgt );
    printf( "TMIN: %f\n", mint );
    printf( "=========================\n\n" );
  }
  free( bmem );
  free( qmem );
  return 0;
}
