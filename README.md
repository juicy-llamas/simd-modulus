# simd-modulus
A small snippet where I made a few versions of modulus with SIMD. These take `y[n] mod x` for a constant x, but could easily be modified to make x a vector.

These functions should be "reasonably" accurate (as in accurate) for y < 0x00FFFFFF and x < 0x0000FFFF. However, floating point error means that these functions are inherently somewhat inaccurate.

simd1-4 are "more inaccurate," simd5 is an attempt to make a fully accurate function, but.....it still does have the risk that the 32 bit unsigned int can overflow. With sufficiently small values, it should be accurate.

mersene.c was taken directly from wikipedia and is not in any way my code.
