# simd-modulus
A small snippet where I made a few versions of modulus with SIMD. These take `y[n] mod x` for a constant x, but could easily be modified to make x a vector.

These functions were tested for `y < 0x007FFFFF` and `x < 0x0000FFFF` as shown in the code.

mersene.c was taken directly from wikipedia and is not in any way my code.
