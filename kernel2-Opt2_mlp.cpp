#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include "bsg_set_tile_x_y.h"
#include "bsg_group_strider.hpp"
#include <cstring>
#include <cstdint>
#include <math.h>

// #ifdef WARM_CACHE
// __attribute__((noinline))
// static void warmup(float *A, float *B, float *C, int N)
// {
//   for (int i = __bsg_id*CACHE_LINE_WORDS; i < N; i += bsg_tiles_X*bsg_tiles_Y*CACHE_LINE_WORDS) {
//       asm volatile ("lw x0, %[p]" :: [p] "m" (A[i]));
//       asm volatile ("lw x0, %[p]" :: [p] "m" (B[i]));
//       asm volatile ("sw x0, %[p]" :: [p] "m" (C[i]));
//   }
//   bsg_fence();
// }
// #endif

extern "C" __attribute__ ((noinline))
int
kernel_gblur(short int * A, short int * B, int H) {

  bsg_barrier_hw_tile_group_init();
#ifdef WARM_CACHE
  warmup(A, B, C, N);
#endif
  bsg_barrier_hw_tile_group_sync();
  bsg_cuda_print_stat_kernel_start();

  // Each tile does a portion of vector_add
  // int len = N / (bsg_tiles_X*bsg_tiles_Y);
  // float *myA = &A[__bsg_id*len];
  // float *myB = &B[__bsg_id*len];
  // float *myC = &C[__bsg_id*len];
  float kernel[9] = {
      0.09474166, 0.11831801, 0.09474166,
      0.11831801, 0.14776132, 0.11831801,
      0.09474166, 0.11831801, 0.09474166
  }; 

  if ((__bsg_id >= 2) && (__bsg_id <= 125)) {
    int idx_start = 32 * __bsg_id + 1 - (__bsg_id % 2);
    int idx_end = idx_start + 31;

    for (int idx = idx_start; idx < idx_end; idx += 11) {
      register short int a00 = A[idx-H-1];
      register short int a01 = A[idx  -1];
      register short int a02 = A[idx+H-1];
      register short int a03 = A[idx-H  ];
      register short int a04 = A[idx    ];
      register short int a05 = A[idx+H  ];
      register short int a06 = A[idx-H+1];
      register short int a07 = A[idx  +1];
      register short int a08 = A[idx+H+1];
      register short int b00 = 0;

      register short int a09 = A[idx-H+2];
      register short int a10 = A[idx  +2];
      register short int a11 = A[idx+H+2];
      register short int b01 = 0;

      register short int a12 = A[idx-H+3];
      register short int a13 = A[idx  +3];
      register short int a14 = A[idx+H+3];
      register short int b02 = 0;

      register short int a15 = A[idx-H+4];
      register short int a16 = A[idx  +4];
      register short int a17 = A[idx+H+4];
      register short int b03 = 0;

      register short int a18 = A[idx-H+5];
      register short int a19 = A[idx  +5];
      register short int a20 = A[idx+H+5];
      register short int b04 = 0;

      register short int a21 = A[idx-H+6];
      register short int a22 = A[idx  +6];
      register short int a23 = A[idx+H+6];
      register short int b05 = 0;

      register short int a24 = A[idx-H+7];
      register short int a25 = A[idx  +7];
      register short int a26 = A[idx+H+7];
      register short int b06 = 0;

      register short int a27 = A[idx-H+8];
      register short int a28 = A[idx  +8];
      register short int a29 = A[idx+H+8];
      register short int b07 = 0;

      register short int a30 = A[idx-H+9];
      register short int a31 = A[idx  +9];
      register short int a32 = A[idx+H+9];
      register short int b08 = 0;

      register short int a33;
      register short int a34;
      register short int a35;
      register short int b09;

      register short int a36;
      register short int a37;
      register short int a38;
      register short int b10;

      if (idx + 9 < idx_end) {
        a33 = A[idx-H+10];
        a34 = A[idx  +10];
        a35 = A[idx+H+10];
        b09 = 0;

        a36 = A[idx-H+11];
        a37 = A[idx  +11];
        a38 = A[idx+H+11];
        b10 = 0;
      }

      b00 += kernel[0] * a00;
      b00 += kernel[1] * a01;
      b00 += kernel[2] * a02;
      b00 += kernel[3] * a03;
      b00 += kernel[4] * a04;
      b00 += kernel[5] * a05;
      b00 += kernel[6] * a06;
      b00 += kernel[7] * a07;
      b00 += kernel[8] * a08;
      B[idx+0] = b00;

      b01 += kernel[0] * a03;
      b01 += kernel[1] * a04;
      b01 += kernel[2] * a05;
      b01 += kernel[3] * a06;
      b01 += kernel[4] * a07;
      b01 += kernel[5] * a08;
      b01 += kernel[6] * a09;
      b01 += kernel[7] * a10;
      b01 += kernel[8] * a11;
      B[idx+1] = b01;

      b02 += kernel[0] * a06;
      b02 += kernel[1] * a07;
      b02 += kernel[2] * a08;
      b02 += kernel[3] * a09;
      b02 += kernel[4] * a10;
      b02 += kernel[5] * a11;
      b02 += kernel[6] * a12;
      b02 += kernel[7] * a13;
      b02 += kernel[8] * a14;
      B[idx+2] = b02;

      b03 += kernel[0] * a09;
      b03 += kernel[1] * a10;
      b03 += kernel[2] * a11;
      b03 += kernel[3] * a12;
      b03 += kernel[4] * a13;
      b03 += kernel[5] * a14;
      b03 += kernel[6] * a15;
      b03 += kernel[7] * a16;
      b03 += kernel[8] * a17;
      B[idx+3] = b03;

      b04 += kernel[0] * a12;
      b04 += kernel[1] * a13;
      b04 += kernel[2] * a14;
      b04 += kernel[3] * a15;
      b04 += kernel[4] * a16;
      b04 += kernel[5] * a17;
      b04 += kernel[6] * a18;
      b04 += kernel[7] * a19;
      b04 += kernel[8] * a20;
      B[idx+4] = b04;

      b05 += kernel[0] * a15;
      b05 += kernel[1] * a16;
      b05 += kernel[2] * a17;
      b05 += kernel[3] * a18;
      b05 += kernel[4] * a19;
      b05 += kernel[5] * a20;
      b05 += kernel[6] * a21;
      b05 += kernel[7] * a22;
      b05 += kernel[8] * a23;
      B[idx+5] = b05;

      b06 += kernel[0] * a18;
      b06 += kernel[1] * a19;
      b06 += kernel[2] * a20;
      b06 += kernel[3] * a21;
      b06 += kernel[4] * a22;
      b06 += kernel[5] * a23;
      b06 += kernel[6] * a24;
      b06 += kernel[7] * a25;
      b06 += kernel[8] * a26;
      B[idx+6] = b06;

      b07 += kernel[0] * a21;
      b07 += kernel[1] * a22;
      b07 += kernel[2] * a23;
      b07 += kernel[3] * a24;
      b07 += kernel[4] * a25;
      b07 += kernel[5] * a26;
      b07 += kernel[6] * a27;
      b07 += kernel[7] * a28;
      b07 += kernel[8] * a29;
      B[idx+7] = b07;

      b08 += kernel[0] * a24;
      b08 += kernel[1] * a25;
      b08 += kernel[2] * a26;
      b08 += kernel[3] * a27;
      b08 += kernel[4] * a28;
      b08 += kernel[5] * a29;
      b08 += kernel[6] * a30;
      b08 += kernel[7] * a31;
      b08 += kernel[8] * a32;
      B[idx+8] = b08;

      if (idx + 9 < idx_end) {

        b09 += kernel[0] * a27;
        b09 += kernel[1] * a28;
        b09 += kernel[2] * a29;
        b09 += kernel[3] * a30;
        b09 += kernel[4] * a31;
        b09 += kernel[5] * a32;
        b09 += kernel[6] * a33;
        b09 += kernel[7] * a34;
        b09 += kernel[8] * a35;
        B[idx+9] = b09;
        
        b10 += kernel[0] * a30;
        b10 += kernel[1] * a31;
        b10 += kernel[2] * a32;
        b10 += kernel[3] * a33;
        b10 += kernel[4] * a34;
        b10 += kernel[5] * a35;
        b10 += kernel[6] * a36;
        b10 += kernel[7] * a37;
        b10 += kernel[8] * a38;
        B[idx+10] = b10;
      }
    }
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
