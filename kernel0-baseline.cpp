#define BSG_TILE_GROUP_X_DIM bsg_tiles_X
#define BSG_TILE_GROUP_Y_DIM bsg_tiles_Y
#include <bsg_manycore.h>
#include <bsg_cuda_lite_barrier.h>
#include "bsg_set_tile_x_y.h"
#include "bsg_group_strider.hpp"
#include <cstring>
#include <cstdint>
#include <math.h>

#ifdef WARM_CACHE
__attribute__((noinline))
static void warmup(float *A, float *B, float *C, int N)
{
  for (int i = __bsg_id*CACHE_LINE_WORDS; i < N; i += bsg_tiles_X*bsg_tiles_Y*CACHE_LINE_WORDS) {
      asm volatile ("lw x0, %[p]" :: [p] "m" (A[i]));
      asm volatile ("lw x0, %[p]" :: [p] "m" (B[i]));
      asm volatile ("sw x0, %[p]" :: [p] "m" (C[i]));
  }
  bsg_fence();
}
#endif


// Vector-Add: C = A + B
// N = vector size
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

  for (int k = __bsg_id*16; k < H*H; k += bsg_tiles_X*bsg_tiles_Y*16) {
    int idx;
    int i, j;
    idx = k;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) B[idx] += kernel[0] * A[idx-H-1];
    if (i > 0             ) B[idx] += kernel[1] * A[idx-1  ];
    if (i > 0   && j < H-1) B[idx] += kernel[2] * A[idx+H-1];
    if (j > 0             ) B[idx] += kernel[3] * A[idx-H  ];
                            B[idx] += kernel[4] * A[idx    ];
    if (j < H-1           ) B[idx] += kernel[5] * A[idx+H  ];
    if (i < H-1 && j > 0  ) B[idx] += kernel[6] * A[idx-H+1];
    if (i < H-1           ) B[idx] += kernel[7] * A[idx+1  ];
    if (i < H-1 && j < H-1) B[idx] += kernel[8] * A[idx+H+1];
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
