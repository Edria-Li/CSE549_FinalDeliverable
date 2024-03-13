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
    register short int a00 = A[idx-H-1+0];
    register short int a01 = A[idx-1  +0];
    register short int a02 = A[idx+H-1+0];
    register short int a03 = A[idx-H  +0];
    register short int a04 = A[idx    +0];
    register short int a05 = A[idx+H  +0];
    register short int a06 = A[idx-H+1+0];
    register short int a07 = A[idx+1  +0];
    register short int a08 = A[idx+H+1+0];

    register short int a10 = A[idx-H-1+1];
    register short int a11 = A[idx-1  +1];
    register short int a12 = A[idx+H-1+1];
    register short int a13 = A[idx-H  +1];
    register short int a14 = A[idx    +1];
    register short int a15 = A[idx+H  +1];
    register short int a16 = A[idx-H+1+1];
    register short int a17 = A[idx+1  +1];
    register short int a18 = A[idx+H+1+1];

    register short int a20 = A[idx-H-1+2];
    register short int a21 = A[idx-1  +2];
    register short int a22 = A[idx+H-1+2];
    register short int a23 = A[idx-H  +2];
    register short int a24 = A[idx    +2];
    register short int a25 = A[idx+H  +2];
    register short int a26 = A[idx-H+1+2];
    register short int a27 = A[idx+1  +2];
    register short int a28 = A[idx+H+1+2];

    register short int a30 = A[idx-H-1+3];
    register short int a31 = A[idx-1  +3];
    register short int a32 = A[idx+H-1+3];
    register short int a33 = A[idx-H  +3];
    register short int a34 = A[idx    +3];
    register short int a35 = A[idx+H  +3];
    register short int a36 = A[idx-H+1+3];
    register short int a37 = A[idx+1  +3];
    register short int a38 = A[idx+H+1+3];

    register short int a40 = A[idx-H-1+4];
    register short int a41 = A[idx-1  +4];
    register short int a42 = A[idx+H-1+4];
    register short int a43 = A[idx-H  +4];
    register short int a44 = A[idx    +4];
    register short int a45 = A[idx+H  +4];
    register short int a46 = A[idx-H+1+4];
    register short int a47 = A[idx+1  +4];
    register short int a48 = A[idx+H+1+4];

    register short int a50 = A[idx-H-1+5];
    register short int a51 = A[idx-1  +5];
    register short int a52 = A[idx+H-1+5];
    register short int a53 = A[idx-H  +5];
    register short int a54 = A[idx    +5];
    register short int a55 = A[idx+H  +5];
    register short int a56 = A[idx-H+1+5];
    register short int a57 = A[idx+1  +5];
    register short int a58 = A[idx+H+1+5];

    register short int a60 = A[idx-H-1+6];
    register short int a61 = A[idx-1  +6];
    register short int a62 = A[idx+H-1+6];
    register short int a63 = A[idx-H  +6];
    register short int a64 = A[idx    +6];
    register short int a65 = A[idx+H  +6];
    register short int a66 = A[idx-H+1+6];
    register short int a67 = A[idx+1  +6];
    register short int a68 = A[idx+H+1+6];

    register short int a70 = A[idx-H-1+7];
    register short int a71 = A[idx-1  +7];
    register short int a72 = A[idx+H-1+7];
    register short int a73 = A[idx-H  +7];
    register short int a74 = A[idx    +7];
    register short int a75 = A[idx+H  +7];
    register short int a76 = A[idx-H+1+7];
    register short int a77 = A[idx+1  +7];
    register short int a78 = A[idx+H+1+7];

    register short int a80 = A[idx-H-1+8];
    register short int a81 = A[idx-1  +8];
    register short int a82 = A[idx+H-1+8];
    register short int a83 = A[idx-H  +8];
    register short int a84 = A[idx    +8];
    register short int a85 = A[idx+H  +8];
    register short int a86 = A[idx-H+1+8];
    register short int a87 = A[idx+1  +8];
    register short int a88 = A[idx+H+1+8];

    register short int a90 = A[idx-H-1+9];
    register short int a91 = A[idx-1  +9];
    register short int a92 = A[idx+H-1+9];
    register short int a93 = A[idx-H  +9];
    register short int a94 = A[idx    +9];
    register short int a95 = A[idx+H  +9];
    register short int a96 = A[idx-H+1+9];
    register short int a97 = A[idx+1  +9];
    register short int a98 = A[idx+H+1+9];

    register short int a100 = A[idx-H-1+10];
    register short int a101 = A[idx-1  +10];
    register short int a102 = A[idx+H-1+10];
    register short int a103 = A[idx-H  +10];
    register short int a104 = A[idx    +10];
    register short int a105 = A[idx+H  +10];
    register short int a106 = A[idx-H+1+10];
    register short int a107 = A[idx+1  +10];
    register short int a108 = A[idx+H+1+10];

    register short int a110 = A[idx-H-1+11];
    register short int a111 = A[idx-1  +11];
    register short int a112 = A[idx+H-1+11];
    register short int a113 = A[idx-H  +11];
    register short int a114 = A[idx    +11];
    register short int a115 = A[idx+H  +11];
    register short int a116 = A[idx-H+1+11];
    register short int a117 = A[idx+1  +11];
    register short int a118 = A[idx+H+1+11];

    register short int a120 = A[idx-H-1+12];
    register short int a121 = A[idx-1  +12];
    register short int a122 = A[idx+H-1+12];
    register short int a123 = A[idx-H  +12];
    register short int a124 = A[idx    +12];
    register short int a125 = A[idx+H  +12];
    register short int a126 = A[idx-H+1+12];
    register short int a127 = A[idx+1  +12];
    register short int a128 = A[idx+H+1+12];

    register short int a130 = A[idx-H-1+13];
    register short int a131 = A[idx-1  +13];
    register short int a132 = A[idx+H-1+13];
    register short int a133 = A[idx-H  +13];
    register short int a134 = A[idx    +13];
    register short int a135 = A[idx+H  +13];
    register short int a136 = A[idx-H+1+13];
    register short int a137 = A[idx+1  +13];
    register short int a138 = A[idx+H+1+13];

    register short int a140 = A[idx-H-1+14];
    register short int a141 = A[idx-1  +14];
    register short int a142 = A[idx+H-1+14];
    register short int a143 = A[idx-H  +14];
    register short int a144 = A[idx    +14];
    register short int a145 = A[idx+H  +14];
    register short int a146 = A[idx-H+1+14];
    register short int a147 = A[idx+1  +14];
    register short int a148 = A[idx+H+1+14];

    register short int a150 = A[idx-H-1+15];
    register short int a151 = A[idx-1  +15];
    register short int a152 = A[idx+H-1+15];
    register short int a153 = A[idx-H  +15];
    register short int a154 = A[idx    +15];
    register short int a155 = A[idx+H  +15];
    register short int a156 = A[idx-H+1+15];
    register short int a157 = A[idx+1  +15];
    register short int a158 = A[idx+H+1+15];

    register short int b00 = 0;
    register short int b01 = 0;
    register short int b02 = 0;
    register short int b03 = 0;
    register short int b04 = 0;
    register short int b05 = 0;
    register short int b06 = 0;
    register short int b07 = 0;
    register short int b08 = 0;

    register short int b10 = 0;
    register short int b11 = 0;
    register short int b12 = 0;
    register short int b13 = 0;
    register short int b14 = 0;
    register short int b15 = 0;
    register short int b16 = 0;
    register short int b17 = 0;
    register short int b18 = 0;

    register short int b20 = 0;
    register short int b21 = 0;
    register short int b22 = 0;
    register short int b23 = 0;
    register short int b24 = 0;
    register short int b25 = 0;
    register short int b26 = 0;
    register short int b27 = 0;
    register short int b28 = 0;

    register short int b30 = 0;
    register short int b31 = 0;
    register short int b32 = 0;
    register short int b33 = 0;
    register short int b34 = 0;
    register short int b35 = 0;
    register short int b36 = 0;
    register short int b37 = 0;
    register short int b38 = 0;

    register short int b40 = 0;
    register short int b41 = 0;
    register short int b42 = 0;
    register short int b43 = 0;
    register short int b44 = 0;
    register short int b45 = 0;
    register short int b46 = 0;
    register short int b47 = 0;
    register short int b48 = 0;

    register short int b50 = 0;
    register short int b51 = 0;
    register short int b52 = 0;
    register short int b53 = 0;
    register short int b54 = 0;
    register short int b55 = 0;
    register short int b56 = 0;
    register short int b57 = 0;
    register short int b58 = 0;

    register short int b60 = 0;
    register short int b61 = 0;
    register short int b62 = 0;
    register short int b63 = 0;
    register short int b64 = 0;
    register short int b65 = 0;
    register short int b66 = 0;
    register short int b67 = 0;
    register short int b68 = 0;

    register short int b70 = 0;
    register short int b71 = 0;
    register short int b72 = 0;
    register short int b73 = 0;
    register short int b74 = 0;
    register short int b75 = 0;
    register short int b76 = 0;
    register short int b77 = 0;
    register short int b78 = 0;

    register short int b80 = 0;
    register short int b81 = 0;
    register short int b82 = 0;
    register short int b83 = 0;
    register short int b84 = 0;
    register short int b85 = 0;
    register short int b86 = 0;
    register short int b87 = 0;
    register short int b88 = 0;

    register short int b90 = 0;
    register short int b91 = 0;
    register short int b92 = 0;
    register short int b93 = 0;
    register short int b94 = 0;
    register short int b95 = 0;
    register short int b96 = 0;
    register short int b97 = 0;
    register short int b98 = 0;

    register short int b100 = 0;
    register short int b101 = 0;
    register short int b102 = 0;
    register short int b103 = 0;
    register short int b104 = 0;
    register short int b105 = 0;
    register short int b106 = 0;
    register short int b107 = 0;
    register short int b108 = 0;

    register short int b110 = 0;
    register short int b111 = 0;
    register short int b112 = 0;
    register short int b113 = 0;
    register short int b114 = 0;
    register short int b115 = 0;
    register short int b116 = 0;
    register short int b117 = 0;
    register short int b118 = 0;

    register short int b120 = 0;
    register short int b121 = 0;
    register short int b122 = 0;
    register short int b123 = 0;
    register short int b124 = 0;
    register short int b125 = 0;
    register short int b126 = 0;
    register short int b127 = 0;
    register short int b128 = 0;

    register short int b130 = 0;
    register short int b131 = 0;
    register short int b132 = 0;
    register short int b133 = 0;
    register short int b134 = 0;
    register short int b135 = 0;
    register short int b136 = 0;
    register short int b137 = 0;
    register short int b138 = 0;

    register short int b140 = 0;
    register short int b141 = 0;
    register short int b142 = 0;
    register short int b143 = 0;
    register short int b144 = 0;
    register short int b145 = 0;
    register short int b146 = 0;
    register short int b147 = 0;
    register short int b148 = 0;

    register short int b150 = 0;
    register short int b151 = 0;
    register short int b152 = 0;
    register short int b153 = 0;
    register short int b154 = 0;
    register short int b155 = 0;
    register short int b156 = 0;
    register short int b157 = 0;
    register short int b158 = 0;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b00 = kernel[0] * a00;
    if (i > 0             ) b01 = kernel[1] * a01;
    if (i > 0   && j < H-1) b02 = kernel[2] * a02;
    if (j > 0             ) b03 = kernel[3] * a03;
                            b04 = kernel[4] * a04;
    if (j < H-1           ) b05 = kernel[5] * a05;
    if (i < H-1 && j > 0  ) b06 = kernel[6] * a06;
    if (i < H-1           ) b07 = kernel[7] * a07;
    if (i < H-1 && j < H-1) b08 = kernel[8] * a08;
    B[idx] = b00 + b01 + b02 + b03 + b04 + b05 + b06 + b07 + b08;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b10 = kernel[0] * a10;
    if (i > 0             ) b11 = kernel[1] * a11;
    if (i > 0   && j < H-1) b12 = kernel[2] * a12;
    if (j > 0             ) b13 = kernel[3] * a13;
                            b14 = kernel[4] * a14;
    if (j < H-1           ) b15 = kernel[5] * a15;
    if (i < H-1 && j > 0  ) b16 = kernel[6] * a16;
    if (i < H-1           ) b17 = kernel[7] * a17;
    if (i < H-1 && j < H-1) b18 = kernel[8] * a18;
    B[idx] = b10 + b11 + b12 + b13 + b14 + b15 + b16 + b17 + b18;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b20 = kernel[0] * a20;
    if (i > 0             ) b21 = kernel[1] * a21;
    if (i > 0   && j < H-1) b22 = kernel[2] * a22;
    if (j > 0             ) b23 = kernel[3] * a23;
                            b24 = kernel[4] * a24;
    if (j < H-1           ) b25 = kernel[5] * a25;
    if (i < H-1 && j > 0  ) b26 = kernel[6] * a26;
    if (i < H-1           ) b27 = kernel[7] * a27;
    if (i < H-1 && j < H-1) b28 = kernel[8] * a28;
    B[idx] = b20 + b21 + b22 + b23 + b24 + b25 + b26 + b27 + b28;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b30 = kernel[0] * a30;
    if (i > 0             ) b31 = kernel[1] * a31;
    if (i > 0   && j < H-1) b32 = kernel[2] * a32;
    if (j > 0             ) b33 = kernel[3] * a33;
                            b34 = kernel[4] * a34;
    if (j < H-1           ) b35 = kernel[5] * a35;
    if (i < H-1 && j > 0  ) b36 = kernel[6] * a36;
    if (i < H-1           ) b37 = kernel[7] * a37;
    if (i < H-1 && j < H-1) b38 = kernel[8] * a38;
    B[idx] = b30 + b31 + b32 + b33 + b34 + b35 + b36 + b37 + b38;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b40 = kernel[0] * a40;
    if (i > 0             ) b41 = kernel[1] * a41;
    if (i > 0   && j < H-1) b42 = kernel[2] * a42;
    if (j > 0             ) b43 = kernel[3] * a43;
                            b44 = kernel[4] * a44;
    if (j < H-1           ) b45 = kernel[5] * a45;
    if (i < H-1 && j > 0  ) b46 = kernel[6] * a46;
    if (i < H-1           ) b47 = kernel[7] * a47;
    if (i < H-1 && j < H-1) b48 = kernel[8] * a48;
    B[idx] = b40 + b41 + b42 + b43 + b44 + b45 + b46 + b47 + b48;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b50 = kernel[0] * a50;
    if (i > 0             ) b51 = kernel[1] * a51;
    if (i > 0   && j < H-1) b52 = kernel[2] * a52;
    if (j > 0             ) b53 = kernel[3] * a53;
                            b54 = kernel[4] * a54;
    if (j < H-1           ) b55 = kernel[5] * a55;
    if (i < H-1 && j > 0  ) b56 = kernel[6] * a56;
    if (i < H-1           ) b57 = kernel[7] * a57;
    if (i < H-1 && j < H-1) b58 = kernel[8] * a58;
    B[idx] = b50 + b51 + b52 + b53 + b54 + b55 + b56 + b57 + b58;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b60 = kernel[0] * a60;
    if (i > 0             ) b61 = kernel[1] * a61;
    if (i > 0   && j < H-1) b62 = kernel[2] * a62;
    if (j > 0             ) b63 = kernel[3] * a63;
                            b64 = kernel[4] * a64;
    if (j < H-1           ) b65 = kernel[5] * a65;
    if (i < H-1 && j > 0  ) b66 = kernel[6] * a66;
    if (i < H-1           ) b67 = kernel[7] * a67;
    if (i < H-1 && j < H-1) b68 = kernel[8] * a68;
    B[idx] = b60 + b61 + b62 + b63 + b64 + b65 + b66 + b67 + b68;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b70 = kernel[0] * a70;
    if (i > 0             ) b71 = kernel[1] * a71;
    if (i > 0   && j < H-1) b72 = kernel[2] * a72;
    if (j > 0             ) b73 = kernel[3] * a73;
                            b74 = kernel[4] * a74;
    if (j < H-1           ) b75 = kernel[5] * a75;
    if (i < H-1 && j > 0  ) b76 = kernel[6] * a76;
    if (i < H-1           ) b77 = kernel[7] * a77;
    if (i < H-1 && j < H-1) b78 = kernel[8] * a78;
    B[idx] = b70 + b71 + b72 + b73 + b74 + b75 + b76 + b77 + b78;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b80 = kernel[0] * a80;
    if (i > 0             ) b81 = kernel[1] * a81;
    if (i > 0   && j < H-1) b82 = kernel[2] * a82;
    if (j > 0             ) b83 = kernel[3] * a83;
                            b84 = kernel[4] * a84;
    if (j < H-1           ) b85 = kernel[5] * a85;
    if (i < H-1 && j > 0  ) b86 = kernel[6] * a86;
    if (i < H-1           ) b87 = kernel[7] * a87;
    if (i < H-1 && j < H-1) b88 = kernel[8] * a88;
    B[idx] = b80 + b81 + b82 + b83 + b84 + b85 + b86 + b87 + b88;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b90 = kernel[0] * a90;
    if (i > 0             ) b91 = kernel[1] * a91;
    if (i > 0   && j < H-1) b92 = kernel[2] * a92;
    if (j > 0             ) b93 = kernel[3] * a93;
                            b94 = kernel[4] * a94;
    if (j < H-1           ) b95 = kernel[5] * a95;
    if (i < H-1 && j > 0  ) b96 = kernel[6] * a96;
    if (i < H-1           ) b97 = kernel[7] * a97;
    if (i < H-1 && j < H-1) b98 = kernel[8] * a98;
    B[idx] = b90 + b91 + b92 + b93 + b94 + b95 + b96 + b97 + b98;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b100 = kernel[0] * a100;
    if (i > 0             ) b101 = kernel[1] * a101;
    if (i > 0   && j < H-1) b102 = kernel[2] * a102;
    if (j > 0             ) b103 = kernel[3] * a103;
                            b104 = kernel[4] * a104;
    if (j < H-1           ) b105 = kernel[5] * a105;
    if (i < H-1 && j > 0  ) b106 = kernel[6] * a106;
    if (i < H-1           ) b107 = kernel[7] * a107;
    if (i < H-1 && j < H-1) b108 = kernel[8] * a108;
    B[idx] = b100 + b101 + b102 + b103 + b104 + b105 + b106 + b107 + b108;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b110 = kernel[0] * a110;
    if (i > 0             ) b111 = kernel[1] * a111;
    if (i > 0   && j < H-1) b112 = kernel[2] * a112;
    if (j > 0             ) b113 = kernel[3] * a113;
                            b114 = kernel[4] * a114;
    if (j < H-1           ) b115 = kernel[5] * a115;
    if (i < H-1 && j > 0  ) b116 = kernel[6] * a116;
    if (i < H-1           ) b117 = kernel[7] * a117;
    if (i < H-1 && j < H-1) b118 = kernel[8] * a118;
    B[idx] = b110 + b111 + b112 + b113 + b114 + b115 + b116 + b117 + b118;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b120 = kernel[0] * a120;
    if (i > 0             ) b121 = kernel[1] * a121;
    if (i > 0   && j < H-1) b122 = kernel[2] * a122;
    if (j > 0             ) b123 = kernel[3] * a123;
                            b124 = kernel[4] * a124;
    if (j < H-1           ) b125 = kernel[5] * a125;
    if (i < H-1 && j > 0  ) b126 = kernel[6] * a126;
    if (i < H-1           ) b127 = kernel[7] * a127;
    if (i < H-1 && j < H-1) b128 = kernel[8] * a128;
    B[idx] = b120 + b121 + b122 + b123 + b124 + b125 + b126 + b127 + b128;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b130 = kernel[0] * a130;
    if (i > 0             ) b131 = kernel[1] * a131;
    if (i > 0   && j < H-1) b132 = kernel[2] * a132;
    if (j > 0             ) b133 = kernel[3] * a133;
                            b134 = kernel[4] * a134;
    if (j < H-1           ) b135 = kernel[5] * a135;
    if (i < H-1 && j > 0  ) b136 = kernel[6] * a136;
    if (i < H-1           ) b137 = kernel[7] * a137;
    if (i < H-1 && j < H-1) b138 = kernel[8] * a138;
    B[idx] = b130 + b131 + b132 + b133 + b134 + b135 + b136 + b137 + b138;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b140 = kernel[0] * a140;
    if (i > 0             ) b141 = kernel[1] * a141;
    if (i > 0   && j < H-1) b142 = kernel[2] * a142;
    if (j > 0             ) b143 = kernel[3] * a143;
                            b144 = kernel[4] * a144;
    if (j < H-1           ) b145 = kernel[5] * a145;
    if (i < H-1 && j > 0  ) b146 = kernel[6] * a146;
    if (i < H-1           ) b147 = kernel[7] * a147;
    if (i < H-1 && j < H-1) b148 = kernel[8] * a148;
    B[idx] = b140 + b141 + b142 + b143 + b144 + b145 + b146 + b147 + b148;
    idx++;

    i = idx % H;
    j = idx / H;
    if (i > 0   && j > 0  ) b150 = kernel[0] * a150;
    if (i > 0             ) b151 = kernel[1] * a151;
    if (i > 0   && j < H-1) b152 = kernel[2] * a152;
    if (j > 0             ) b153 = kernel[3] * a153;
                            b154 = kernel[4] * a154;
    if (j < H-1           ) b155 = kernel[5] * a155;
    if (i < H-1 && j > 0  ) b156 = kernel[6] * a156;
    if (i < H-1           ) b157 = kernel[7] * a157;
    if (i < H-1 && j < H-1) b158 = kernel[8] * a158;
    B[idx] = b150 + b151 + b152 + b153 + b154 + b155 + b156 + b157 + b158;
  }

  bsg_fence();
  bsg_cuda_print_stat_kernel_end();
  bsg_fence();
  bsg_barrier_hw_tile_group_sync();

  return 0;
}
