#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <bsg_manycore_regression.h>

#define ALLOC_NAME "default_allocator"

#ifndef SIZE
// Please define SIZE in Makefile.
#endif

typedef struct {
    unsigned short type;                 // Magic identifier: 0x4d42
    unsigned int size;                   // File size in bytes
    unsigned short reserved1, reserved2;
    unsigned int offset;                 // Offset to image data, bytes
} HEADER;

typedef struct {
    unsigned int size;                   // Header size in bytes      
    unsigned int width;
    unsigned int height;                    // Width and height of image
    unsigned short planes;               // Number of colour planes   
    unsigned short bits;                 // Bits per pixel            
    unsigned int compression;            // Compression type
    unsigned int imagesize;              // Image size in bytes       
    unsigned int xresolution;
    unsigned int yresolution;         // Pixels per meter
    unsigned int ncolours;               // Number of colours         
    unsigned int importantcolours;       // Important colours         
} INFOHEADER;

typedef struct {
    unsigned char b,g,r;
} PIXEL;

int ReadBMP(const char* filename, short* pixelData, int width, int height);
void rgb2gray(short int* A_host, short int* gray, int image_size_x, int image_size_y);


int kernel_gblur(int argc, char **argv) {
    int rc;
    char *bin_path, *test_name;
    struct arguments_path args = {NULL, NULL};

    argp_parse(&argp_path, argc, argv, 0, 0, &args);
    bin_path = args.path;
    test_name = args.name;

    bsg_pr_test_info("Running kernel_gblur.\n");
    srand(time);

    // Initialize Device.
    hb_mc_device_t device;
    BSG_CUDA_CALL(hb_mc_device_init(&device, test_name, 0));

    hb_mc_pod_id_t pod;
    hb_mc_device_foreach_pod_id(&device, pod)
    {
        bsg_pr_info("Loading program for pod %d\n.", pod);
        BSG_CUDA_CALL(hb_mc_device_set_default_pod(&device, pod));
        BSG_CUDA_CALL(hb_mc_device_program_init(&device, bin_path, ALLOC_NAME, 0));

        // Allocate a block of memory in host.
        short int * A_host = (short int *) malloc(sizeof(short int) * SIZE * 3);
        short int * A_gray_host = (short int *) malloc(sizeof(short int) * SIZE);

        short int * B_host = (short*) malloc(sizeof(short int)*SIZE);
        short int * B_expected_host = (short*) malloc(sizeof(short int)*SIZE);
        // normalized gaussian function defined on {-1, 0, 1}^2
        float kernel[9] = {
            0.09474166, 0.11831801, 0.09474166,
            0.11831801, 0.14776132, 0.11831801,
            0.09474166, 0.11831801, 0.09474166
        }; 
        int H = 64;

        ReadBMP("/local1/yijueli/bsg_bladerunner/bsg_replicant/examples/hb_hammerbench/apps/gblur/yak.bmp", A_host, 64, 64);
        rgb2gray(A_host, A_gray_host, H, H);

        int idx;
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < H; j++) {
                idx = i + j * H;
                B_expected_host[idx] = 0;
                B_host[idx] = 0;
            }
        }
        for (int i = 1; i < H-1; i++) {
            for (int j = 1; j < H-1; j++) {
                idx = i + j * H;
                B_expected_host[idx] += kernel[0] * A_gray_host[idx-H-1];
                B_expected_host[idx] += kernel[1] * A_gray_host[idx-1  ];
                B_expected_host[idx] += kernel[2] * A_gray_host[idx+H-1];
                B_expected_host[idx] += kernel[3] * A_gray_host[idx-H  ];
                B_expected_host[idx] += kernel[4] * A_gray_host[idx    ];
                B_expected_host[idx] += kernel[5] * A_gray_host[idx+H  ];
                B_expected_host[idx] += kernel[6] * A_gray_host[idx-H+1];
                B_expected_host[idx] += kernel[7] * A_gray_host[idx+1  ];
                B_expected_host[idx] += kernel[8] * A_gray_host[idx+H+1];
            }
        }
        createBMP("/local1/yijueli/bsg_bladerunner/bsg_replicant/examples/hb_hammerbench/apps/gblur/expyak.bmp", 64,64, B_expected_host);

        // Make it pod-cache aligned
#define POD_CACHE_ALIGNED
#ifdef POD_CACHE_ALIGNED
    eva_t temp_device1, temp_device2;
    BSG_CUDA_CALL(hb_mc_device_malloc(&device, CACHE_LINE_WORDS*sizeof(int), &temp_device1));
    printf("temp Addr: %x\n", temp_device1);
    int align_size = (32)-1-((temp_device1>>2)%(CACHE_LINE_WORDS*32)/CACHE_LINE_WORDS);
    BSG_CUDA_CALL(hb_mc_device_malloc(&device, align_size*sizeof(int)*CACHE_LINE_WORDS, &temp_device2));
#endif

        // Allocate a block of memory in device.
        eva_t A_device, B_device;
        BSG_CUDA_CALL(hb_mc_device_malloc(&device, SIZE * sizeof(short int), &A_device));
        BSG_CUDA_CALL(hb_mc_device_malloc(&device, SIZE * sizeof(short int), &B_device));

        printf("A_device Addr: %x\n", A_device);
        printf("B_device Addr: %x\n", B_device);

        // DMA Transfer to device.  
        hb_mc_dma_htod_t htod_job [] = {
            {
            .d_addr = A_device,
            .h_addr = (void *) &A_gray_host[0],
            .size = SIZE * sizeof(short int)
            },
            {
            .d_addr = B_device,
            .h_addr = (void *) &B_host[0],
            .size = SIZE * sizeof(short int)
            }
        };

        BSG_CUDA_CALL(hb_mc_device_dma_to_device(&device, htod_job, 2));

        // CUDA arguments
        hb_mc_dimension_t tg_dim = { .x = bsg_tiles_X, .y = bsg_tiles_Y};
        hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};
        #define CUDA_ARGC 4
        uint32_t cuda_argv[CUDA_ARGC] = {A_device, B_device, H};

        // Enqueue Kernel.
        BSG_CUDA_CALL(hb_mc_kernel_enqueue (&device, grid_dim, tg_dim, "kernel_gblur", CUDA_ARGC, cuda_argv));
    
        // Launch kernel.
        hb_mc_manycore_trace_enable((&device)->mc);
        BSG_CUDA_CALL(hb_mc_device_tile_groups_execute(&device));
        hb_mc_manycore_trace_disable((&device)->mc);

        // Copy result and validate.
        hb_mc_dma_dtoh_t dtoh_job [] = {
            {
            .d_addr = B_device,
            .h_addr = (void *) &B_host[0],
            .size = SIZE * sizeof(short int)
            }
        };
        BSG_CUDA_CALL(hb_mc_device_dma_to_host(&device, &dtoh_job, 1));

        createBMP("/local1/yijueli/bsg_bladerunner/bsg_replicant/examples/hb_hammerbench/apps/gblur/output_yak.bmp", 64,64, B_host);

        // Validate
        for (int i = 0; i < SIZE; i++) {
            if (B_host[i] != B_expected_host[i]) {
                // printf("FAIL [%d]: expected=%f, actual=%f\n", i, B_expected_host[i], B_host[i]);
                printf("FAIL [%d, %d]: expected=%d, actual=%d, A_gray[i]=%d\n", i / H, i % H, B_expected_host[i], B_host[i], A_gray_host[i]);
                BSG_CUDA_CALL(hb_mc_device_finish(&device));
                return HB_MC_FAIL;
            }
        }

        // Freeze tiles.
        BSG_CUDA_CALL(hb_mc_device_program_finish(&device));
    }

    BSG_CUDA_CALL(hb_mc_device_finish(&device));
    return HB_MC_SUCCESS;  
}


void rgb2gray(short int* A_host, short int* gray, int image_size_x, int image_size_y) {
    for(int y = 0; y < image_size_y; y++) {
        for(int x = 0; x < image_size_x; x++) {
            int middle_index = x + y * image_size_x;
            gray[middle_index]=(short int) ((77 *A_host[middle_index*3] + 150 * A_host[middle_index*3+1]+ 29 * A_host[middle_index*3+2]) >> 8);
        }
    }
}

int ReadBMP(const char* filename, short* pixelData, int width, int height) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    unsigned char *bitmapImage;
    HEADER fileHeader;
    INFOHEADER infoHeader;

    fread(&fileHeader.type, sizeof(fileHeader.type), 1, file);
    fread(&fileHeader.size, sizeof(fileHeader.size), 1, file);
    fread(&fileHeader.reserved1, sizeof(fileHeader.reserved1), 1, file);
    fread(&fileHeader.reserved2, sizeof(fileHeader.reserved2), 1, file);
    fread(&fileHeader.offset, sizeof(fileHeader.offset), 1, file);
    fread(&infoHeader, sizeof(INFOHEADER), 1, file);

    if (fileHeader.type != 0x4D42) {
        printf("Not a valid BMP file\n");
        fclose(file);
        return -2;
    }

    int padding = (4 - ((width) * 3) % 4) % 4;

    fseek(file, fileHeader.offset, SEEK_SET);
    printf("BMP offset %d\n",fileHeader.offset);

    bitmapImage = (unsigned char*)malloc(width*height*3);

    //verify memory allocation
    if (!bitmapImage)
    {
        free(bitmapImage);
        fclose(file);
        return NULL;
    }

    fread(bitmapImage,width*height*3,1,file);

    //make sure pixelData image data was read
    if (bitmapImage == NULL)
    {
        fclose(file);
        printf("Reading BMP file error!!!!!\n");
        return NULL;
    }

    char* output;

    for (int imageIdx_y = 0 ;imageIdx_y < height ;imageIdx_y+=1)
    {
        for (int imageIdx_x = 0 ;imageIdx_x < width ;imageIdx_x+=1){
            int index=(imageIdx_x+imageIdx_y*width)*3;
            int index_row_rev=(imageIdx_x+(height-1-imageIdx_y)*width)*3;
    
            pixelData[index_row_rev] = (short) bitmapImage[index + 2];
            pixelData[index_row_rev+1] = (short) bitmapImage[index + 1];
            pixelData[index_row_rev+2] = (short) bitmapImage[index];
        }
    }

    free(bitmapImage);
    fclose(file);

    return 0; // Success
}

void createBMP(const char *filename, int width, int height, short int *pixels) {
    HEADER header;
    INFOHEADER infoheader;
    FILE *file;
    PIXEL pixel = {0,0,0};

    int imagesize = width * height * sizeof(PIXEL);

    // Set up the BMP header
    memcpy(&header.type, "BM", 2);
    //header.type = 0x4D42;
    header.size = sizeof(HEADER) + sizeof(INFOHEADER) + imagesize;
    printf("BMP SIZE= %d,%x\n",header.size,header.size);
    header.reserved1 = 0;
    header.reserved2 = 0;
    header.offset = sizeof(HEADER) + sizeof(INFOHEADER)-2;
    printf("OFFSET SIZE= %d,%x\n",header.offset,header.offset);

    // Set up the BMP info header
    infoheader.size = sizeof(INFOHEADER);
    printf("INFO HEADER SIZE= %d,%x\n",infoheader.size,infoheader.size);
    infoheader.width = width;
    infoheader.height = height;
    infoheader.planes = 1;
    infoheader.bits = 24;
    infoheader.compression = 0;
    infoheader.imagesize = imagesize;
    infoheader.xresolution = 0x0B12; // 2835 pixels per meter
    infoheader.yresolution = 0x0B12; // 2835 pixels per meter
    infoheader.ncolours = 0;
    infoheader.importantcolours = 0;

    // Create file
    file = fopen(filename, "wb");
    if (!file) {
        printf("Unable to open file!");
        return;
    }

    // Write headers
    fwrite(&header.type, 1, sizeof(header.type),  file);
    fwrite(&header.size, 1, sizeof(header.size),  file);
    fwrite(&header.reserved1, 1, sizeof(header.reserved1),  file);
    fwrite(&header.reserved2, 1, sizeof(header.reserved2),  file);
    fwrite(&header.offset, 1, sizeof(header.offset),  file);
    fwrite(&infoheader.size , 1,sizeof(infoheader.size ), file);
    fwrite(&infoheader.width , 1,sizeof(infoheader.width ), file);
    fwrite(&infoheader.height , 1,sizeof(infoheader.height ), file);
    fwrite(&infoheader.planes , 1,sizeof(infoheader.planes ), file);
    fwrite(&infoheader.bits , 1,sizeof(infoheader.bits), file);
    fwrite(&infoheader.compression , 1,sizeof(infoheader.compression), file);
    fwrite(&infoheader.imagesize , 1,sizeof(infoheader.imagesize ), file);
    fwrite(&infoheader.xresolution , 1,sizeof(infoheader.xresolution ), file);
    fwrite(&infoheader.yresolution , 1,sizeof(infoheader.yresolution ), file);
    fwrite(&infoheader.ncolours , 1,sizeof(infoheader.ncolours), file);
    fwrite(&infoheader.importantcolours , 1,sizeof(infoheader.importantcolours ), file);

    // Write bitmap
    for (int y = height - 1; y >= 0; y--) { // BMP files are stored bottom-to-top
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            pixel.b = pixel.g = pixel.r = (unsigned char)(pixels[i]);
            fwrite(&pixel, 1, sizeof(PIXEL),file);
        }
    }

    fclose(file);
}

declare_program_main("gblur", kernel_gblur)