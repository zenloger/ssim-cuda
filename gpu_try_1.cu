%%writefile gpu.cu


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

// Константы для SSIM
#define C1 (0.01f * 0.01f)
#define C2 (0.03f * 0.03f)

#define MAX_PATCH_SIZE 100

// Структура для хранения результатов
struct MatchResult {
    int x;
    int y;
    float score;
};

__constant__ float const_patch[MAX_PATCH_SIZE * MAX_PATCH_SIZE];

// Ядро для вычисления SSIM между участком и окном
__global__ void computeSSIMKernel(const float* map, 
                                 float* correlation, int map_width, int map_height,
                                 int patch_width, int patch_height, 
                                 int step_x, int step_y) {
    // Разделяемая память для статистик
    __shared__ float map_val[MAX_PATCH_SIZE][MAX_PATCH_SIZE];
    __shared__ float map_cov[MAX_PATCH_SIZE][MAX_PATCH_SIZE];
    __shared__ float map_patch[MAX_PATCH_SIZE][MAX_PATCH_SIZE];

    // Координаты в корреляционной матрице
    int grid_x = blockIdx.x;
    int grid_y = blockIdx.y;

    // Координаты верхнего левого угла окна на карте
    int map_x = grid_x * step_x;
    int map_y = grid_y * step_y;

    // Координаты верхнего левого угла окна на участке
    int patch_x = threadIdx.x * blockDim.x;
    int patch_y = threadIdx.y * blockDim.y;

    // Потоки заполняют shared memory
    for (int y = patch_y; y < patch_y + blockDim.y && y < patch_height; y += blockDim.y) {
        for (int x = patch_x; x < patch_x + blockDim.x && x < patch_width; x += blockDim.x) {
            int global_x = patch_x + x;
            int global_y = patch_y + y;
            
            float val = 0.0f;
            if (global_x < map_width && global_y < map_height) {
                val = map[global_y * map_width + global_x];
            }

            map_val[y][x] = val;
            map_cov[y][x] = val * val;
            map_patch[y][x] = val * const_patch[y * patch_width + x];
        }
    }

    __syncthreads();

    
    const int window_size = 8; // Размер локального окна для SSIM
    float total_ssim = 0.0f;
    int window_count = 0;

    // Проход по всем локальным окнам в пределах участка
    for (int wy = patch_y; wy < patch_y + blockDim.y && wy <= patch_height - window_size; wy++) {
        for (int wx = 0; wx < patch_x + blockDim.x && wx <= patch_width - window_size; wx++) {
            float sum_map = 0.0f, sum_patch = 0.0f;
            float sum_map_sq = 0.0f, sum_patch_sq = 0.0f;
            float sum_map_patch = 0.0f;
            
            // Вычисление статистик ТОЛЬКО в пределах локального окна
            for (int y = 0; y < window_size; y++) {
                for (int x = 0; x < window_size; x++) {
                    float patch_val = const_patch[(wy + y) * patch_width + (wx + x)];
                    
                    sum_map += map_val[y][x];
                    sum_patch += patch_val;
                    sum_map_sq += map_cov[y][x];
                    sum_patch_sq += patch_val * patch_val;
                    sum_map_patch += map_patch[y][x];
                }
            }
            
            // Вычисление SSIM для этого окна
            float mean_map = sum_map / (window_size * window_size);
            float mean_patch = sum_patch / (window_size * window_size);
            float var_map = (sum_map_sq - mean_map * sum_map) / (window_size * window_size);
            float var_patch = (sum_patch_sq - mean_patch * sum_patch) / (window_size * window_size);
            float covar = (sum_map_patch - mean_map * sum_patch) / (window_size * window_size);
            
            float numerator = (2 * mean_map * mean_patch + C1) * (2 * covar + C2);
            float denominator = (mean_map*mean_map + mean_patch*mean_patch + C1) * 
                               (var_map + var_patch + C2);
            
            if (denominator != 0) {
                total_ssim += numerator / denominator;
                window_count++;
            }
        }
    }
    
    int grid_idx = grid_y * ((map_width - patch_width) / step_x + 1) + grid_x;
    correlation[grid_idx] = 0;

    __syncthreads();

    correlation[grid_idx] += window_count > 0 ? total_ssim / window_count : 0.0f;
}

// Функция для поиска участка на карте с помощью SSIM
MatchResult findPatchOnMap(const float* h_map, const float* h_patch, 
                          int map_width, int map_height,
                          int patch_width, int patch_height,
                          int step_x = 1, int step_y = 1) {
    size_t patch_size = patch_width * patch_height * sizeof(float);
    cudaMemcpyToSymbol(const_patch, h_patch, patch_size);

    // Выделение памяти на устройстве
    float *d_map, *d_patch, *d_correlation;
    size_t map_size = map_width * map_height * sizeof(float);
    
    cudaMalloc(&d_map, map_size);
    cudaMalloc(&d_patch, patch_size);
    
    // Копирование данных на устройство
    cudaMemcpy(d_map, h_map, map_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_patch, h_patch, patch_size, cudaMemcpyHostToDevice);
    
    // Размеры корреляционной матрицы
    int grid_width = (map_width - patch_width) / step_x + 1;
    int grid_height = (map_height - patch_height) / step_y + 1;
    size_t correlation_size = grid_width * grid_height * sizeof(float);
    
    cudaMalloc(&d_correlation, correlation_size);
    
    // Настройка размеров блоков и гридов
    dim3 blockSize(16, 16);
    dim3 gridSize(grid_width, grid_height);
    
    // Вычисление корреляционной матрицы
    computeSSIMKernel<<<gridSize, blockSize>>>(d_map, d_correlation,
                                             map_width, map_height,
                                             patch_width, patch_height,
                                             step_x, step_y);
    
    // Копирование результата обратно на хост
    float* h_correlation = (float*)malloc(correlation_size);
    cudaMemcpy(h_correlation, d_correlation, correlation_size, cudaMemcpyDeviceToHost);
    
    // Поиск максимального значения SSIM
    MatchResult best_match = {0, 0, 0.0f};
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            float score = h_correlation[y * grid_width + x];
            if (score > best_match.score) {
                best_match.x = x * step_x;
                best_match.y = y * step_y;
                best_match.score = score;
            }
        }
    }
    
    // Освобождение памяти
    free(h_correlation);
    cudaFree(d_map);
    cudaFree(d_patch);
    cudaFree(d_correlation);
    
    return best_match;
}

// Функция для конвертации RGB в grayscale
void rgbToGrayscale(const unsigned char* rgb, float* gray, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int rgb_idx = idx * channels;
            
            float r = rgb[rgb_idx];
            float g = channels > 1 ? rgb[rgb_idx + 1] : r;
            float b = channels > 2 ? rgb[rgb_idx + 2] : r;
            
            gray[idx] = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
        }
    }
}

// Структура для хранения изображения
struct WHDImage {
    uint32_t width;
    uint32_t height;
    uint8_t* data;  // данные в формате uint8_t (0-255) для каждого канала RGB
};

// Функция для чтения WHD файла
WHDImage readWHD(const char* filename) {
    WHDImage img = {0, 0, NULL};
    FILE* file = fopen(filename, "rb");
    
    if (!file) {
        printf("Error: Could not open WHD file %s\n", filename);
        return img;
    }

    // Чтение заголовка (width, height)
    if (fread(&img.width, sizeof(uint32_t), 1, file) != 1 ||
        fread(&img.height, sizeof(uint32_t), 1, file) != 1) {
        printf("Error: Invalid WHD header in %s\n", filename);
        fclose(file);
        return img;
    }

    // Выделение памяти для данных (в формате float для CUDA)
    size_t pixel_count = img.width * img.height * 3;
    img.data = (uint8_t*)malloc(pixel_count * sizeof(uint8_t));
    
    if (!img.data) {
        printf("Error: Memory allocation failed for WHD data\n");
        fclose(file);
        img.width = img.height = 0;
        return img;
    }

    // Чтение пиксельных данных
    uint8_t* pixel_buffer = (uint8_t*)malloc(pixel_count);
    if (!pixel_buffer) {
        printf("Error: Memory allocation failed for pixel buffer\n");
        free(img.data);
        fclose(file);
        img.width = img.height = 0;
        img.data = NULL;
        return img;
    }

    if (fread(pixel_buffer, 1, pixel_count, file) != pixel_count) {
        printf("Error: Invalid pixel data in WHD file %s\n", filename);
        free(pixel_buffer);
        free(img.data);
        fclose(file);
        img.width = img.height = 0;
        img.data = NULL;
        return img;
    }

    for (size_t i = 0; i < pixel_count; i++) {
        img.data[i] = (uint8_t)pixel_buffer[i];
    }
    
    printf("WHD Image: %dx%d\n", img.width, img.height);
    // printf("WHD First 10 pixels: ");
    // for (size_t i = 0; i < 10; i++) {
    //     printf("%d ", img.data[i]);
    // }
    // printf("\n");

    free(pixel_buffer);
    fclose(file);
    return img;
}

// Функция для освобождения памяти WHD изображения
void freeWHD(WHDImage* img) {
    if (img) {
        free(img->data);
        img->width = img->height = 0;
        img->data = NULL;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <map.whd> <patch.whd>\n", argv[0]);
        return 1;
    }

    // Загрузка изображений
    WHDImage map = readWHD(argv[1]);
    WHDImage patch = readWHD(argv[2]);
    
    if (!map.data || !patch.data) {
        printf("Error loading images\n");
        freeWHD(&map);
        freeWHD(&patch);
        return 1;
    }
    
    // Проверка размеров
    if (patch.width > map.width || patch.height > map.height) {
        printf("Error: Patch must be smaller than map\n");
        freeWHD(&map);
        freeWHD(&patch);
        return 1;
    }
    
    // Конвертация в grayscale
    float* map_gray = (float*)malloc(map.width * map.height * sizeof(float));
    float* patch_gray = (float*)malloc(patch.width * patch.height * sizeof(float));
    
    rgbToGrayscale(map.data, map_gray, map.width, map.height, 3);
    rgbToGrayscale(patch.data, patch_gray, patch.width, patch.height, 3);

    // Поиск участка на карте
    int step = 3; // Шаг скользящего окна (можно изменять)
    MatchResult result = findPatchOnMap(map_gray, patch_gray, 
                                      map.width, map.height,
                                      patch.width, patch.height,
                                      step, step);
    
    // Вывод результатов
    printf("Best match found at (%d, %d) with SSIM score: %f\n", 
           result.x, result.y, result.score);
    printf("Patch dimensions: %dx%d\n", patch.width, patch.height);
    
    // Освобождение памяти
    free(map_gray);
    free(patch_gray);

    // Освобождаем память оригинальных изображений
    freeWHD(&map);
    freeWHD(&patch);
    
    return 0;
}