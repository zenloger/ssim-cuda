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

// Структура для хранения результатов
struct MatchResult {
    int x;
    int y;
    float score;
};

MatchResult findPatchOnMapCPU(const float* map, const float* patch, 
                            int map_width, int map_height,
                            int patch_width, int patch_height,
                            int step_x = 1, int step_y = 1) {
    // Размер локального окна для SSIM (обычно 8x8 или 11x11)
    const int window_size = 8;
    
    // Размеры корреляционной матрицы
    int grid_width = (map_width - patch_width) / step_x + 1;
    int grid_height = (map_height - patch_height) / step_y + 1;
    
    // Выделение памяти для корреляционной матрицы
    float* correlation = (float*)malloc(grid_width * grid_height * sizeof(float));
    
    // Перебор всех возможных положений патча на карте
    for (int gy = 0; gy < grid_height; ++gy) {
        for (int gx = 0; gx < grid_width; ++gx) {
            // Координаты верхнего левого угла окна на карте
            int map_x = gx * step_x;
            int map_y = gy * step_y;
            
            // Проверка выхода за границы карты
            if (map_x + patch_width > map_width || map_y + patch_height > map_height) {
                correlation[gy * grid_width + gx] = 0.0f;
                continue;
            }
            
            float total_ssim = 0.0f;
            int window_count = 0;
            
            // Перебор всех локальных окон внутри патча
            for (int wy = 0; wy <= patch_height - window_size; wy += 1) {
                for (int wx = 0; wx <= patch_width - window_size; wx += 1) {
                    float sum_map = 0.0f, sum_patch = 0.0f;
                    float sum_map_sq = 0.0f, sum_patch_sq = 0.0f;
                    float sum_map_patch = 0.0f;
                    
                    // Вычисление статистик в пределах локального окна
                    for (int y = 0; y < window_size; ++y) {
                        for (int x = 0; x < window_size; ++x) {
                            float map_val = map[(map_y + wy + y) * map_width + (map_x + wx + x)];
                            float patch_val = patch[(wy + y) * patch_width + (wx + x)];
                            
                            sum_map += map_val;
                            sum_patch += patch_val;
                            sum_map_sq += map_val * map_val;
                            sum_patch_sq += patch_val * patch_val;
                            sum_map_patch += map_val * patch_val;
                        }
                    }
                    
                    // Вычисление SSIM для текущего локального окна
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
            
            // Усреднение SSIM по всем локальным окнам
            correlation[gy * grid_width + gx] = window_count > 0 ? total_ssim / window_count : 0.0f;
        }
    }
    
    // Поиск максимального значения SSIM
    MatchResult best_match = {0, 0, 0.0f};
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            float score = correlation[y * grid_width + x];
            if (score > best_match.score) {
                best_match.x = x * step_x;
                best_match.y = y * step_y;
                best_match.score = score;
            }
        }
    }
    
    free(correlation);
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
    MatchResult result = findPatchOnMapCPU(map_gray, patch_gray, 
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