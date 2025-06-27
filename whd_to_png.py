import struct
import numpy as np
from PIL import Image
import sys
import os

def whd_to_png(whd_path, png_path):
    try:
        with open(whd_path, 'rb') as f:
            # Читаем заголовок (width, height)
            width, height = struct.unpack('<II', f.read(8))
            
            # Читаем пиксели [width][height][3]
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            
            for y in range(height):
                for x in range(width):
                    r, g, b = struct.unpack('<BBB', f.read(3))
                    pixels[y, x] = [r, g, b]
            
            # Создаем изображение
            img = Image.fromarray(pixels, 'RGB')
            img.save(png_path)
            
            print(f"Успешно конвертировано {whd_path} -> {png_path}")
            return True
    except Exception as e:
        print(f"Ошибка при конвертации WHD в PNG: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python whd_to_png.py <input.whd> <output.png>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not input_file.lower().endswith('.whd'):
        print("Входной файл должен быть WHD")
        sys.exit(1)
    
    if not output_file.lower().endswith('.png'):
        print("Выходной файл должен иметь расширение .png")
        sys.exit(1)
    
    success = whd_to_png(input_file, output_file)
    sys.exit(0 if success else 1)