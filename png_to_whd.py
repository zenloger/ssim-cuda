import struct
import numpy as np
from PIL import Image
import sys
import os

def png_to_whd(png_path, whd_path):
    # Загружаем PNG изображение
    try:
        img = Image.open(png_path)
    except Exception as e:
        print(f"Ошибка при загрузке PNG: {e}")
        return False
    
    # Конвертируем в RGB, если нужно
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    pixels = np.array(img)
    
    # Записываем в WHD формат
    try:
        with open(whd_path, 'wb') as f:
            # Записываем заголовок (width, height)
            f.write(struct.pack('<II', width, height))
            
            # Записываем пиксели [width][height][3]
            for y in range(height):
                for x in range(width):
                    r, g, b = pixels[y, x]
                    f.write(struct.pack('<BBB', r, g, b))
        
        print(f"Успешно конвертировано {png_path} -> {whd_path}")
        return True
    except Exception as e:
        print(f"Ошибка при записи WHD: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Использование: python png_to_whd.py <input.png> <output.whd>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not input_file.lower().endswith('.png'):
        print("Входной файл должен быть PNG")
        sys.exit(1)
    
    if not output_file.lower().endswith('.whd'):
        print("Выходной файл должен иметь расширение .whd")
        sys.exit(1)
    
    success = png_to_whd(input_file, output_file)
    sys.exit(0 if success else 1)