import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter

class MotionHeatmapGenerator:
    def __init__(self, num_vertical_divisions=30, num_horizontal_divisions=30,
                 sigma=2.0, color_intensity_factor=10):
        self.num_vertical_divisions = num_vertical_divisions
        self.num_horizontal_divisions = num_horizontal_divisions
        self.sigma = sigma
        self.color_intensity_factor = color_intensity_factor
    
    def generate_motion_heatmap(self, frames):
        if len(frames) < 2:
            raise ValueError("Недостаточно кадров для анализа!")
        
        # Конвертируем кадры в grayscale
        gray_frames = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frames.append(gray.astype(np.float32))
        
        # Вычисляем разницу между последовательными кадрами
        diffs = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i], gray_frames[i-1])
            diffs.append(diff)
        
        # Усредняем разницы для получения общей карты движения
        motion_map = np.mean(diffs, axis=0)
        
        # Применяем сеточное сглаживание
        h, w = motion_map.shape
        cell_h = h // self.num_vertical_divisions
        cell_w = w // self.num_horizontal_divisions
        
        smoothed = motion_map.copy()
        for i in range(self.num_vertical_divisions):
            for j in range(self.num_horizontal_divisions):
                y1 = i * cell_h
                y2 = (i + 1) * cell_h if i != self.num_vertical_divisions - 1 else h
                x1 = j * cell_w
                x2 = (j + 1) * cell_w if j != self.num_horizontal_divisions - 1 else w
                
                cell_mean = np.mean(motion_map[y1:y2, x1:x2])
                smoothed[y1:y2, x1:x2] = cell_mean
        
        # Сглаживание Гауссом
        smoothed = gaussian_filter(smoothed, sigma=self.sigma)
        
        # Нормализация и применение цветовой карты
        smoothed = np.clip(smoothed * self.color_intensity_factor, 0, 255)
        smoothed_8bit = smoothed.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(smoothed_8bit, cv2.COLORMAP_JET)
        
        return heatmap_color
    
    def generate_sharpness_map(self, frames):
        """Создает карту резкости"""
        sharpness_map = np.zeros(frames[0].shape[:2], dtype=np.float32)
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_map += np.abs(laplacian)
        
        sharpness_map /= len(frames)
        sharpness_map = cv2.normalize(sharpness_map, None, 0, 255, cv2.NORM_MINMAX)
        sharpness_8bit = sharpness_map.astype(np.uint8)
        
        return cv2.applyColorMap(sharpness_8bit, cv2.COLORMAP_VIRIDIS)

def extract_frames(video_path, max_frames=100):
    """Извлекает кадры из видео"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Берем каждый N-й кадр для ускорения обработки
    step = max(1, frame_count // max_frames)
    
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def create_composite_image(original, heatmap, sharpness, output_path="composite.jpg"):
    """Создает составное изображение из всех анализов"""
    # Изменяем размер для отображения
    target_height = 300
    scale = target_height / original.shape[0]
    target_width = int(original.shape[1] * scale)
    
    orig_resized = cv2.resize(original, (target_width, target_height))
    heatmap_resized = cv2.resize(heatmap, (target_width, target_height))
    sharpness_resized = cv2.resize(sharpness, (target_width, target_height))
    
    # Создаем наложение тепловой карты на оригинал
    overlay = cv2.addWeighted(orig_resized, 0.6, heatmap_resized, 0.4, 0)
    
    # Собираем сетку 2x2
    top_row = np.hstack([orig_resized, heatmap_resized])
    bottom_row = np.hstack([sharpness_resized, overlay])
    composite = np.vstack([top_row, bottom_row])
    
    # Добавляем подписи
    font = cv2.FONT_HERSHEY_SIMPLEX
    positions = [
        (10, 25),  # Оригинал
        (target_width + 10, 25),  # Движение
        (10, target_height + 25),  # Резкость
        (target_width + 10, target_height + 25)  # Наложение
    ]
    
    labels = ["Оригинал", "Движение", "Резкость", "Наложение"]
    
    for pos, label in zip(positions, labels):
        cv2.putText(composite, label, pos, font, 0.7, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, composite)
    return composite

def create_html_report(output_dir, video_path, num_frames):
    """Создает HTML отчет для просмотра результатов"""
    import datetime
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Анализ видео</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .result {{ margin: 20px 0; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            .info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Результаты анализа видео</h1>
            <div class="info">
                <p><strong>Видео:</strong> {os.path.basename(video_path)}</p>
                <p><strong>Количество анализируемых кадров:</strong> {num_frames}</p>
                <p><strong>Дата анализа:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="result">
                <h2>1. Тепловая карта движения</h2>
                <p>Области с интенсивным движением отмечены красным/желтым цветом</p>
                <img src="motion_heatmap.jpg" alt="Тепловая карта движения">
            </div>
            
            <div class="result">
                <h2>2. Карта резкости</h2>
                <p>Области с высокой резкостью отмечены ярким цветом</p>
                <img src="sharpness_map.jpg" alt="Карта резкости">
            </div>
            
            <div class="result">
                <h2>3. Составной анализ</h2>
                <p>Сравнение всех результатов на одном изображении</p>
                <img src="composite_analysis.jpg" alt="Составной анализ">
            </div>
        </div>
    </body>
    </html>
    """
    
    html_path = os.path.join(output_dir, "report.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML отчет создан: {html_path}")
    print("Откройте этот файл в браузере для просмотра результатов")

def main():
    print("Тепловая карта движения и анализ видео")
    print("=" * 50)
    
    # Запрашиваем путь к видео
    video_path = input("Введите путь к видео файлу: ").strip()
    
    if not os.path.exists(video_path):
        print(f"Ошибка: Файл '{video_path}' не найден!")
        return
    
    try:
        # Создаем папку для результатов
        output_dir = "analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Извлекаем кадры
        print("Извлечение кадров...")
        frames = extract_frames(video_path)
        print(f"Извлечено {len(frames)} кадров")
        
        # Создаем анализатор
        generator = MotionHeatmapGenerator(
            num_vertical_divisions=30,
            num_horizontal_divisions=30,
            sigma=2.0,
            color_intensity_factor=10
        )
        
        # Генерируем тепловую карту движения
        print("Генерация тепловой карты движения...")
        heatmap = generator.generate_motion_heatmap(frames)
        
        # Генерируем карту резкости
        print("Генерация карты резкости...")
        sharpness_map = generator.generate_sharpness_map(frames)
        
        # Сохраняем отдельные результаты
        heatmap_path = os.path.join(output_dir, "motion_heatmap.jpg")
        sharpness_path = os.path.join(output_dir, "sharpness_map.jpg")
        
        cv2.imwrite(heatmap_path, heatmap)
        cv2.imwrite(sharpness_path, sharpness_map)
        
        # Создаем составное изображение
        composite_path = os.path.join(output_dir, "composite_analysis.jpg")
        original_frame = frames[len(frames) // 2]  # Берем средний кадр
        composite = create_composite_image(original_frame, heatmap, sharpness_map, composite_path)
        
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ:")
        print(f"1. Тепловая карта движения: {heatmap_path}")
        print(f"2. Карта резкости: {sharpness_path}")
        print(f"3. Составной анализ: {composite_path}")
        print("=" * 50)
        print("\nВсе результаты сохранены в папке 'analysis_results'")
        
        # Создаем HTML отчет для удобного просмотра
        create_html_report(output_dir, video_path, len(frames))
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()