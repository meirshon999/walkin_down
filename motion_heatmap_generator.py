import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class MotionHeatmapGenerator:
    def __init__(self, images, num_vertical_divisions=10, num_horizontal_divisions=10,
                 sigma=2.0, color_intensity_factor=5, use_average_image_overlay=True, print_debug=False):
        """
        images: список путей к кадрам
        num_vertical_divisions: вертикальное деление сетки
        num_horizontal_divisions: горизонтальное деление сетки
        sigma: сглаживание тепловой карты
        color_intensity_factor: масштаб интенсивности цвета
        use_average_image_overlay: накладывать средний кадр как фон
        print_debug: печатать прогресс
        """
        self.images = images
        self.num_vertical_divisions = num_vertical_divisions
        self.num_horizontal_divisions = num_horizontal_divisions
        self.sigma = sigma
        self.color_intensity_factor = color_intensity_factor
        self.use_average_image_overlay = use_average_image_overlay
        self.print_debug = print_debug

    def generate_motion_heatmap(self, output_path="motion_heatmap.jpg"):
        # Загружаем все кадры как серые изображения
        frames = []
        for i, img_path in enumerate(self.images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frames.append(gray.astype(np.float32))
            if self.print_debug:
                print(f"Loaded frame {i+1}/{len(self.images)}")

        if len(frames) < 2:
            raise ValueError("Недостаточно кадров для анализа движения!")

        frames = np.stack(frames, axis=0)  # (num_frames, height, width)

        # Вычисляем разницу между кадрами
        diffs = np.diff(frames, axis=0)
        motion_intensity = np.std(diffs, axis=0)  # стандартное отклонение как показатель движения

        # Делим на сетку и усредняем
        h, w = motion_intensity.shape
        cell_h = h // self.num_vertical_divisions
        cell_w = w // self.num_horizontal_divisions
        heatmap = np.zeros_like(motion_intensity)
        for i in range(self.num_vertical_divisions):
            for j in range(self.num_horizontal_divisions):
                y1 = i*cell_h
                y2 = (i+1)*cell_h if i != self.num_vertical_divisions-1 else h
                x1 = j*cell_w
                x2 = (j+1)*cell_w if j != self.num_horizontal_divisions-1 else w
                cell_mean = np.mean(motion_intensity[y1:y2, x1:x2])
                heatmap[y1:y2, x1:x2] = cell_mean

        # Сглаживание
        heatmap = gaussian_filter(heatmap, sigma=self.sigma)

        # Нормализация
        heatmap = np.clip(heatmap * self.color_intensity_factor, 0, 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Накладываем средний кадр как фон
        if self.use_average_image_overlay:
            avg_frame = np.mean(frames, axis=0).astype(np.uint8)
            avg_frame_color = cv2.cvtColor(avg_frame, cv2.COLOR_GRAY2BGR)
            heatmap_color = cv2.addWeighted(avg_frame_color, 0.5, heatmap_color, 0.5, 0)

        cv2.imwrite(output_path, heatmap_color)
        if self.print_debug:
            print(f"Motion heatmap saved to {output_path}")
