import cv2
import os
import glob
from motion_heatmap_generator import MotionHeatmapGenerator


def extract_frames(video_path, frames_dir="frames"):
    """Извлекает кадры из видео в папку frames/."""
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Не удалось открыть видео. Проверь путь!")

    idx = 0
    print("Извлечение кадров...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(frames_dir, f"frame_{idx:04d}.jpg"), frame)
        idx += 1

    cap.release()
    print(f"Готово! Извлечено {idx} кадров.")
    return frames_dir


def generate_heatmap(frames_dir, output_path="motion_heatmap.jpg"):
    """Создаёт тепловую карту движения."""
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    if len(frame_paths) < 2:
        raise ValueError("Недостаточно кадров для анализа!")

    print("Создание тепловой карты движения...")
    generator = MotionHeatmapGenerator(
        num_vertical_divisions=10,
        num_horizontal_divisions=10,
        images=frame_paths,
        use_average_image_overlay=True,
        sigma=2.0,
        color_intensity_factor=5,
        print_debug=True,
        random_seed=42
    )

    generator.generate_motion_heatmap(output_path)
    print(f"Тепловая карта сохранена как: {output_path}")


def main():
    video_path = input("Введите имя видеофайла (например: input.mp4): ").strip()

    frames_directory = extract_frames(video_path)
    generate_heatmap(frames_directory)


if __name__ == "__main__":
    main()
