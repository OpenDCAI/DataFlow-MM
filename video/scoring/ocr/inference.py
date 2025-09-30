"""
OCR analysis script for video frames using PaddleOCR.
Calculates text area ratios for video clips using distributed processing.
"""

import os
import glob
import argparse
import pandas as pd
from multiprocessing import Manager
import queue
import concurrent.futures
from tqdm import tqdm
import cv2
from glob import glob
from paddleocr import PaddleOCR


def process_single_row(row, args, model):
    """Process a single row to calculate OCR text area ratio."""
    img_dir = os.path.join(args.fig_load_dir, row["id"])
    img_list = sorted(glob(f"{img_dir}/img/*.jpg"))[:3]
    # Load images
    images = [cv2.imread(img_path) for img_path in img_list]

    result = model.predict(input=images)
    area = row["height"] * row["width"]

    area_list = []
    for res in result:
        total_text_area = 0  # Initialize total text area
        for rec_box in res["rec_boxes"]:
            x_min, y_min, x_max, y_max = (
                float(rec_box[0]),
                float(rec_box[1]),
                float(rec_box[2]),
                float(rec_box[3]),
            )  # Extract top-left and bottom-right coordinates
            text_area = (x_max - x_min) * (y_max - y_min)  # Calculate text area
            total_text_area += text_area
        ratio = total_text_area / area
        area_list.append(ratio)
    return (
        max(area_list) if area_list else 0.0
    )  # Return max area ratio, 0.0 if no text detected


def worker(task_queue, result_queue, args, id):
    """Worker function for multiprocessing OCR inference."""
    gpu_id = id % args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Bind to specific GPU
    device = "gpu:0"  # if torch.cuda.is_available() else "cpu"

    # Initialize PaddleOCR model with disabled orientation and unwarping features
    model = PaddleOCR(
        device=device,
        use_doc_orientation_classify=False,  # Disable document orientation classification
        use_doc_unwarping=False,  # Disable text image correction
        use_textline_orientation=False,  # Disable text line orientation classification
    )

    while True:
        try:
            index, row = task_queue.get_nowait()
        except queue.Empty:
            break

        area_list = process_single_row(row, args, model)
        result_queue.put((index, area_list))


def parse_args():
    """Parse command line arguments for OCR inference."""
    parser = argparse.ArgumentParser(description="SAM2 Image Predictor")
    parser.add_argument("--csv_path", type=str, help="Path to the csv file")
    parser.add_argument(
        "--fig_load_dir", type=str, default="img", help="Directory containing images"
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="#workers for concurrent.futures"
    )
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    parser.add_argument("--skip_if_existing", action="store_true")
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv_path):
        print(f"csv file '{args.csv_path}' not found. Exit.")
        return

    wo_ext, ext = os.path.splitext(args.csv_path)
    out_path = f"{wo_ext}_ocr{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output csv file '{out_path}' already exists. Exit.")
        exit()

    df = pd.read_csv(args.csv_path)
    results = []

    if args.disable_parallel:
        # Sequential processing
        model = PaddleOCR(
            device="gpu:0",  # if torch.cuda.is_available() else "cpu"
            use_doc_orientation_classify=False,  # Disable document orientation classification
            use_doc_unwarping=False,  # Disable text image correction
            use_textline_orientation=False,  # Disable text line orientation classification
        )
        ocr_scores = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            score = process_single_row(row, args, model)
            ocr_scores.append(score)
            results.append((index, score))
    else:
        # Set up multiprocessing queues
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        for index, row in df.iterrows():
            task_queue.put((index, row))

        # Process tasks with multiple workers
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_workers
        ) as executor:
            futures = []
            for id in range(args.num_workers):
                futures.append(
                    executor.submit(worker, task_queue, result_queue, args, id)
                )

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Finished workers",
            ):
                future.result()

        # Collect and sort results
        while not result_queue.empty():
            index, area_list = result_queue.get()
            results.append((index, area_list))

    results.sort(key=lambda x: x[0])
    df["ocr score"] = [x[1] for x in results]

    df.to_csv(out_path, index=False)
    print(f"New csv (shape={df.shape}) with ocr results saved to '{out_path}'.")


if __name__ == "__main__":
    main()
