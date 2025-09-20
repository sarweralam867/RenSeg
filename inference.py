import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from skimage.segmentation import mark_boundaries
import cv2

from postprocessing import mask_image, process_and_resize_image
from preprocessing import darken_image2, refine_segmentation, segmentation_and_merge

# Paths should come from config.py
images_folder_path = None
output_folder_path = None
model = None

def load_model(weights_path):
    global model
    print("Model is preparing")
    model = YOLO(weights_path)
    print("It's ready")

def set_paths(images_path, output_path):
    global images_folder_path, output_folder_path
    images_folder_path = images_path
    output_folder_path = output_path
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

def process_images_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                results = model([image_path])

                for result in results:
                    box = result.boxes.xyxy
                    a = mask_image(image_path, box)
                    masked_img_np = np.array(a)
                    matched_img = darken_image2(masked_img_np)
                    refined_image = refine_segmentation(matched_img)
                    seg = segmentation_and_merge(refined_image)
                    boundary_img = mark_boundaries(masked_img_np, seg, color=(1, 0, 0), mode='thick')

                    relative_path = os.path.relpath(root, folder_path)
                    output_subdir = os.path.join(output_folder_path, relative_path)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    save_path = os.path.join(output_subdir, filename)
                    boundary_img_pil = Image.fromarray((boundary_img * 255).astype(np.uint8))
                    rec = process_and_resize_image(boundary_img_pil)
                    rec.save(save_path)
                    print(filename, "Saved to", save_path)
