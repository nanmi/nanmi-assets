import json
import os

# Define the input COCO dataset annotations JSON file path
coco_annotation_file = "path/to/coco/annotations/file.json"

# Define the output directory to save the YOLO format annotations
output_dir = "path/to/output/directory"

# Define the mapping of class names to class IDs
# class_map = {"person": 0, "car": 1, "truck": 2, "bus": 3}

# Load the COCO dataset annotations
with open(coco_annotation_file, encoding='utf-8') as f:
    coco_data = json.load(f)

# Loop through each image in the dataset
for image in coco_data["images"]:
    image_id = image["id"]
    image_filename = image["file_name"]
    image_width = image["width"]
    image_height = image["height"]

    # Create a new YOLO format annotation file for the image
    output_file_path = os.path.join(output_dir, os.path.splitext(image_filename)[0] + ".txt")
    with open(output_file_path, "w") as f:

        # Loop through each annotation for the image
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_id:

                # Convert the bounding box coordinates from COCO format to YOLO format
                bbox = annotation["bbox"]
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                x_center /= image_width
                y_center /= image_height
                width = bbox[2] / image_width
                height = bbox[3] / image_height

                # Write the YOLO format annotation to the output file
                class_id = int(annotation["category_id"])
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, x_center, y_center, width, height))
