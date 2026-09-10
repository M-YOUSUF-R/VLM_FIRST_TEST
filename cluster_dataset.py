import os
import shutil

ROOT_DIR = "Dataset"
IMAGE_FOLDER = "./dataset_folder/data/images/"
OUTPUT_FOLDER = "./dataset_folder/data/text/"

os.makedirs(ROOT_DIR, exist_ok=True)

image_file_names = sorted(os.listdir(IMAGE_FOLDER))
output_file_names = set(os.listdir(OUTPUT_FOLDER))  # Converted to set for O(1) lookups

cluster = ["yousuf", "sakib", "suprov"]
size = len(image_file_names)
cluster_size = size // 3
itr = 0

# 3. Distribute files equally
for i in range(len(cluster)):
    person = cluster[i]
    destination = os.path.join(ROOT_DIR, person)
    os.makedirs(destination, exist_ok=True)
    
    # Process files up to the calculated cluster size for this person
    count = 0
    while count < cluster_size and itr < size:
        image = image_file_names[itr]
        base, _ = os.path.splitext(image)
        text_file = f"{base}.txt"

        img_destination = f"{destination}/images/"
        os.makedirs(img_destination,exist_ok=True)
        text_destination = f"{destination}/text/"
        os.makedirs(text_destination,exist_ok=True)

        # Copy the image
        shutil.copy(os.path.join(IMAGE_FOLDER, image), img_destination)
        
        # Copy the matching text file if it exists
        if text_file in output_file_names:
            shutil.copy(os.path.join(OUTPUT_FOLDER, text_file), text_destination)
            
        itr += 1
        count += 1

while itr < size:
    person = cluster[0]
    destination = os.path.join(ROOT_DIR, person)
    
    img_destination = f"{destination}/images/"
    os.makedirs(img_destination,exist_ok=True)
    text_destination = f"{destination}/text/"
    os.makedirs(text_destination,exist_ok=True)

    image = image_file_names[itr]
    base, _ = os.path.splitext(image)
    text_file = f"{base}.txt"
    
    shutil.copy(os.path.join(IMAGE_FOLDER, image), img_destination)
    if text_file in output_file_names:
        shutil.copy(os.path.join(OUTPUT_FOLDER, text_file), text_destination)
        
    itr += 1

print("Distribution completed successfully!")


   




