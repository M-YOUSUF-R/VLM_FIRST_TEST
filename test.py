import os
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# 1. Initialize Chrome options
chrome_options = Options()

# 2. Add the experimental detach option
chrome_options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=chrome_options)

wait = WebDriverWait(driver,10)

url = "https://banglaocr.com/"
driver.get(url)
input_tag = driver.find_element(By.CSS_SELECTOR,"input[type='file']")

IMAGE_FOLDER = "./data_folder/data/images/"
OUTPUT_FOLDER = "./data_folder/data/text/"

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        base_name = os.path.splitext(filename)[0]

        # Keep the trusted txt_0..txt_30 labels paired with img_0..img_30.
        if base_name.startswith("img_"):
            image_number = base_name.removeprefix("img_")
            trusted_text_path = os.path.join(OUTPUT_FOLDER, f"txt_{image_number}")
            if image_number.isdigit() and os.path.isfile(trusted_text_path):
                print(f"Skipping trusted label: {filename}")
                continue

        out_name = f"{base_name}.txt"
        out_path = os.path.join(OUTPUT_FOLDER, out_name).replace("\\", "/")

        img_path = os.path.join(IMAGE_FOLDER, filename)
        print(f"Regenerating label for: {filename}...")
        
        input_tag.send_keys(img_path)

        # use 'start-ocr' button clicked
        driver.execute_script("document.getElementsByTagName('button')[5].click()")
        driver.implicitly_wait(10)
        # answer = wait.until(EC.element_to_be_selected(By.TAG_NAME,"textarea"))
        label = driver.find_element(By.TAG_NAME,"textarea").text
        with open(out_path,'w') as f:
            f.write(label)

        driver.refresh()
