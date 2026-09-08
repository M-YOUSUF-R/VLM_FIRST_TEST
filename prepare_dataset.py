import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 1. Initialize Chrome options
chrome_options = Options()
chrome_options.add_experimental_option("detach", True)

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 15)

url = "https://banglaocr.com/"
driver.get(url)

IMAGE_FOLDER = "./dataset_folder/data/images/"
OUTPUT_FOLDER = "./dataset_folder/data/text/"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        base_name = os.path.splitext(filename)[0]

        # Keep trusted txt_0..txt_30 labels paired with img_0..img_30.
        if filename.lower().endswith('.txt') in os.listdir(OUTPUT_FOLDER):
            continue

        out_name = f"{base_name}.txt"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        img_path = os.path.abspath(os.path.join(IMAGE_FOLDER, filename))
        
        print(f"Regenerating label for: {filename}...")

        # Re-query input element after refresh to avoid StaleElementReferenceException
        input_tag = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
        )
        input_tag.send_keys(img_path)

        # Trigger "Start OCR" button via JS
        driver.execute_script("document.getElementsByTagName('button')[5].click()")

        # Wait until the textarea element exists and contains OCR results
        textarea = wait.until(
            EC.presence_of_element_located((By.TAG_NAME, "textarea"))
        )
        wait.until(lambda d: textarea.get_attribute("value").strip() != "")

        # Read output value from textarea DOM property
        label = textarea.get_attribute("value")

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(label)

        # Refresh browser session for next loop iteration
        driver.refresh()

