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
file_path = os.path.abspath("nw.jpeg")
input_tag.send_keys(file_path)

# use 'start-ocr' button clicked
driver.execute_script("document.getElementsByTagName('button')[5].click()")
driver.implicitly_wait(10)
# answer = wait.until(EC.element_to_be_selected(By.TAG_NAME,"textarea"))
answre = driver.find_element(By.TAG_NAME,"textarea")
print(answre.text)





