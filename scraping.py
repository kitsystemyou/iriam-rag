from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

# Selenium のセットアップ
options = Options()
options.add_argument("--headless")  # GUIなしで実行
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

def scrape_with_selenium():
    driver.get("https://support.iriam.com/hc/ja")
    time.sleep(3)  # 読み込み待ち
    page_source = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(page_source, "html.parser")
    return soup.prettify()

print(scrape_with_selenium())
