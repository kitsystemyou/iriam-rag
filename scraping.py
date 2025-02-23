from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import shutil

# Chrome の実行ファイルを探す
chrome_path = shutil.which("google-chrome") or shutil.which("chromium") or "/usr/bin/chromium-browser"

# Chrome オプションを設定
options = Options()
options.binary_location = chrome_path  # ← Chrome のパスを指定
options.add_argument("--headless=new")  # ヘッドレスモード（新バージョンの安定版）
options.add_argument("--no-sandbox")  # 必須（サンドボックスを無効化）
options.add_argument("--disable-dev-shm-usage")  # 必須（共有メモリの問題を回避）
options.add_argument("--remote-debugging-port=9222")  # デバッグポートを指定
options.add_argument("--disable-gpu")  # GPU を無効化（特に Linux での問題回避）
options.add_argument("--disable-software-rasterizer")  # ソフトウェアレンダリングを無効化
options.add_argument("--disable-background-networking")  # バックグラウンドの通信を抑制
options.add_argument("--disable-sync")  # Chrome の同期を無効化
options.add_argument("--disable-translate")  # 翻訳機能を無効化
options.add_argument("--disable-extensions")  # 拡張機能を無効化
options.add_argument("--disable-popup-blocking")  # ポップアップブロックを無効化
options.add_argument("--disable-features=OptimizationGuideModelDownloading,OptimizationHintsFetching,OptimizationTargetPrediction")  # 一部の最適化機能を無効化

# ChromeDriver を取得
service = Service(ChromeDriverManager().install())

# WebDriver の起動
driver = webdriver.Chrome(service=service, options=options)

# Web ページを開く
driver.get("https://support.iriam.com/hc/ja")
print(driver.title)

# 終了
driver.quit()
