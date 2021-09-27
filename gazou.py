# URLの画像取得してローカルに保存するだけ
import urllib.request as req
url = "http://uta.pw/shodou/img/28/214.png"
req.urlretrieve(url, "test.png")

import cv2
img = cv2.imread("test.png")
print(img)
