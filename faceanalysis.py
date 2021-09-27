# OpenCVを使用して画像ファイルから顔認識(四角枠を付ける)させるプログラム

import matplotlib.pyplot as plt
import cv2

# 画像の指定 (人物を含む画像を指定する)
img = cv2.imread("metyaike.jpg")
# グレイスケールに変換
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# グレイスケール変換確認
#plt.imshow(img_gray)
#plt.show()

# カスケードファイルの指定 
# https://github.com/opencv/opencv/tree/master/data/haarcascadesの下にあるXMLファイルは読み込み失敗するので
# 最新のOpenCVリリース版https://github.com/opencv/opencv/releases
# のソース(zip)の中身に含まれるXMLファイルを使用
# 以下はOpenCV バージョン4.5.3で動作確認済
cascade_file = "haarcascade_frontalface_alt.xml" #faceanalaysis.pyと同じ階層に置く
cascade = cv2.CascadeClassifier(cascade_file)

# 顔認識
face_list = cascade.detectMultiScale(img_gray)

# 結果
if len(face_list) == 0:
    print("顔認識失敗")
    quit()

# 認識した部分に赤枠を付ける
for (x, y, w, h) in face_list:
    print("顔の座標 =", x, y, w, h)
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), red, 2)

# 赤枠付けた画像を出力
cv2.imwrite("result.jpg", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
