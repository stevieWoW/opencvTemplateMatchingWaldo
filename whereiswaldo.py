import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

face = cv.imread('waldoface.jpg',0)
img_rgb = cv.imread('whereiswaldo.jpg')
img_gray = cv.imread('whereiswaldo.jpg',0)
w,h = face.shape[::-1]

res = cv.matchTemplate(face,img_gray,cv.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

print(min_val, max_val, min_loc, max_loc)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
print(top_left, bottom_right)

cv.rectangle(img_rgb,top_left,bottom_right, (0, 255, 0), 2)
cv.putText(img_rgb,'Waldo', bottom_right, cv.FONT_HERSHEY_TRIPLEX,3, (0,255,0),3)
plt.imshow(img_rgb)
plt.suptitle('Ergebnis')
plt.show()

cv.imwrite('Waldogefunden.jpg', img_rgb)

