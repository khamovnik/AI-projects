import cv2
import pytesseract
from google.colab.patches import cv2_imshow

def read_text_from_image(image):

# Convert the image to grayscale
  custom_config = r'-l rus --psm 6'
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Perform OTSU Threshold
  ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

  rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
  dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

  contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  image_copy = image.copy()

  for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image_copy[y: y + h, x: x + w]

    file = open("results.txt", "a")
    text = pytesseract.image_to_string(cropped, config=custom_config)
    file.write(text)
    file.write("\n")

    file.close()

image = cv2.imread("/content/Semen.jpg")
read_text_from_image(image)


cv2_imshow(image)
f = open("results.txt", "r", encoding="utf-8")
lines = f.readlines()
lines.reverse()
for line in lines:
  print(line)
  f.close()
