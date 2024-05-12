import cv2
import numpy as np

# Initialize variables
rect = (0, 0, 1, 1)
drawing = False  # true if mouse is pressed
rect_over = False  # true if rect drawn
ix, iy = -1, -1


# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, rect_over

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect = (x, y, 0, 0)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            rect = (ix, iy, x - ix, y - iy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_over = True
        rect = (ix, iy, x - ix, y - iy)


# Load the image
image_path = '/Users/ziad/Downloads/LatexCurrent/img/XAI//.jpg'
img = cv2.imread(image_path)
if img is None:
    print("Image not found")

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

# Keep looping until the rectangle is drawn
while (1):
    img2 = img.copy()
    if drawing:
        cv2.rectangle(img2, (ix, iy), (ix + rect[2], iy + rect[3]), (0, 255, 0), 2)
    cv2.imshow('image', img2)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Press 'ESC' to exit
        break
    if rect_over:  # Rectangle finalization
        break

cv2.destroyAllWindows()

# Apply GrabCut with the drawn rectangle
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

# Convert the mask to binary format
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
segmented_img = img.copy()
segmented_img[mask2 == 0] = (255, 255, 255)  # Dimming the background

# Display the result
cv2.imshow('Segmented Image', segmented_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
