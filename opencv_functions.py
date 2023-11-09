import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
   
def colour_mask(img, lower_colour, upper_colour):
    mask = cv2.inRange(img, lower_colour, upper_colour)
    masked = cv2.bitwise_and(img,img, mask=mask)
    return masked

def colour_filter(img, lower_colour, upper_colour):
    mask = cv2.inRange(img, lower_colour, upper_colour)
    masked = cv2.bitwise_and(img,img, mask=mask)
    result = img - masked
    return result

def colour_histogram(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def detect_lines(img):
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    display_image(edges)
    
    minLineLength=5
    maxLineGap=5
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=200,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)

    a,b,c = lines.shape
    for i in range(a):
        cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
        display_image(edges)


# Rotates on multiples of 90 degrees
# cv2.ROTATE_90_CLOCKWISE
# cv2.ROTATE_180
# cv2.ROTATE_90_COUNTERCLOCKWISE
def rotate(img, rot):
    return cv2.rotate(img, rot)

# Scales the image maintaining aspect ratio
# cv2.INTER_NEAREST
# cv2.INTER_LINEAR
# cv2.INTER_CUBIC
# cv2.INTER_AREA
def uniform_scale(img, factor, interp):
    scaled_resolution = (img.shape[0] * factor, img.shape[1] * factor)
    print(scaled_resolution)
    return cv2.resize(img, scaled_resolution, interpolation=interp)

# Scales the image using seperate scaling factors for each dimension
# cv2.INTER_NEAREST
# cv2.INTER_LINEAR
# cv2.INTER_CUBIC
# cv2.INTER_AREA
def scale(img, x_factor, y_factor, interp):
    scaled_resolution = (int(img.shape[0] * x_factor), int(img.shape[1] * y_factor))
    print(scaled_resolution)
    return cv2.resize(img, scaled_resolution, interpolation=interp)

# Adds guassian noise to image according to parameters
def guassian_noise(img, strength, mean=0, variance=0.1):
    gauss = np.random.normal(mean,variance**0.5, img.shape) * 255
    gauss = gauss.reshape(img.shape).astype(np.uint8)
    return gauss

# Load image from file
# cv2.IMREAD_COLOR
# cv2.IMREAD_GRAYSCALE 
# cv2.IMREAD_UNCHANGED
def load_image(path, colourspace=cv2.IMREAD_COLOR):
    return cv2.imread(path, colourspace)

# NOTE: images are BGR
def display_image(img):
    plt.imshow(img)
    plt.show()

# cv2.COLOR_BGR2RGB
# cv2.COLOR_RGB2BGR
def convert_colourspace(img, conversion=cv2.COLOR_RGB2BGR):
    return cv2.cvtColor(img, conversion)

if __name__ == "__main__":
    cwd = os.getcwd()
    bricks_dir = os.path.join(cwd, "bricks")
    bg_dir = os.path.join(cwd, "backgrounds")

    test_img_path = os.path.join(bricks_dir, "3023_0.png")
    test_img = load_image(test_img_path)
    test_img = convert_colourspace(test_img)

    display_image(rotate(test_img, cv2.ROTATE_90_CLOCKWISE))
    # display_image(uniform_scale(test_img, 2, cv2.INTER_NEAREST))
    # display_image(scale(test_img, 2, 0.5, cv2.INTER_NEAREST))
    display_image(guassian_noise(test_img, 0.1))

