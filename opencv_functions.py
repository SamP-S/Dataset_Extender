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
def uniform_scale(img, factor, interp=cv2.INTER_NEAREST):
    scaled_resolution = (img.shape[0] * factor, img.shape[1] * factor)
    print(scaled_resolution)
    return cv2.resize(img, scaled_resolution, interpolation=interp)

# Scales the image using seperate scaling factors for each dimension
# cv2.INTER_NEAREST
# cv2.INTER_LINEAR
# cv2.INTER_CUBIC
# cv2.INTER_AREA
def scale(img, x_factor, y_factor, interp=cv2.INTER_NEAREST):
    scaled_resolution = (int(img.shape[0] * x_factor), int(img.shape[1] * y_factor))
    print(scaled_resolution)
    return cv2.resize(img, scaled_resolution, interpolation=interp)

# Adds guassian noise to image according to parameters
def guassian_noise(img, strength=1, mean=0, variance=400):
    gauss = np.random.normal(mean, variance**0.5, img.shape)
    gauss = gauss.reshape(img.shape)
    noisy = img + gauss * 2
    noisy = np.clip(noisy, 0, 255)
    noisy = noisy.astype(np.uint8)
    return noisy

# Adds salt and pepper noise ot image
# ratio of salt to pepper (higher ration -> more salt)
# single channel or B & W
def salt_pepper_noise(img, ratio=0.5, freq=0.05, b_w=False):
    noisy = np.copy(img)
    amount = int(freq * img.shape[0] * img.shape[1] * img.shape[2])
    salt = int(ratio * amount)

    for i in range(amount):
        coord = [np.random.randint(0, i) for i in img.shape]
        if i <= salt:
            if b_w:
                noisy[coord[0]][coord[1]] = (255, 255, 255)
            else:
                noisy[coord[0]][coord[1]][coord[2]] = 255
        else:
            if b_w:
                noisy[coord[0]][coord[1]] = (0, 0, 0)
            else:
                noisy[coord[0]][coord[1]][coord[2]] = 0
    return noisy

# Mix img and img2 using ratio
def mix(img, img2, ratio=0.2):
    return (img * ratio).astype(np.uint8) + (img2 * (1 - ratio)).astype(np.uint8)

def insert_image(img, img2, x, y):
    result = np.copy(img)
    result[y: y + img2.shape[1], x: x + img2.shape[0]] = img2
    return result

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
    test_img2_path = os.path.join(bricks_dir, "4073_0.png")
    test_img = load_image(test_img_path)
    test_img2 = load_image(test_img2_path)
    test_img = convert_colourspace(test_img)
    test_img2 = convert_colourspace(test_img2)

    # display_image(test_img)
    # display_image(test_img2)
    # display_image(rotate(test_img, cv2.ROTATE_90_CLOCKWISE))
    # display_image(uniform_scale(test_img, 2, cv2.INTER_NEAREST))
    # display_image(scale(test_img, 2, 0.5, cv2.INTER_NEAREST))
    # display_image(guassian_noise(test_img, 2))
    # display_image(salt_pepper_noise(test_img))
    # display_image(salt_pepper_noise(test_img, b_w=True))
    # display_image(mix(test_img, test_img2))
    scaled_test2_img = scale(test_img2, 0.5, 0.5)
    display_image(insert_image(test_img, scaled_test2_img, 100, 100))

