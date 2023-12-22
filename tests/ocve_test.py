import ocve
import os

red_colour_lower = (164, 98, 91)
red_colour_higher = (195, 141, 141)
black_colour_lower = (0, 0, 0)
black_colour_higher = (120, 120, 120)
HOME = os.getenv("HOME")

print(f"~ = {HOME}")
test_file = HOME + "/Pictures/" + input()
print(f"test_file = {test_file}")

# read core image
img = ocve.read_img()
red_filter = ocve.colour_filter(img, red_colour_lower, red_colour_higher)
black_filter = ocve.colour_filter(img, black_colour_lower, black_colour_higher)
lines_detect = ocve.detect_lines(img)

ocve.display_image(img)
ocve.display_image(red_filter)
ocve.display_image(black_filter)
ocve.display_image(lines_detect)
