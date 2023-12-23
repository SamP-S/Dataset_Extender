import os
import unittest
import ocve

class OCVETest(unittest.TestCase):

    red_colour_lower = (164, 98, 91)
    red_colour_higher = (195, 141, 141)
    black_colour_lower = (0, 0, 0)
    black_colour_higher = (120, 120, 120)

    HOME = os.getenv("HOME")
    CWD = os.getcwd()

    def check_version(self):
        print("check_version")
        self.assertEqual(ocve.OCVE_VERSION, "0.1.0")

    def test_add(self):
        a = 1
        b = 1
        print(f"a + b = {a+b}")
        self.assertEqual(a + b, 2)

    def tests(self):
        print(f"~ = {self.HOME}")
        print(f". = {self.CWD}")
        test_file = self.CWD + "/resources/backgrounds/coast.jpg"
        print(f"test_file = {test_file}")

        img = ocve.read_img(test_file)
        red_filter = ocve.colour_filter(img, self.red_colour_lower, self.red_colour_higher)
        black_filter = ocve.colour_filter(img, self.black_colour_lower, self.black_colour_higher)
        lines_detect = ocve.detect_lines(img)

        ocve.display_image(img)
        ocve.display_image(red_filter)
        ocve.display_image(black_filter)
        ocve.display_image(lines_detect)

        self.assertIsNotNone(img)

if __name__ == "__main__":
    unittest.main()