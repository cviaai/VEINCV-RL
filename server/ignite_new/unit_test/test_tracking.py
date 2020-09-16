import unittest
import sys
import cv2
import sys
from skimage.measure import compare_ssim
sys.path.append("../")

from tracking import find_peaks, convex_hull, generate_target_candidates, first_align

class TrackingTest(unittest.TestCase):

    def test_find_peaks(self):
        image = cv2.imread("../../img_check/unit_test/crop.jpg")
        peaks = [202, 190, 180, 165, 147, 130, 120, 108, 104,  80,  67,  60,  50,  40,  31,   0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = find_peaks(gray)
        self.assertListEqual(peaks, result)

    def test_convex_hull(self):
        image = cv2.imread("../../img_check/unit_test/mask.jpg")
        convex = cv2.cvtColor(cv2.imread("../../img_check/unit_test/convex_hull.jpg"), cv2.COLOR_BGR2GRAY)
        #print(image.shape)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = convex_hull(gray)

        ssim = compare_ssim(convex, result)
        self.assertGreater(ssim, 0.999)

    def test_generate_target_candidates(self):
        image = cv2.imread("../../img_check/unit_test/crop.jpg")
        mask = cv2.imread("../../img_check/unit_test/mask.jpg")
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_list = generate_target_candidates(image_gray, mask_gray)
        self.assertEqual(len(mask_list), 16)

    def test_first_align(self):
        image = cv2.imread("../../img_check/unit_test/image.jpg")
        mask = cv2.imread("../../img_check/unit_test/mask.jpg")
        crop = cv2.imread("../../img_check/unit_test/crop.jpg")

        aligned = cv2.cvtColor(cv2.imread("../../img_check/unit_test/aligned.jpg"), cv2.COLOR_BGR2GRAY)
        mask, _, x, y = first_align(image, mask, crop)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for row in range(mask.shape[0]):
            for col in range(mask.shape[1]):
                try:
                    if (mask[row, col] > 0):
                        image[row + y, col + x] = 255
                except IndexError as e:
                    print(e)
        ssim = compare_ssim(image, aligned)
        self.assertGreater(ssim, 0.999)
if __name__ == '__main__':
    unittest.main()
