import unittest
import sys
import cv2
import sys
from skimage.metrics import structural_similarity as ssim

sys.path.append("../")

from RL import load_image_mask, analyze_image, mask_red, rotate_scale

class TrackingTest(unittest.TestCase):


    def test_load_image_mask(self):
        image = cv2.cvtColor(cv2.imread("../../img_check/unit_test/image.jpg"), cv2.COLOR_BGR2RGB)
        mask = cv2.imread("../../img_check/unit_test/mask_tresh.jpg")
        res_img, res_msk = load_image_mask("../../img_check/unit_test/", "image.jpg", "mask.jpg")
        ssim1 = ssim(image, res_img, multichannel=True)
        ssim2 = ssim(mask, res_msk, multichannel=True)
        self.assertGreater(ssim1, 0.999)
        self.assertGreater(ssim2, 0.999)

    def test_analyze_image(self):
        image = cv2.cvtColor(cv2.imread("../../img_check/unit_test/image.jpg"), cv2.COLOR_BGR2RGB)

        mask = cv2.imread("../../img_check/unit_test/mask_tresh.jpg")
        self.assertEqual(1.0039, analyze_image(image, mask))

    def test_mask_red(self):
        mask = cv2.cvtColor(cv2.imread("../../img_check/unit_test/mask_tresh.jpg"), cv2.COLOR_BGR2RGB)
        red_mask = cv2.imread("../../img_check/unit_test/mask_red.jpg")
        red = mask_red(mask)
        sim = ssim(red_mask, red, multichannel=True)
        self.assertGreater(sim, 0.95)

    def test_rotate_scale(self):
        mask = cv2.imread("../../img_check/unit_test/mask_red.jpg")
        rot_mask = cv2.imread("../../img_check/unit_test/mask_rot_scaled.jpg")
        rot = rotate_scale(mask, 90, 2)
        sim = ssim(rot_mask, rot, multichannel=True)
        self.assertGreater(sim, 0.99)


if __name__ == '__main__':
    unittest.main()
