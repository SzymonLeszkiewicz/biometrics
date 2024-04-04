from PIL import Image
import cv2
import numpy as np


class GaussianTransformer:
    """
    Class to transform images with gaussian noise.
    """

    def __init__(self):
        """
        Initialize a GaussianTransformer
        """
        self.PSNR_thresholds = {
            50: (50, 80),
            40: (40, 50),
            30: (30, 40),
            20: (20, 30),
            10: (10, 20),
        }  # thresholds for transformations
        # with PSNR

    def transform(self, input_image: Image, PSNR_dB=20, verbose=True) -> Image:
        """

        :param input_image: Image to transform
        :param PSNR_dB: Value of PSNR in dB
        :param verbose: If True, generated PSNR value will be printed
        :return: transformed image
        """
        sigma = np.sqrt(
            (255**2) / (10 ** (PSNR_dB / 10))
        )  # standard deviation based on PSNR_dB
        noise = np.random.normal(loc=0, scale=sigma, size=input_image.shape)
        noisy_image = np.clip(input_image + noise, 0, 255).astype(np.uint8)
        generated_psnr = cv2.PSNR(input_image, noisy_image)
        if PSNR_dB in list(self.PSNR_thresholds.keys()) and verbose:
            lower_th = self.PSNR_thresholds[PSNR_dB][0]
            upper_th = self.PSNR_thresholds[PSNR_dB][1]
            if lower_th <= np.round(generated_psnr) <= upper_th:
                print(
                    f"PSNR == {np.round(generated_psnr)} is in range {lower_th}, {upper_th}"
                )
            else:
                print(
                    f"Could not generate noise in range for PSNR={PSNR_dB}, generated PSNR={generated_psnr}"
                )

        return noisy_image

    def transform_directory(
        self,
        images_transformation_directory: str = None,
        transformed_images_directory: str = None,
    ):
        """

        :param images_transformation_directory: Directory from which images will be transformed
        :param transformed_images_directory: Directory to which transformed images will be stored/
        """
        raise NotImplementedError
