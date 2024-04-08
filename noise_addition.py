import cv2
import numpy as np
from typing import Optional
import os
import glob
from typing import Dict
import shutil


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

    def transform(
        self, input_image: np.ndarray, PSNR_dB=20, verbose=True
    ) -> np.ndarray:
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
        fine_tune: bool = False,
        parametrized: list = [70, 50, 30, 20, 10],
    ):
        """

        :param images_transformation_directory: Directory from which images will be transformed
        :param transformed_images_directory: Directory to which transformed images will be stored/
        """
        psnr_values = parametrized
        for psnr in psnr_values:
            print(f"Transforming images with PSNR={psnr} dB")
            for per in os.listdir(images_transformation_directory):
                if per.endswith(".pkl"):
                    print(
                        f"copying .pkl {os.path.join(images_transformation_directory, per)}, {os.path.join(transformed_images_directory, per)}"
                    )
                    print(os.getcwd())
                    shutil.copyfile(
                        os.path.join(images_transformation_directory, per),
                        os.path.join(
                            transformed_images_directory + "_psnr" + str(psnr), per
                        ),
                    )
                    continue
                per_dir = os.path.join(images_transformation_directory, per)
                trans_per_dir = os.path.join(
                    transformed_images_directory + "_psnr_" + str(psnr), per
                )
                os.makedirs(trans_per_dir, exist_ok=True)
                for img in glob.glob(per_dir + "/*.jpg"):
                    image = cv2.imread(img)
                    if fine_tune:
                        cv2.imwrite(trans_per_dir + "/" + os.path.basename(img), image)

                    noisy_image = self.transform(image, PSNR_dB=psnr, verbose=False)
                    cv2.imwrite(
                        trans_per_dir + "/t" + os.path.basename(img), noisy_image
                    )


def luminance_transform(
    input_image: np.ndarray,
    scaling_type: str = "linear",
    scale_factor: Optional[float] = 0.5,
) -> np.ndarray:
    """
    Function to perform luminance transformation
    :param input_image: Image to be transformed
    :param scaling_type: Type of transformation, available: linear, quadratic, constant
    :param scale_factor: Factor of transformation scaling. Refers to linear and constant scaling_type
    :return: Transformed image.
    """
    yuv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)  # convert image
    y_channel = yuv_image[..., 0].astype(np.float32)  # get y_channel only
    if scaling_type == "linear":
        transformed_y_channel = np.clip(y_channel * scale_factor, 0, 255).astype(
            np.uint8
        )
        yuv_image[..., 0] = transformed_y_channel
        transformed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    elif scaling_type == "quadratic":
        transformed_y_channel = y_channel**2
        min_value = np.min(transformed_y_channel)
        max_value = np.max(transformed_y_channel)
        scaled_y_channel = (
            255 * (transformed_y_channel - min_value) / (max_value - min_value)
        ).astype(
            np.uint8
        )  # perform min max scaling with range 0, 255
        yuv_image[..., 0] = scaled_y_channel
        transformed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    elif scaling_type == "constant":
        transformed_y_channel = np.clip(y_channel + scale_factor, 0, 255).astype(
            np.uint8
        )
        yuv_image[..., 0] = transformed_y_channel
        transformed_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    else:
        raise ValueError(
            "Unknown scaling type, possible scaling types: linear, quadratic, constant"
        )

    return transformed_image


def luminance_transform_directory(
    images_transformation_directory: str = None,
    transformed_images_directory: str = None,
    finetune: bool = False,
    parametrized: Dict[str, list] = {
        "quadratic": [None],
        "linear": [0.5, 0.6, 0.75, 1.33, 1.5],
        "constant": [-100, -20, -10, 30],
    },
):
    """
    Function to perform luminance transformation on images from a directory
    :param images_transformation_directory: Directory from which images will be transformed
    :param transformed_images_directory: Directory to which transformed images will be stored/
    """
    lum_transformations = parametrized
    for lum_type, scale_factors in zip(
        lum_transformations.keys(), lum_transformations.values()
    ):
        for scale_factor in scale_factors:
            print(
                f"Transforming images with {lum_type} transformation with scale factor {scale_factor}"
            )
            for per in os.listdir(images_transformation_directory):
                if per.endswith(".pkl"):
                    print(
                        f"copying .pkl {os.path.join(images_transformation_directory, per)}, {os.path.join(transformed_images_directory, per)}"
                    )
                    print(os.getcwd())
                    shutil.copyfile(
                        os.path.join(images_transformation_directory, per),
                        os.path.join(
                            transformed_images_directory
                            + "_"
                            + str(lum_type)
                            + "_"
                            + str(scale_factor),
                            per,
                        ),
                    )
                    continue
                per_dir = os.path.join(images_transformation_directory, per)
                trans_per_dir = os.path.join(
                    transformed_images_directory
                    + "_"
                    + str(lum_type)
                    + "_"
                    + str(scale_factor),
                    per,
                )
                os.makedirs(trans_per_dir, exist_ok=True)
                for img in glob.glob(per_dir + "/*.jpg"):
                    image = cv2.imread(img)
                    if finetune:
                        cv2.imwrite(trans_per_dir + "/" + os.path.basename(img), image)
                    transformed_image = luminance_transform(
                        image, scaling_type=lum_type, scale_factor=scale_factor
                    )
                    cv2.imwrite(
                        trans_per_dir + "/l" + os.path.basename(img), transformed_image
                    )
