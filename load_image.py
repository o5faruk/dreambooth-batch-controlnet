import subprocess
from PIL import Image
import os
import logging
import time
import PIL.Image
import PIL.ImageOps

# Setting up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def load_image_from_url(url, max_retries=3, delay=0.2):
    """
    Downloads an image from a URL using wget and loads it into a PIL Image object.

    :param url: URL of the image to be downloaded.
    :param max_retries: Maximum number of retries for downloading the image.
    :param delay: Delay in seconds before retrying.
    :return: PIL Image object.
    """
    temp_filename = "temp_image.jpg"
    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt+1} to download image from URL: {url}")
            subprocess.run(
                ["wget", "-O", temp_filename, url],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.debug("Image successfully downloaded")
            image = Image.open(temp_filename)
            os.remove(temp_filename)  # Clean up the temporary file
            image = PIL.ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            return image
        except subprocess.CalledProcessError as e:
            logger.error(f"Attempt {attempt+1} failed to download image: {e}")
            if attempt < max_retries - 1:
                logger.debug(f"Waiting for {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                logger.error(f"Failed to download image after {max_retries} attempts")
                raise Exception(
                    f"Failed to download image after {max_retries} attempts: {e}"
                )
        except IOError as e:
            logger.error(f"Failed to load image from file: {e}")
            raise
        finally:
            if os.path.exists(temp_filename):
                os.remove(
                    temp_filename
                )  # Ensure the temporary file is removed in case of an exception
