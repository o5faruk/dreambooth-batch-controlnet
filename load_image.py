import requests
from PIL import Image
from io import BytesIO
import time


def load_image_from_url(url, max_retries=3, delay=0.2):
    """
    Loads an image from a given URL into a PIL Image object with retry mechanism.

    :param url: URL of the image to be loaded.
    :param max_retries: Maximum number of retries if the first attempt fails.
    :param delay: Delay in seconds before retrying.
    :return: PIL Image object.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            image = Image.open(BytesIO(response.content))
            return image
        except (requests.HTTPError, requests.ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait for a bit before retrying
                continue
            else:
                raise Exception(
                    f"Failed to load image after {max_retries} attempts: {e}"
                )


# Example usage:
# image_url = "https://example.com/image.jpg"
# image = load_image_from_url(image_url)
