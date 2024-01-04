import requests
from PIL import Image
from io import BytesIO
import time


def load_image_from_url(url, max_retries=3, delay=2):
    """
    Loads an image from a given URL into a PIL Image object with retry mechanism and debug logging.

    :param url: URL of the image to be loaded.
    :param max_retries: Maximum number of retries if the first attempt fails.
    :param delay: Delay in seconds before retrying.
    :return: PIL Image object.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1} to load image from URL: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            print("Image successfully loaded")
            return image
        except requests.RequestException as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting for {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print(f"Failed to load image after {max_retries} attempts")
                raise Exception(
                    f"Failed to load image after {max_retries} attempts: {e}"
                )


# Example usage:
# image_url = "https://example.com/image.jpg"
# image = load_image_from_url(image_url)
