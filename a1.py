# Pseudocode for Image Generation and Post-Processing Program
from config import HF_API_KEY
import requests
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
def generate_image_from_text(prompt):
    api_url = "https://router.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    payload = {
        "inputs": prompt
    }
    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise ValueError(f"Error generating image: {response.status_code} - {response.text}")

def post_process_image(image):
    # Increase the brightness of the image:
    brightness_enhancer = ImageEnhance.Brightness(image)
    brightened_image = brightness_enhancer.enhance(1.2)
    # Enhance the contrast of the image:
    contrast_enhancer = ImageEnhance.Contrast(brightened_image)
    contrasted_image = contrast_enhancer.enhance(1.3)
    # Apply a soft-focus effect:
    blurred_image = contrasted_image.filter(ImageFilter.GaussianBlur(radius=2))
    return blurred_image
def main():
    prompt = "A serene landscape with mountains and a river at sunset"
    try:
        generated_image = generate_image_from_text(prompt)
        final_image = post_process_image(generated_image)
        final_image.show()  # Display the final image
        final_image.save("final_image.png")  # Save the final image
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
