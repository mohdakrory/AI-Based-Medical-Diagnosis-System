from PIL import Image

def convert_black_to_white(image_path):
    # Open the image using PIL
    image = Image.open(image_path)
    print(image)
    # Convert the image to RGBA mode (if not already)
    image = image.convert("RGBA")
    
    # Get the image data as a list of pixels
    pixel_data = image.getdata()

    # Create a new list to store the modified pixel data
    new_pixel_data = []

    # Iterate over the pixel data
    for pixel in pixel_data:
        # Check if the pixel is black
        if pixel[:3] == (0, 0, 0):
            # Set the pixel to white
            new_pixel_data.append((255, 255, 255, pixel[3]))  # White pixel
        else:
            # Set the pixel to transparent
            new_pixel_data.append((0, 0, 0, 0))  # Transparent pixel

    # Update the image data with the modified pixel data
    image.putdata(new_pixel_data)

    # Save the modified image
    image.save("output.png")

# Example usage
convert_black_to_white("6974779.png")
