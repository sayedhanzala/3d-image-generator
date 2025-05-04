import argparse
from image_to_3d import generate_3d_from_image
from text_to_3d import generate_3d_from_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert photo or text to 3D model")
    parser.add_argument("--image", type=str, help="Path to input image (.jpg/.png)")
    parser.add_argument(
        "--text", type=str, help='Text prompt (e.g., "A small toy car")'
    )
    args = parser.parse_args()

    if args.image:
        generate_3d_from_image(args.image)
    elif args.text:
        generate_3d_from_text(args.text)
    else:
        print("Please provide either an image path or a text prompt.")
