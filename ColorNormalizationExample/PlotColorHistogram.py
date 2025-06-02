from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse

phoenix_dataset_stats = {
    "mean": [0.6374226, 0.5848234, 0.56568706],
    "std": [0.20125638, 0.22521368, 0.2639905]
}

affectnet_dataset_stats = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

def plot_rgb_histogram(image_path, save_path="rgb_histogram.pdf"):
    """
    Plots the histogram of RGB color distributions for a given image.

    Args:
        image_path (str): Path to the input image.
        save_path (str, optional): Path to save the generated plot.
                                   Defaults to "rgb_histogram.pdf".
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    print("Channels", len(img.split()))

    # Split the image into individual R, G, B bands
    # Drop the alpha channel, if needed
    r_band, g_band, b_band = img.split()[:3]

    # Convert PIL Image bands to NumPy arrays for calculations
    r_data = np.array(r_band)
    g_data = np.array(g_band)
    b_data = np.array(b_band)

    # Calculate histograms for each band
    # .flatten() is used to get all pixel values into a 1D array for histogram calculation
    hist_r, bins_r = np.histogram(r_data.flatten(), bins=256, range=(0, 256))
    hist_g, bins_g = np.histogram(g_data.flatten(), bins=256, range=(0, 256))
    hist_b, bins_b = np.histogram(b_data.flatten(), bins=256, range=(0, 256))

    # Calculate the mean for each channel
    mean_r = np.mean(r_data)
    mean_g = np.mean(g_data)
    mean_b = np.mean(b_data)

    # Plotting the histograms
    plt.figure(figsize=(7, 3))

    bin_centers = (bins_r[:-1] + bins_r[1:]) / 2

    plt.plot(bin_centers, hist_r, color='red', alpha=0.7)
    plt.plot(bin_centers, hist_g, color='green', alpha=0.7)
    plt.plot(bin_centers, hist_b, color='blue', alpha=0.7)

    # Set horizontal range to cover 0-255
    plt.xlim(0, 255)

    # Plot vertical dotted lines at the mean value for each channel
    plt.axvline(x=mean_r, color='red', linestyle=':', label=f'Mean Red')
    plt.axvline(x=mean_g, color='green', linestyle=':', label=f'Mean Green')
    plt.axvline(x=mean_b, color='blue', linestyle=':', label=f'Mean Blue')

    # Plot vertical dotted lines at the mean value for each channel for the AffectNet dataset
    afnet_means = np.array(affectnet_dataset_stats["mean"]) * 256.0
    plt.axvline(x=afnet_means[0], color='red', linestyle='--', label=f'AffectNet Mean Red')
    plt.axvline(x=afnet_means[1], color='green', linestyle='--', label=f'AffectNet Mean Green')
    plt.axvline(x=afnet_means[2], color='blue', linestyle='--', label=f'AffectNet Mean Blue')

    # Final touch
    plt.title('RGB Color Distribution Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot in PDF format
    try:
        plt.savefig(save_path, format='pdf')
        print(f"RGB histogram plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot to PDF: {e}")
    finally:
        plt.close() # Close the plot to free up memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the histogram of RGB color distributions for an input image.")
    parser.add_argument("-i", "--input_image", type=str, required=True,
                        help="Path to the input image file.")
    parser.add_argument("-o", "--output_plot", type=str, default="rgb_histogram.pdf", required=True,
                        help="Path to save the output histogram plot (PDF format). Defaults to 'rgb_histogram.pdf'.")

    args = parser.parse_args()

    plot_rgb_histogram(args.input_image, args.output_plot)

    # Example of how to run this from the command line:
    # python your_script_name.py my_image.png -o my_histogram.pdf
    # (Replace 'your_script_name.py' with the actual name of your Python file)