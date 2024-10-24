import os
import sys
import cv2
import numpy as np
from multiprocessing import cpu_count, Pool

outputPath = "./Results/"
num_processes = cpu_count()


def warp(img):
    height = 256
    width = 256
    input = np.float32([[9,15],[233,5],[30,236],[251,227]])
    output = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    mat = cv2.getPerspectiveTransform(input, output)
    out = cv2.warpPerspective(img, mat, (width, height))
    return out


def edge_enhance(img, alpha=1, beta=0.5):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    out = cv2.addWeighted(img, alpha, laplacian, beta, 0)
    return out


def salt_and_pepper_filter(img, kernel):
    b, g, r = cv2.split(img)
    b_domain = cv2.medianBlur(b, kernel)
    g_domain = cv2.medianBlur(g, kernel)
    r_domain = cv2.medianBlur(r, kernel)
    out = cv2.merge((b_domain, g_domain, r_domain))
    return out


def nlm_filter(img):
    out = cv2.fastNlMeansDenoisingColored(img, None, 11, 6, 7, 21)
    return out


def gimp_brightness(input_img, brightness=0, contrast=0):
    # Brightness and contrast identical to gimp
    # Credits to bfris in this stackoverflow thread: https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def adjust_gamma(img, gamma=1.0):

    inv_gamma = 1.0 / gamma
    # Creates lookup tables from 0-255 where pixel/255 ^ 1/gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    #Applies lookup table to image.
    adjusted_image = cv2.LUT(img, table)

    return adjusted_image


def per_channel_gamma(img, gamma_r=1.0, gamma_g=1.0, gamma_b=1.0):
    # Based off normal gamma function but applied to each colour channel
    inv_gamma_r = 1.0 / gamma_r
    table_r = np.array([((i / 255.0) ** inv_gamma_r) * 255 for i in np.arange(0, 256)]).astype("uint8")

    inv_gamma_g = 1.0 / gamma_g
    table_g = np.array([((i / 255.0) ** inv_gamma_g) * 255 for i in np.arange(0, 256)]).astype("uint8")

    inv_gamma_b = 1.0 / gamma_b
    table_b = np.array([((i / 255.0) ** inv_gamma_b) * 255 for i in np.arange(0, 256)]).astype("uint8")

    b, g, r = cv2.split(img)

    b_corrected = cv2.LUT(b, table_b)
    g_corrected = cv2.LUT(g, table_g)
    r_corrected = cv2.LUT(r, table_r)

    adjusted_image = cv2.merge((b_corrected, g_corrected, r_corrected))

    return adjusted_image


def white_balance(image, threshold):
    '''
    This function is based on the description of the auto-white balance feature in GIMP. 
    This balances all the colour channels by stretching them to fill 0-255
    '''
    hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

    cdf_r = np.cumsum(hist_r) / np.sum(hist_r)
    cdf_g = np.cumsum(hist_g) / np.sum(hist_g)
    cdf_b = np.cumsum(hist_b) / np.sum(hist_b)

    threshold_lower = threshold
    threshold_upper = 1-threshold

    index_lower_r = np.argmax(cdf_r >= threshold_lower)
    index_upper_r = np.argmax(cdf_r >= threshold_upper)
    index_lower_g = np.argmax(cdf_g >= threshold_lower)
    index_upper_g = np.argmax(cdf_g >= threshold_upper)
    index_lower_b = np.argmax(cdf_b >= threshold_lower)
    index_upper_b = np.argmax(cdf_b >= threshold_upper)

    ''' 
    Stretches channel by subtracting lowest threshold value
    then multiplying by 255 * difference in threshold max/min
    and finally clipping from 0-255
    '''
    image[:, :, 0] = np.clip((image[:, :, 0] - index_lower_r) * (255 / (index_upper_r - index_lower_r)), 0, 255)
    image[:, :, 1] = np.clip((image[:, :, 1] - index_lower_g) * (255 / (index_upper_g - index_lower_g)), 0, 255)
    image[:, :, 2] = np.clip((image[:, :, 2] - index_lower_b) * (255 / (index_upper_b - index_lower_b)), 0, 255)


def inpaint(img, mask):
    inpainted = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
    return inpainted


def get_mask(img, diluteRadius=3):
    grey_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gets mask with a threshold of 25
    scale, mask = cv2.threshold(grey_scale, 25, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=diluteRadius)
    return mask

def conservative_smooth(img, ksize=3):
    border = ksize // 2
    out = np.zeros_like(img)
    
    for y in range(border, img.shape[0] - border):
        for x in range(border, img.shape[1] - border):
            for c in range(img.shape[2]):
                # Define neighbourhood of pixels
                neighbourhood = img[y - border:y + border + 1, x - border:x + border + 1, c]
                # Flatten neighbourhood and remove current pixel
                neighbourhood = neighbourhood.flatten()
                current_pixel = neighbourhood[(ksize**2) // 2]
                neighbourhood = np.delete(neighbourhood, (ksize**2) // 2)

                min_val = neighbourhood.min()
                max_val = neighbourhood.max()

                # Compares pixel against min/max values of neighbourhood and adjusts accordingly
                if current_pixel < min_val:
                    out[y, x, c] = min_val
                elif current_pixel > max_val:
                    out[y, x, c] = max_val
                else:
                    out[y, x, c] = current_pixel
    
    return out


def adaptive_median_filter(image, neighbourhood_size, max_neighbourhood_size):
    b, g, r = cv2.split(image)
    channels = [b, g, r]
    output_channels = []
    for channel in channels:
        intermediate = np.zeros_like(channel)
        pad_size = max_neighbourhood_size // 2

        padded_image = cv2.copyMakeBorder(channel, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

        for y in range(pad_size, padded_image.shape[0] - pad_size):
            for x in range(pad_size, padded_image.shape[1] - pad_size):
                # Initial neighbourhood definition
                neighbourhood = padded_image[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]
                current_neighbourhood_size = neighbourhood_size

                while current_neighbourhood_size <= max_neighbourhood_size:
                    median = np.median(neighbourhood)
                    current_pixel = padded_image[y, x]

                    if median > np.min(neighbourhood) and median < np.max(neighbourhood): # Comparison to zmin and zmax, level 1 check
                        if current_pixel != median: # Level 2 check
                            intermediate[y - pad_size, x - pad_size] = median 
                            break
                        else:
                            intermediate[y - pad_size, x - pad_size] = current_pixel
                            break
                    else:
                        current_neighbourhood_size += 2
                        # Defines new neighbourhood based on the updated neighbourhood size.
                        pad = current_neighbourhood_size // 2
                        neighbourhood = padded_image[y - pad:y + pad + 1,
                                x - pad:x + pad + 1]

        output_channels.append(intermediate)
    out = cv2.merge(output_channels)
    return out


def apply_CLAHE(image, clip_limit, grid_size):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    # Applying CLAHE on l channel as described in lectures.
    cl = clahe.apply(l)
    out = cv2.merge((cl, a, b))
    out = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
    return out

def apply_bilateral_filter(image, d, sigmaColor, sigmaSpace):
    filtered_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

    return filtered_image

def gaussian_blur(image,ksize,sigma):
    n_image = cv2.GaussianBlur(image, (ksize,ksize), sigma, sigma)
    return n_image


def process_image(path, filename):
    file = os.path.join(path, filename)
    if os.path.isfile(file):
        print(file)
        img = cv2.imread(file)

        img = warp(img)
        mask = get_mask(img)

        img = conservative_smooth(img)
        img = adaptive_median_filter(img, 3, 7)

        img = nlm_filter(img)
        
        img = apply_CLAHE(img, 3.0, 10)
        img = adjust_gamma(img, 0.825)
        img = per_channel_gamma(img, 1, 0.82, 0.9)
        white_balance(img, 0.0005)
        img = inpaint(img, mask)
        
        cv2.imwrite(os.path.join(outputPath, filename), img)
    else:
        print('File', file, 'not found, folder path =', path)


def process_chunk(args):
    path, chunk = args
    for filename in chunk:
        process_image(path, filename)


def chunk_files(files, num_chunks):
    # Splits files into evenly sized chunks for each process to work through
    chunk_size = len(files) // num_chunks
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    return chunks


def main():
    if len(sys.argv) > 1:
        input = sys.argv[1]
        path = "./" + input
        files = [filename for filename in os.listdir(path) if not filename.startswith('.')]
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        with Pool(num_processes) as pool:
            chunks = chunk_files(files, num_processes)
            args_list = [(path, chunk) for chunk in chunks]

            pool.map(process_chunk, args_list)
    else:
        print("No input directory provided")


if __name__ == "__main__":
    main()
