import cv2 as cv
import numpy as np


def display_disparity_map(disparity_map):
    """
    Display the disparity map in a range and colorize it.
    Args:
        disparity_map: The disparity map to display

    Returns:
        _disparity_map: The disparity map with color overlay
    """
    _disparity_map = disparity_map.copy()
    # Color mark everything that is not in the range we want to display
    mask_lower = np.zeros(_disparity_map.shape, dtype=np.uint8)
    mask_lower[_disparity_map < 1] = 255
    _disparity_map[_disparity_map < 1] = 1
    # Normalize the disparity_map map to the range we want to display
    _disparity_map = cv.normalize(_disparity_map, _disparity_map, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                  dtype=cv.CV_8U)
    # Apply the masks as a color overlay
    _disparity_map = cv.applyColorMap(_disparity_map, cv.COLORMAP_JET)
    _disparity_map[mask_lower == 255] = [0, 0, 0]
    return _disparity_map


def compute_disparity(images, mask, num_disp, block_size, uniquenessRatio, speckleWindowSize, speckleRange,
                      disp12MaxDiff, mode, labda, sigma, use_mask=True):
    """
    Compute the disparity map from two rectified images.
    Args:
        images:
        mask:
        num_disp:
        block_size:
        uniquenessRatio:
        speckleWindowSize:
        speckleRange:
        disp12MaxDiff:
        mode:
        labda:
        sigma:
        use_mask:

    Returns:
        disparity: The disparity map
    """
    im1 = images[0].copy()
    im2 = images[1].copy()
    # # Convert the images to grayscale
    im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Histogram equalization
    im1 = cv.equalizeHist(im1)
    im2 = cv.equalizeHist(im2)

    # Create the stereo matcher objects with the parameters we set above
    left_matcher = cv.StereoSGBM_create(
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 3 * block_size ** 2,  # 8*img_channels*block_size**2
        P2=32 * 3 * block_size ** 2,  # 32*img_channels*block_size**2
        mode=mode  # Use Graph Cut mode
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

    # Create the wls filter and set the parameters
    wls_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
    wls_filter.setLambda(labda)
    wls_filter.setSigmaColor(sigma)

    # Compute the disparity maps
    left_disp = left_matcher.compute(im1, im2)
    right_disp = right_matcher.compute(im2, im1)

    # Apply the wls filter to the disparity map
    disparity = wls_filter.filter(disparity_map_left=left_disp, left_view=im1, disparity_map_right=right_disp)

    # Apply first mask to disparity map
    if use_mask:
        disparity[mask[0] != 255] = 0
    # important conversion for the reprojectImageTo3D function
    disparity = np.float32(np.divide(disparity, 16.0))
    return disparity


def compute_disparity_map(images, suffix, mask=None, save_path='output', display=False):
    """
    Compute the disparity map from two rectified images.
    Args:
        display:
        save_path:
        images: The two rectified images to compute the disparity map from needs to be in uint8 format
        suffix:
        mask:
    Returns:
        disparity: The disparity map
    """
    disparity = compute_disparity(images, mask,
                                  block_size=5,
                                  num_disp=64,  # Needs to be divisible by 16
                                  disp12MaxDiff=50,
                                  uniquenessRatio=5,
                                  speckleWindowSize=9,
                                  speckleRange=2,
                                  mode=cv.STEREO_SGBM_MODE_HH,
                                  labda=8000,
                                  sigma=1.5)

    # Create display disparity map
    _disparity_map = display_disparity_map(disparity)

    # Save the disparity map
    cv.imwrite(f'{save_path}{suffix}_disparity_map.png', _disparity_map)

    # Display the disparity map
    if display:
        cv.imshow("Disparity", _disparity_map)
        cv.waitKey(0)

    return disparity


def compute_disparity_map_interactively(images, mask):
    """
    Compute the disparity map from two rectified images in an interactive window,
    where the user can adjust the parameters.
    Args:
        images:
        mask:

    Returns:

    """
    # Default values
    use_mask = True
    block_size = 5
    num_disp = 64  # Needs to be divisible by 16
    uniquenessRatio = 5
    speckleWindowSize = 9
    speckleRange = 2
    disp12MaxDiff = 50
    mode = cv.STEREO_SGBM_MODE_HH  # Use Graph Cut mode
    labda = 8000
    sigma = 150

    # Create an interactive window to adjust the parameters
    cv.namedWindow("Disparity", cv.WINDOW_NORMAL)
    cv.createTrackbar("block_size", "Disparity", block_size, 50, lambda x: x)
    cv.createTrackbar("num_disparities", "Disparity", num_disp, 100, lambda x: x * 16)
    cv.createTrackbar("uniquenessRatio", "Disparity", uniquenessRatio, 100, lambda x: x)
    cv.createTrackbar("speckleWindowSize", "Disparity", speckleWindowSize, 200, lambda x: x)
    cv.createTrackbar("speckleRange", "Disparity", speckleRange, 100, lambda x: x)
    cv.createTrackbar("disp12MaxDiff", "Disparity", disp12MaxDiff, 100, lambda x: x)
    cv.createTrackbar("mode", "Disparity", mode, 1, lambda x: x)
    cv.createTrackbar("use_mask", "Disparity", use_mask, 1, lambda x: x)
    cv.createTrackbar("lambda", "Disparity", labda, 10000, lambda x: x)
    cv.createTrackbar("sigma", "Disparity", sigma, 1000, lambda x: x)

    # Wait until the user presses 'q' on the keyboard
    while cv.waitKey(1) != ord('q'):
        # Get the current trackbar positions
        block_size = cv.getTrackbarPos("block_size", "Disparity")
        num_disp = cv.getTrackbarPos("num_disparities", "Disparity")
        uniquenessRatio = cv.getTrackbarPos("uniquenessRatio", "Disparity")
        speckleWindowSize = cv.getTrackbarPos("speckleWindowSize", "Disparity")
        speckleRange = cv.getTrackbarPos("speckleRange", "Disparity")
        disp12MaxDiff = cv.getTrackbarPos("disp12MaxDiff", "Disparity")
        mode = cv.getTrackbarPos("mode", "Disparity")
        use_mask = cv.getTrackbarPos("use_mask", "Disparity")
        labda = cv.getTrackbarPos("lambda", "Disparity")
        sigma = cv.getTrackbarPos("sigma", "Disparity") / 100

        disparity = compute_disparity(images, mask, num_disp, block_size, uniquenessRatio, speckleWindowSize,
                                      speckleRange,
                                      disp12MaxDiff, mode, labda, sigma, use_mask=use_mask)
        # Create display disparity map
        _disparity_map = display_disparity_map(disparity)
        # Show the disparity map in the interactive window let the curser display the disparity value
        cv.imshow("Disparity", _disparity_map)
        # Print the disparity value at the current mouse position in the interactive window
        cv.setMouseCallback("Disparity", lambda event, x, y, flags, param: print(disparity[y, x]))
        # wait for 100ms
        cv.waitKey(100)
    # Close the window
    cv.destroyWindow("Disparity")
