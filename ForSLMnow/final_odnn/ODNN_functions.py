import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import h5py
#%%
def place_square_by_index(index, array_size=100, square_size=5, distance=15):
    """
    Given an index, generate a 100x100 array and place the corresponding square region as 1,
    while the other regions remain 0. The square regions are distributed in a 3 4 3 arrangement
    and centered in the array.
    
    Parameters:
    - index: The index of the square (from 0 to 9).
    - array_size: The size of the array, default is 100x100.
    - square_size: The size of each square, default is 5x5.
    - distance: The distance between squares, default is 15.
    """
    
    # Initialize the array of size 100x100 with all values set to 0
    array = np.zeros((array_size, array_size), dtype=int)
    
    # Fixed region distribution: 3 4 3 (3 squares in the first row, 4 in the second, 3 in the third)
    arrangement = [3, 4, 3]  # 3 squares in the first row, 4 in the second, 3 in the third
    
    # Calculate the width of each row of squares
    region_width = square_size
    region_height = square_size
    row_widths = []
    for row_count in arrangement:
        total_width = row_count * region_width + (row_count - 1) * distance
        row_widths.append(total_width)
    
    # Calculate the vertical offset for the entire region (to center it vertically)
    total_height = sum([region_height] * len(arrangement)) + (len(arrangement) - 1) * distance
    start_y = (array_size - total_height) // 2  # Vertical starting position

    # Calculate the horizontal offset for the entire region (to center it horizontally)
    total_width = sum(row_widths)
    start_x = (array_size - total_width) // 2  # Horizontal starting position
    
    # Find the position of the square region corresponding to the given index
    region_count = 0
    row, col = None, None
    for r in range(len(arrangement)):  # Iterate over rows
        row_count = arrangement[r]  # Number of squares in the current row
        for c in range(row_count):  # Iterate over squares in the current row
            # If index matches, record the position
            if index == region_count:
                row, col = r, c
            region_count += 1
    
    # If the index corresponds to a valid square, calculate its position and fill the region
    if row is not None and col is not None:
        # Calculate the total width of the current row's squares
        total_width = arrangement[row] * region_width + (arrangement[row] - 1) * distance
        # Calculate the starting offset for each row (to center it horizontally)
        start_x = (array_size - total_width) // 2
        
        # Calculate the starting position of the square
        region_start_x = start_x + col * (region_width + distance)
        region_start_y = start_y + row * (region_height + distance)
        
        # Set the region of the square to 1
        array[region_start_y:region_start_y + square_size, 
              region_start_x:region_start_x + square_size] = 1

    # Visualize the array with only the square corresponding to the given index
    # plt.imshow(array, cmap='gray')
    # plt.title(f"Square in Region {index}")
    # plt.axis('off')  # Turn off axis display
    # plt.show()
    return array

# Example: Display the square in region with index 5
# index =  9   # Choose the index (0 to 9)
# sq_distance = 15
# sq_size = 7
# N_pixels = 100
# array_with_sq = place_square_by_index(index=index, array_size = N_pixels, distance=sq_distance, square_size=sq_size)
# plt.imshow(array_with_sq, cmap='gray')
# plt.title(f"Square in Region {index}")
# plt.axis('off')  # Turn off axis display
# plt.show()
#%% 

def place_square_by_index_p(index, array_size=100, square_size=5, distance=15):
    """
    Given an index, generate a 100x100 array and place the corresponding square region as 1,
    while the other regions remain 0. The square regions are distributed in a 3-4-3 arrangement
    and centered in the array.
    
    Parameters:
    - index: The index of the square (from 0 to 9).
    - array_size: The size of the array, default is 100x100.
    - square_size: The size of each square, default is 5x5.
    - distance: The distance between squares, default is 15.
    
    Returns:
    - array: A 100x100 array with the square for the given index.
    - position: A tuple (start_x, start_y) indicating the top-left corner of the square for the given index.
    """
    
    # Initialize the array of size 100x100 with all values set to 0
    array = np.zeros((array_size, array_size), dtype=int)
    
    # Fixed region distribution: 3 4 3 (3 squares in the first row, 4 in the second, 3 in the third)
    arrangement = [3, 4, 3]  # 3 squares in the first row, 4 in the second, 3 in the third
    
    # Calculate the width of each row of squares
    region_width = square_size
    region_height = square_size
    row_widths = []
    for row_count in arrangement:
        total_width = row_count * region_width + (row_count - 1) * distance
        row_widths.append(total_width)
    
    # Calculate the vertical offset for the entire region (to center it vertically)
    total_height = sum([region_height] * len(arrangement)) + (len(arrangement) - 1) * distance
    start_y = (array_size - total_height) // 2  # Vertical starting position

    # Calculate the horizontal offset for the entire region (to center it horizontally)
    total_width = sum(row_widths)
    start_x = (array_size - total_width) // 2  # Horizontal starting position
    
    # Find the position of the square region corresponding to the given index
    region_count = 0
    row, col = None, None
    for r in range(len(arrangement)):  # Iterate over rows
        row_count = arrangement[r]  # Number of squares in the current row
        for c in range(row_count):  # Iterate over squares in the current row
            # If index matches, record the position
            if index == region_count:
                row, col = r, c
            region_count += 1
    
    # If the index corresponds to a valid square, calculate its position and fill the region
    if row is not None and col is not None:
        # Calculate the total width of the current row's squares
        total_width = arrangement[row] * region_width + (arrangement[row] - 1) * distance
        # Calculate the starting offset for each row (to center it horizontally)
        start_x = (array_size - total_width) // 2
        
        # Calculate the starting position of the square
        region_start_x = start_x + col * (region_width + distance)
        region_start_y = start_y + row * (region_height + distance)
        
        # Set the region of the square to 1 in the array
        array[region_start_y:region_start_y + square_size, 
              region_start_x:region_start_x + square_size] = 1

        # Return the position of the square (top-left corner)
        position = (region_start_x, region_start_y)
    else:
        position = None  # In case the index is invalid (though it shouldn't be)

    return array, position

# Example usage:
# index = 5
# array, position = place_square_by_index_p(index)

# print(f"Position of square with index {index}: {position}")

#%%
def MNIST_lable_to_image(train_dataset,N_pixels=100,sq_distance = 15,sq_size=7):
    """
    Convert the lable to an image 
    
    Parameters:
    - train_dataset: MNIST dataset 
    - index: The index of the square (from 0 to 9).
    - array_size: The size of the array, default is 100x100.
    - square_size: The size of each square, default is 5x5.
    - distance: The distance between squares, default is 15.
    """
    arrays_with_sq = torch.zeros(N_pixels,N_pixels,10)
    
    for i in range(10):
        arrays_with_sq[:,:,i] = torch.from_numpy(place_square_by_index(index=i, array_size = N_pixels, distance=sq_distance, square_size=sq_size))
     
    # Create tensor for storing converted images and labels
    Imgaes = torch.zeros([train_dataset.data.size(0),1,N_pixels,N_pixels],dtype=torch.double)
    Labels = torch.zeros(train_dataset.data.size(0),1,N_pixels,N_pixels,dtype=torch.double)
    print('In processing...')
    for i, (images, labels) in enumerate(train_dataset):
        Imgaes[i,0,:,:]=   images 
        
        Label_image = arrays_with_sq[:,:,labels]
        Labels[i,0,:,:]=   Label_image

        # images_phase = torch.exp(1j*2.0*np.pi*images) # convert to phase image 
        # print(i)
        # if i>9999:
        #     break
    print('done')
    train_dataset_new = torch.utils.data.TensorDataset(Imgaes, Labels)
        
    return train_dataset_new

# #%%  test 
# if 0:
        
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Using Device: ',device)
    
#     BATCH_SIZE = 16
#     IMG_SIZE = 50
#     N_pixels = 100
#     PADDING = (N_pixels - IMG_SIZE) // 2  # zero padding, avoid loss of edge information
    
#     # Define your transformations
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to target size
#         transforms.Pad(PADDING, fill=0, padding_mode='constant'),  # Add padding with 0 (black padding)
#         transforms.ToTensor()  # Convert to tensor
#     ])
    
#     # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE))])
#     train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    
#     train_dataset_new = MNIST_lable_to_image(train_dataset,N_pixels=100,sq_distance = 15,sq_size=7)


# %% energy calculation 
# aa = test_output * arrays_with_sq[:,:,0]

def detector(output,detection_arrangement):
    """
    Simulates a detector that selects the region with the highest energy response 
    from a set of predefined detection masks.

    Parameters:
    -----------
    output : torch.Tensor
        A 2D tensor representing the output optical field (e.g., amplitude or intensity).
    detection_arrangement : np.ndarray
        A 3D NumPy array of shape [H, W, 10], where each slice [:, :, i] is a binary mask 
        for the i-th detection region.

    Returns:
    --------
    max_energy_index : int
        The index (0–9) of the detection mask that collects the maximum energy.

    Functionality:
    --------------
    - Multiplies the output with each of the 10 binary masks.
    - Computes the total energy (sum of pixel values) within each region.
    - Returns the index of the region with the highest energy.
    """
    energies = []
    
    # Iterate over each mask and calculate the energy
    for m_idx in range(10):
    
        masked_image = output.detach().cpu() * detection_arrangement[:,:,m_idx]
        
        # Calculated energy: sum of pixels in the region
        energy = torch.sum(masked_image)
        energies.append(energy)
    
        # Return the mask index corresponding to the maximum energy
    
    max_energy_index = np.argmax(energies)
    return max_energy_index

#%%
#create circles auto positions
def create_circles_auto_positions(H, W, N, radius):
    """
    Automatically generates and visualizes N non-overlapping circular regions arranged 
    in a regular grid layout within a 2D image of specified dimensions.

    Parameters:
    -----------
    H : int
        Height of the output image (in pixels).
    W : int
        Width of the output image (in pixels).
    N : int
        Number of circles to place in the image.
    radius : int
        Radius of each circle (in pixels).

    Returns:
    --------
    output_image : np.ndarray
        A 2D binary array of shape (H, W) with N circular regions set to 1, background 0.

    Functionality:
    --------------
    - Computes a grid layout based on N to fit circles with equal spacing.
    - Places circles centered on calculated grid positions.
    - Renders the result as a binary mask and displays the image using matplotlib.
    
    Raises:
    -------
    ValueError
        If the given number and size of circles cannot be placed within the array dimensions.
    """
    
    # Initialize the array with zeros
    output_image = np.zeros((H, W))
    
    # Calculate number of rows and columns in the grid based on N
    num_rows = int(np.floor(np.sqrt(N)))  # Number of rows in the grid
    num_cols = int(np.ceil(N / num_rows))  # Number of columns in the grid
    
    # Calculate horizontal and vertical spacing between circles
    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    
    # Check if there is enough space to place the circles
    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('The circles cannot fit into the array with the given size and number of circles.')
    
    # Initialize counter for the number of circles placed
    circle_count = 0
    
    # Loop through rows and columns to place circles
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                # Calculate the center of each circle
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                
                # Create the circle by setting pixels within the radius to 1
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                output_image[dist_from_center <= radius] = 1
                
                # Increment the circle count
                circle_count += 1
    
    # Display the generated image
    plt.figure()
    plt.imshow(output_image, cmap='gray')
    plt.title(f'{N} circles with radius {radius}')
    plt.axis('off')
    plt.show()
    
    return output_image

#%%

def create_detection_regions(H, W, N, radius, detectsize):
    """
    Creates a grid of N circular detection regions in a 2D image and returns bounding boxes 
    (square regions) centered on each circle for further processing or evaluation.

    Parameters:
    -----------
    H : int
        Height of the output image (in pixels).
    W : int
        Width of the output image (in pixels).
    N : int
        Total number of detection circles to place.
    radius : int
        Radius of each circular region (in pixels).
    detectsize : int
        Extra padding around each circle to define a square detection region (in pixels).

    Returns:
    --------
    detection_regions : list of tuples
        A list of N tuples, each representing a square region around a detection circle 
        in the form (x_start, x_end, y_start, y_end).

    Functionality:
    --------------
    - Circles are placed in a regular grid layout within the given image size.
    - Each detection region is a square centered on a circle, with size based on radius + detectsize.
    - A visualization is shown using matplotlib with circles marked in white (value 1).
    
    Raises:
    -------
    ValueError
        If the circles cannot fit into the given image dimensions with the specified spacing.
    """

    output_image = np.zeros((H, W))
    

    num_rows = int(np.floor(np.sqrt(N))) 
    num_cols = int(np.ceil(N / num_rows)) 
    
    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)

    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('Circles of a given size and number cannot be placed in an array.')
    
    circle_count = 0
    detection_regions = []
    
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                

                half_size = radius + detectsize
                x_start = max(center_col - half_size, 0)
                x_end = min(center_col + half_size, W)
                y_start = max(center_row - half_size, 0)
                y_end = min(center_row + half_size, H)
                detection_regions.append((x_start, x_end, y_start, y_end))
                
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                output_image[dist_from_center <= radius] = 1
                

                circle_count += 1
    

    plt.figure()
    plt.imshow(output_image, cmap='gray')
    plt.title(f'{N} Detection Regions')
    plt.axis('off')
    plt.show()
    
    return detection_regions

H, W, N, radius, detectsize = 100, 100, 3, 5, 10

# # generate the detection regions
# detection_regions = create_detection_regions(H, W, N, radius, detectsize)
# print(detection_regions)
#%%
def create_evaluation_regions(H, W, N, radius, detectsize):
    """
        Generates an array with N circular detector regions arranged in a grid, 
        and creates square evaluation regions centered on each circle's center.
    
        Parameters:
        -----------
        H : int
            Height of the output image (in pixels).
        W : int
            Width of the output image (in pixels).
        N : int
            Total number of circular detector regions to generate.
        radius : int
            Radius of each circular region (in pixels).
        detectsize : int
            Side length of the square evaluation region centered on each circle (in pixels).
    
        Returns:
        --------
        evaluation_regions : list of tuples
            A list of N tuples, each containing the bounding box of a square evaluation region 
            in the form (x_start, x_end, y_start, y_end).
    
        Functionality:
        --------------
        - Circles are evenly placed in a 2D grid with spacing computed automatically.
        - Around each circle center, a square evaluation region is defined and recorded.
        - The output image marks circular regions in white (1.0) and square regions in gray (0.5).
        - A visual preview of the layout is displayed using matplotlib.
    
        Raises:
        -------
        ValueError
            If the circles cannot fit within the given image dimensions.
        """
    
    output_image = np.zeros((H, W))
    
    
    num_rows = int(np.floor(np.sqrt(N)))  
    num_cols = int(np.ceil(N / num_rows))  
    

    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    

    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('Circles of a given size and number cannot be placed in an array.')
    
    
    circle_count = 0
    evaluation_regions = []
    
    
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                               
                half_size = detectsize // 2
                x_start = max(center_col - half_size, 0)
                x_end = min(center_col + half_size, W)
                y_start = max(center_row - half_size, 0)
                y_end = min(center_row + half_size, H)
                evaluation_regions.append((x_start, x_end, y_start, y_end))
                
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                output_image[dist_from_center <= radius] = 1
                output_image[y_start:y_end, x_start:x_end] = 0.5  
                
                circle_count += 1
    
    plt.figure(figsize=(8, 8))
    plt.imshow(output_image, cmap='gray')
    plt.title(f'{N} Evaluation Regions')
    plt.axis('off')
    plt.show()
    
    return evaluation_regions


#%%
# # create the labels of the dataset
def create_labels(H, W, N, radius, Index):
    """
    Generates a 2D binary label image containing N equally spaced circular regions, 
    with only the circle at the given Index set to 1 (all others remain zero).

    Parameters:
    -----------
    H : int
        Height of the output image (in pixels).
    W : int
        Width of the output image (in pixels).
    N : int
        Total number of circle labels to arrange in a grid layout.
    radius : int
        Radius of each circle (in pixels).
    Index : int
        Index (1-based) of the circle to activate (set to 1) in the output image.

    Returns:
    --------
    output_image : np.ndarray
        A 2D array of shape (H, W) with one active circular region (value 1), others are zero.

    Raises:
    -------
    ValueError
        If the circles cannot fit into the array with the given parameters.
    """
    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # Initialize the array with zeros
    output_image = np.zeros((H, W))
    
    # Calculate number of rows and columns in the grid based on N
    num_rows = int(np.floor(np.sqrt(N)))  # Number of rows in the grid
    num_cols = int(np.ceil(N / num_rows))  # Number of columns in the grid
    
    # Calculate horizontal and vertical spacing between circles
    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    
    # Check if there is enough space to place the circles
    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('The circles cannot fit into the array with the given size and number of circles.')
    
    # Initialize counter for the number of circles placed
    circle_count = 0
    
    # Loop through rows and columns to place circles
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                circle_count += 1  # Increment the circle count

                # Calculate the center of each circle
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                
                # Only set pixels for the circle at the specified Index
                if circle_count == Index:
                    # Create the circle by setting pixels within the radius to 1
                    Y, X = np.ogrid[:H, :W]
                    dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                    output_image[dist_from_center <= radius] = 1
                # No action needed for other circles
            else:
                break  # All circles have been processed
   
    # # plot the figures
    # plt.figure(figsize=(8, 8))
    # plt.imshow(output_image, cmap='gray')
    # plt.title(f'{N} labels')
    # plt.axis('off')
    # plt.show()
    
    return output_image


# def create_labels(H, W, N, radius, Index):
    
#     assert 1 <= Index <= N, "Index out of range."
#     num_rows = int(np.floor(np.sqrt(N)))
#     num_cols = int(np.ceil(N / num_rows))
#     row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
#     col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
#     if row_spacing < 0 or col_spacing < 0:
#         raise ValueError("Circles can't fit; shrink radius or N, or enlarge H/W.")

#     # 找第 Index 个中心
#     circle_count = 0
#     cx = cy = None
#     for r in range(1, num_rows + 1):
#         for c in range(1, num_cols + 1):
#             if circle_count >= N: break
#             circle_count += 1
#             cy0 = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
#             cx0 = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
#             if circle_count == Index:
#                 cx, cy = cx0, cy0
#                 break
#         if cx is not None: break

#     Y, X = np.ogrid[:H, :W]
#     dx, dy = X - cx, Y - cy
#     dist = np.sqrt(dx*dx + dy*dy)

#     # ---- 形状生成----
#     mask = np.zeros((H, W), dtype=np.float32)
#     which = (Index - 1) % 3  # 只 3 个模式则 0/1/2；>3 时循环

#     if which == 0:
#         # 1) 实心圆
#         mask[dist <= radius] = 1.0

#     elif which == 1:
#         # 2) 粗环：厚度约 0.5R（至少 6 像素）
#         thick = max(6, int(round(0.5 * radius)))
#         r1 = max(1, radius - thick // 2)
#         r2 = radius + thick // 2
#         mask[(dist >= r1) & (dist <= r2)] = 1.0

#     else:
#         # 3) 加号：在圆内画水平+竖直两条粗条（厚度约 0.6R）
#         thick = max(6, int(round(0.6 * radius)))
#         in_circle = dist <= radius
#         # 水平条：|y-cy| <= thick/2
#         horiz = (np.abs(dy) <= thick // 2) & in_circle
#         # 竖直条：|x-cx| <= thick/2
#         vert  = (np.abs(dx) <= thick // 2) & in_circle
#         mask[horiz | vert] = 1.0


#     return mask

def create_labels_sf(H, W, N, radius, Index, spacing_factor=1.0):
    """
    Generates a 2D binary label image containing N equally spaced circular regions,
    with only the circle at the given Index set to 1.

    Parameters
    ----------
    H : int
        Height of output image.
    W : int
        Width of output image.
    N : int
        Total number of circular labels.
    radius : int
        Radius of each circular region.
    Index : int
        Index (1-based) of the circle to activate.
    spacing_factor : float, optional
        Multiplier for spacing between circles.
        Values <1.0 = tighter layout, >1.0 = more spread out. Default is 1.0.

    Returns
    -------
    output_image : np.ndarray
        A 2D array with one circular region active (value 1), others zero.
    """
    # def create_labels(H, W, N, radius, Index, spacing_factor=1.0):
    # import numpy as np

    if Index < 1 or Index > N:
        raise ValueError("Index must be between 1 and N inclusive.")

    output_image = np.zeros((H, W))

    num_rows = int(np.floor(np.sqrt(N)))
    num_cols = int(np.ceil(N / num_rows))

    base_row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    base_col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)

    row_spacing = base_row_spacing * spacing_factor
    col_spacing = base_col_spacing * spacing_factor

    if row_spacing < 0 or col_spacing < 0:
        raise ValueError("Circles can't fit: decrease radius or spacing_factor, or increase image size.")

    # 使整个网格居中
    grid_height = num_rows * 2 * radius + (num_rows - 1) * row_spacing
    grid_width = num_cols * 2 * radius + (num_cols - 1) * col_spacing
    start_y = (H - grid_height) / 2
    start_x = (W - grid_width) / 2

    circle_count = 0
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                circle_count += 1
                center_row = round(start_y + (r - 1) * (2 * radius + row_spacing) + radius)
                center_col = round(start_x + (c - 1) * (2 * radius + col_spacing) + radius)

                if circle_count == Index:
                    Y, X = np.ogrid[:H, :W]
                    dist = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                    output_image[dist <= radius] = 1
            else:
                break

    return output_image


#%%
def create_labels_4_MMF3_phase(H, W, radius, Index):
    """
    Create a label with 5 circular regions for 3-mode fiber with phase information.
    The layout is 3 circles in the top row and 2 circles in the bottom row. 
    The 4th circle is centered below the 1st and 2nd circles, and the 5th circle
    is below the 2nd and 3rd circles.

    Parameters:
        H (int): Height of the label (image) in pixels.
        W (int): Width of the label (image) in pixels.
        radius (int): Radius of the circular regions in pixels.
        Index (int): Index of the circle to set (1 to 5).

    Returns:
        output_image (ndarray): 2D array with phase information, representing the label.
    """
    # Check for valid index
    if Index < 1 or Index > 5:
        raise ValueError("Index must be between 1 and 5 (inclusive).")

    # Initialize the array with zeros
    output_image = np.zeros((H, W))

    # Calculate horizontal spacing
    top_spacing = (W - 6 * radius) / 4  # Spacing between top 3 circles

    # Row positions for top and bottom rows
    top_row_y = H // 3  # Top row is approximately 1/3 height
    bottom_row_y = 2 * H // 3  # Bottom row is approximately 2/3 height

    # Coordinates of the top row circles
    top_centers_x = [
        int((2 * radius + top_spacing) * i + radius + top_spacing) for i in range(3)
    ]

    # Coordinates for the bottom row circles
    bottom_center_x_4 = (top_centers_x[0] + top_centers_x[1]) // 2  # 4th circle
    bottom_center_x_5 = (top_centers_x[1] + top_centers_x[2]) // 2  # 5th circle

    # Top row: 3 circles
    if Index <= 3:  # If Index corresponds to a circle in the top row
        center_x = top_centers_x[Index - 1]
        center_y = top_row_y
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1
        #output_image[dist_from_center <= radius] = 0

    # Bottom row: 4th circle
    elif Index == 4:
        center_x = bottom_center_x_4
        center_y = bottom_row_y
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1 #/ (np.sqrt(5)*np.pi)#1 #2

    # Bottom row: 5th circle
    elif Index == 5:
        center_x = bottom_center_x_5
        center_y = bottom_row_y
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1 #/ (np.sqrt(5)*np.pi) ##1

    # plt.imshow(output_image, cmap='gray')
    # plt.title(f"Label for Index {Index}")
    # plt.axis('off')
    # plt.show()
    return output_image

# usage example
H = 100
W = 100
radius = 5
Index = 3
create_labels_4_MMF3_phase(H, W, radius, Index)

#%%
def create_evaluation_regions_4_MMF3_phase(H, W, radius, detectsize):
    """
    Create evaluation regions (5 regions) for a 3-mode fiber.
    The layout includes:
    - 3 circles in the top row
    - 2 circles in the bottom row:
      - The 4th circle is centered below the 1st and 2nd circles
      - The 5th circle is centered below the 2nd and 3rd circles

    Each detection region is a `detectsize × detectsize` square centered at the circle's position.

    Parameters:
        H (int): Height of the label (image) in pixels.
        W (int): Width of the label (image) in pixels.
        radius (int): Radius of the circular regions in pixels.
        detectsize (int): Side length of the square detection region.

    Returns:
        evaluation_regions (list): List of tuples containing (x_start, x_end, y_start, y_end) 
                                   for each of the 5 evaluation regions.
    """
    
    output_image = np.zeros((H, W))

    top_spacing = (W - 6 * radius) / 4  

    top_row_y = H // 3  
    bottom_row_y = 2 * H // 3  


    top_centers_x = [
        int((2 * radius + top_spacing) * i + radius + top_spacing) for i in range(3)
    ]


    bottom_center_x_4 = (top_centers_x[0] + top_centers_x[1]) // 2  
    bottom_center_x_5 = (top_centers_x[1] + top_centers_x[2]) // 2  


    evaluation_regions = []


    for center_x, center_y in zip(
        top_centers_x + [bottom_center_x_4, bottom_center_x_5], 
        [top_row_y] * 3 + [bottom_row_y] * 2
    ):
        half_size = detectsize // 2
        x_start = max(center_x - half_size, 0)
        x_end = min(center_x + half_size, W)
        y_start = max(center_y - half_size, 0)
        y_end = min(center_y + half_size, H)
        

        evaluation_regions.append((x_start, x_end, y_start, y_end))

        output_image[y_start:y_end, x_start:x_end] = 0.5  


        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1 

    plt.figure(figsize=(8, 8))
    plt.imshow(output_image, cmap='gray')
    plt.title('Evaluation Regions for 3-Mode Fiber')
    plt.axis('off')
    plt.show()

    return evaluation_regions


# # example usage
# H = 100
# W = 100
# radius = 5
# detectsize = 10  # the size of the squart detection size
# evaluation_regions = create_evaluation_regions_4_MMF3_phase(H, W, radius, detectsize)

# # print the coordinates of the detection area 
# for i, region in enumerate(evaluation_regions):
#     print(f"Region {i + 1}: x[{region[0]}:{region[1]}], y[{region[2]}:{region[3]}]")


#%%
def generate_complex_weights(num_data, num_modes,phase_option):
    #phase_option 1: The phase array will be all zeros. This will result in a phase array where the phase of all elements is 0.
    #phase_option 2: The phase values are randomly generated between 0 and 2π for all elements, except for the first column, which is set to 0.
    #phase_option 3: Similar to Option 2, but the second column is constrained to random phase values between 0 and π.
    
    # Step 1: Create a 2D array of shape (num_data, num_modes) with values in the range (0, 1)
    amplitude_raw = np.random.rand(num_data, num_modes)
    
    # Step 2: Compute the L2 norm for each row
    norms = np.linalg.norm(amplitude_raw, axis=1, keepdims=True)
    
    # Step 3: Normalize each row by dividing by its L2 norm
    amplitude = amplitude_raw / norms
    
    # phase weights generation with different setting         
    phase = np.zeros((num_data, num_modes))

    if phase_option == 1:
    # Option 1: All phases are 0--> for ODNN-based mode decomposition
        phase[:] = 0

    elif phase_option == 2:
    # Option 2: Random phase between 0 and 2π, but first column is 0
        phase[:, 1:] = np.random.uniform(0, 2*np.pi, size=(num_data, num_modes-1))
        # phase[:, 0] = 0  # Set the first column to 0

    elif phase_option == 3:
    # Option 3: Random phase between 0 and 2π, but first column is 0, second column between 0 and π
        phase[:, 1] = np.random.uniform(0, np.pi, size=num_data)  # Set second column between 0 and π    
        phase[:, 2:] = np.random.uniform(0, 2*np.pi, size=(num_data, num_modes-2))
        # phase[:, 0] = 0  # Set the first column to 0
    
    elif phase_option == 5:
    # Option 3: Random phase between 0 and 2π, but first column is 0, second column between 0 and π
        phase[:, 1] = np.random.uniform(0, np.pi, size=num_data)  # Set second column between 0 and π    
        phase[:, 2:] = np.random.uniform(0, np.pi, size=(num_data, num_modes-2))
        # phase[:, 0] = 0  # Set the first column to 0
    
    
    return amplitude, phase


#%%
# def generate_fields_ts(complex_weights,MMF_data,num_data,num_modes,image_size):
#     # could be accelerated by using tensor 
#     # MMF_data with size of (N, H, W) -> (H, W, N)
#     image_data = torch.zeros([num_data,1,image_size,image_size], dtype=torch.complex64)
#     field = torch.zeros([image_size,image_size], dtype=torch.complex64)
#     for index in range(num_data):
#         complex_weight = complex_weights[index]
#         field = (complex_weight * MMF_data).sum(dim=2)
#         image_data[index,:,:,:]=field 
            
#         # plt.figure()
#         # plt.imshow(abs(field))
#         # plt.show()
    
#     return image_data

#%%
def generate_fields_ts(complex_weights, MMF_data, num_data, num_modes, image_size):
    """
    Generate field distributions using complex weights and MMF mode data.

    Parameters:
        complex_weights (torch.Tensor): Shape [num_data, num_modes], complex tensor.
        MMF_data (torch.Tensor): Shape [num_modes, image_size, image_size], complex tensor.
        num_data (int): Number of data samples.
        num_modes (int): Number of MMF modes.
        image_size (int): Size of the field image.

    Returns:
        image_data (torch.Tensor): Shape [num_data, 1, image_size, image_size], complex tensor.
    """
    # Ensure tensors are complex and have the correct dtype
    # complex_weights = complex_weights.to(dtype=torch.complex64)
    # MMF_data = MMF_data.to(dtype=torch.complex64)

    # Initialize output tensor
    image_data = torch.zeros([num_data, 1, image_size, image_size], dtype=torch.complex64)

    # Compute field for each sample
    for index in range(num_data):
        complex_weight = complex_weights[index]  # Shape: [num_modes]
        complex_weight = complex_weight.view(num_modes, 1, 1)  # Reshape to [num_modes, 1, 1]

        # Element-wise multiplication and sum over modes
        field = torch.sum(complex_weight * MMF_data, dim=0)  # Sum over `num_modes`, output [50, 50]
        #print(f"Complex weight       : {complex_weight}")
        # Assign field to image_data
        image_data[index, 0, :, :] = field  # Shape: [1, 50, 50]

    return image_data


def load_mmf_modes_hdf5(
    filename,
    number_of_modes,
    *,
    normalize: bool = True,
    plot: bool = True,
    save_path: str | None = None,
):
    """
    读取 HDF5/Matlab7.3 格式的多模光纤模式，并可视化每个模式的振幅与相位。

    Parameters
    ----------
    filename : str | Path
        输入文件路径。
    number_of_modes : int
        需要读取的模式数量，当前支持 3 / 5 / 10 或更多。
    normalize : bool, optional
        是否按最大幅值做归一化，默认 True。
    plot : bool, optional
        是否绘制每个模式的振幅、相位图，默认 True。
    save_path : str | None, optional
        若指定则将可视化结果保存到该路径，否则调用 plt.show()。

    Returns
    -------
    torch.Tensor
        复数模式张量，dtype=torch.complex64，shape 视数据而定。
    """
    with h5py.File(filename, "r") as f:
        print("Available groups:", list(f.keys()))

        if number_of_modes == 3:
            real_part = f["mmf_3modes_32"]["real"][()]
            imag_part = f["mmf_3modes_32"]["imag"][()]
        elif number_of_modes == 5:
            real_part = f["mmf_5modes_32"]["real"][()]
            imag_part = f["mmf_5modes_32"]["imag"][()]
        else:
            real_part = f["modes_field"]["real"][()]
            imag_part = f["modes_field"]["imag"][()]
            real_part = np.transpose(real_part, (2, 0, 1))
            imag_part = np.transpose(imag_part, (2, 0, 1))

    mmf_modes = real_part + 1j * imag_part
    if mmf_modes.ndim == 3 and mmf_modes.shape[0] != number_of_modes and mmf_modes.shape[-1] == number_of_modes:
        mmf_modes = np.transpose(mmf_modes, (2, 0, 1))

    if normalize:
        max_amp = np.max(np.abs(mmf_modes))
        if max_amp > 0:
            mmf_modes = mmf_modes / max_amp

    if plot:
        amplitude = np.abs(mmf_modes)
        phase = np.angle(mmf_modes)

        fig, axes = plt.subplots(2, amplitude.shape[0], figsize=(3.5 * amplitude.shape[0], 6), squeeze=False)

        for idx in range(amplitude.shape[0]):
            im_amp = axes[0, idx].imshow(amplitude[idx], cmap="turbo")
            axes[0, idx].set_title(f"Mode {idx + 1} |E|")
            axes[0, idx].axis("off")
            fig.colorbar(im_amp, ax=axes[0, idx], fraction=0.046, pad=0.02)

            im_phase = axes[1, idx].imshow(phase[idx], cmap="twilight")
            axes[1, idx].set_title(f"Mode {idx + 1} ∠E")
            axes[1, idx].axis("off")
            fig.colorbar(im_phase, ax=axes[1, idx], fraction=0.046, pad=0.02)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)

    return torch.tensor(mmf_modes, dtype=torch.complex64)
