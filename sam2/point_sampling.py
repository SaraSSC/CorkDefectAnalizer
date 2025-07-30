import numpy as np
from scipy import ndimage
from skimage import measure

def sample_points_from_mask(mask, num_points_per_component=3, min_area=20):
    """
    Sample points from each connected component in a binary mask
    
    Args:
        mask: Binary mask image
        num_points_per_component: Number of points to sample from each component
        min_area: Minimum area of a component to sample points from
        
    Returns:
        List of points as [x, y] coordinates
    """
    # Label connected components
    labeled_mask, num_components = ndimage.label(mask)
    
    if num_components == 0:
        return []
    
    # Get component properties
    regions = measure.regionprops(labeled_mask)
    
    all_points = []
    
    for region in regions:
        # Skip small regions
        if region.area < min_area:
            continue
            
        # Get coordinates of pixels in this region
        component_mask = (labeled_mask == region.label)
        coords = np.argwhere(component_mask)
        
        if len(coords) == 0:
            continue
            
        # Sample points from this component
        num_points = min(num_points_per_component, len(coords))
        
        # Get random indices without replacement if possible
        if len(coords) >= num_points:
            idx = np.random.choice(len(coords), num_points, replace=False)
        else:
            idx = np.random.choice(len(coords), num_points, replace=True)
            
        for i in idx:
            yx = coords[i]
            # Add as [x, y] format (notice the coordinate swap)
            all_points.append([yx[1], yx[0]])
    
    return all_points
