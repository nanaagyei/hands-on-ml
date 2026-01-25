"""
Image handling utilities for house images.
"""
import requests
from pathlib import Path
from apps.config import DEFAULT_HOUSE_IMAGES, IMAGE_CACHE_DIR, UNSPLASH_ACCESS_KEY


def get_house_image_url(house_style=None, use_cache=True):
    """
    Get house image URL based on house style.
    
    Parameters
    ----------
    house_style : str, optional
        House style (e.g., '1Story', '2Story')
    use_cache : bool
        Whether to use cached images
        
    Returns
    -------
    str
        Image URL
    """
    if not house_style:
        return DEFAULT_HOUSE_IMAGES['default']
    
    # Map house style to image
    style_key = house_style if house_style in DEFAULT_HOUSE_IMAGES else 'default'
    return DEFAULT_HOUSE_IMAGES.get(style_key, DEFAULT_HOUSE_IMAGES['default'])


def get_unsplash_image(query="house", width=800, height=600):
    """
    Get image from Unsplash API.
    
    Parameters
    ----------
    query : str
        Search query
    width : int
        Image width
    height : int
        Image height
        
    Returns
    -------
    str
        Image URL
    """
    if not UNSPLASH_ACCESS_KEY:
        # Fallback to default
        return DEFAULT_HOUSE_IMAGES['default']
    
    try:
        url = f"https://api.unsplash.com/photos/random"
        params = {
            "query": query,
            "client_id": UNSPLASH_ACCESS_KEY,
            "w": width,
            "h": height
        }
        
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data['urls']['regular']
    except:
        pass
    
    # Fallback
    return DEFAULT_HOUSE_IMAGES['default']


def get_house_images_by_type(house_type):
    """Get multiple images for a house type."""
    base_url = get_house_image_url(house_type)
    # Return a list of similar image URLs (in production, you'd fetch multiple)
    return [base_url]
