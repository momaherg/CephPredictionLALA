import numpy as np
import cv2
import torch
import random
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance

class RandomRotation:
    """
    Rotate the image and landmarks by a random angle
    
    Args:
        max_angle (float): Maximum rotation angle in degrees
        p (float): Probability of applying the transform
    """
    def __init__(self, max_angle=10.0, p=0.5):
        self.max_angle = max_angle
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Generate a random angle between -max_angle and max_angle
            angle = random.uniform(-self.max_angle, self.max_angle)
            
            # Get image dimensions
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            
            # Generate rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Rotate the image
            rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Rotate landmarks
            if landmarks.size > 0:
                # Add a column of ones to the landmarks
                ones = np.ones(shape=(len(landmarks), 1))
                points_ones = np.hstack([landmarks, ones])
                
                # Transform landmarks
                transformed_landmarks = M.dot(points_ones.T).T
                
                return {'image': rotated_image, 'landmarks': transformed_landmarks}
            
            return {'image': rotated_image, 'landmarks': landmarks}
        
        return sample


class RandomScaling:
    """
    Scale the image and landmarks randomly
    
    Args:
        scale_factor (tuple): Range for scaling factor (e.g., (0.9, 1.1) for ±10%)
        p (float): Probability of applying the transform
    """
    def __init__(self, scale_factor=(0.9, 1.1), p=0.5):
        self.scale_factor = scale_factor
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Generate a random scale factor
            scale = random.uniform(self.scale_factor[0], self.scale_factor[1])
            
            # Get image dimensions
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            
            # Create scaling matrix
            M = cv2.getRotationMatrix2D(center, 0, scale)
            
            # Scale the image
            scaled_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Scale landmarks
            if landmarks.size > 0:
                # Scale landmarks around the center
                scaled_landmarks = landmarks.copy()
                
                # Adjust landmarks based on scale factor around center
                scaled_landmarks[:, 0] = center[0] + (scaled_landmarks[:, 0] - center[0]) * scale
                scaled_landmarks[:, 1] = center[1] + (scaled_landmarks[:, 1] - center[1]) * scale
                
                return {'image': scaled_image, 'landmarks': scaled_landmarks}
            
            return {'image': scaled_image, 'landmarks': landmarks}
        
        return sample


class RandomShift:
    """
    Shift the image and landmarks randomly
    
    Args:
        max_pixels (int): Maximum shift in pixels
        p (float): Probability of applying the transform
    """
    def __init__(self, max_pixels=10, p=0.5):
        self.max_pixels = max_pixels
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Generate random shifts in pixels
            tx = random.randint(-self.max_pixels, self.max_pixels)
            ty = random.randint(-self.max_pixels, self.max_pixels)
            
            # Create transformation matrix
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            
            # Apply affine transformation to the image
            shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            
            # Shift landmarks
            if landmarks.size > 0:
                transformed_landmarks = landmarks + np.array([tx, ty])
                return {'image': shifted_image, 'landmarks': transformed_landmarks}
            
            return {'image': shifted_image, 'landmarks': landmarks}
        
        return sample


class RandomBrightness:
    """
    Randomly adjust the brightness of the image
    
    Args:
        brightness_factor (tuple): Range for brightness adjustment factor (0.8, 1.2 for ±20%)
        p (float): Probability of applying the transform
    """
    def __init__(self, brightness_factor=(0.8, 1.2), p=0.5):
        self.brightness_factor = brightness_factor
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Convert to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Randomly select a brightness factor
            factor = random.uniform(self.brightness_factor[0], self.brightness_factor[1])
            
            # Apply brightness adjustment
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)
            
            # Convert back to numpy array
            image = np.array(pil_image)
            
            return {'image': image, 'landmarks': landmarks}
        
        return sample


class RandomContrast:
    """
    Randomly adjust the contrast of the image
    
    Args:
        contrast_factor (tuple): Range for contrast adjustment factor (0.8, 1.2 for ±20%)
        p (float): Probability of applying the transform
    """
    def __init__(self, contrast_factor=(0.8, 1.2), p=0.5):
        self.contrast_factor = contrast_factor
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Convert to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Randomly select a contrast factor
            factor = random.uniform(self.contrast_factor[0], self.contrast_factor[1])
            
            # Apply contrast adjustment
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)
            
            # Convert back to numpy array
            image = np.array(pil_image)
            
            return {'image': image, 'landmarks': landmarks}
        
        return sample


class RandomBlur:
    """
    Apply random blur to the image
    
    Args:
        blur_limit (tuple): Range for blur radius
        p (float): Probability of applying the transform
    """
    def __init__(self, blur_limit=(0, 3), p=0.2):
        self.blur_limit = blur_limit
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Convert to PIL Image
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Randomly select a blur radius
            radius = random.randint(self.blur_limit[0], self.blur_limit[1])
            
            # Apply Gaussian blur
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            
            # Convert back to numpy array
            image = np.array(pil_image)
            
            return {'image': image, 'landmarks': landmarks}
        
        return sample


class RandomGaussianNoise:
    """
    Add random Gaussian noise to the image
    
    Args:
        mean (float): Mean of Gaussian noise (usually 0)
        std_range (tuple): Range for standard deviation of Gaussian noise as fraction of pixel range
        p (float): Probability of applying the transform
    """
    def __init__(self, mean=0, std_range=(0.001, 0.01), p=0.2):
        self.mean = mean
        self.std_range = std_range
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Convert image to float for adding noise
            image_float = image.astype(np.float32)
            
            # Calculate standard deviation as a fraction of pixel range
            pixel_range = 255.0
            std = random.uniform(self.std_range[0], self.std_range[1]) * pixel_range
            
            # Generate Gaussian noise
            noise = np.random.normal(self.mean, std, image.shape)
            
            # Add noise to image
            noisy_image = image_float + noise
            
            # Clip values to valid range
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
            
            return {'image': noisy_image, 'landmarks': landmarks}
        
        return sample


class RandomHorizontalFlip:
    """
    Horizontally flip the image and landmarks
    Note: This should be used carefully for facial landmarks as it changes the anatomical meaning
    
    Args:
        p (float): Probability of applying the transform
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        if random.random() < self.p:
            # Flip image
            flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
            
            # Flip landmarks
            if landmarks.size > 0:
                w = image.shape[1]
                flipped_landmarks = landmarks.copy()
                flipped_landmarks[:, 0] = w - flipped_landmarks[:, 0]  # Flip x-coordinates
                
                # Here you might need to swap certain landmark pairs
                # For example, if landmarks 0 and 1 are left and right eye
                # you would swap them after flipping
                # This depends on the specific landmark ordering in your dataset
                
                return {'image': flipped_image, 'landmarks': flipped_landmarks}
            
            return {'image': flipped_image, 'landmarks': landmarks}
        
        return sample


class Compose:
    """
    Compose several transforms together
    
    Args:
        transforms (list): List of transforms to compose
    """
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def get_train_transforms(include_horizontal_flip=False):
    """
    Get transforms for training based on specified requirements
    
    Args:
        include_horizontal_flip (bool): Whether to include horizontal flipping
            Note: Horizontal flipping should be used carefully with anatomical landmarks
            
    Returns:
        Compose: Composed transforms
    """
    transforms_list = [
        # Geometric augmentations
        RandomRotation(max_angle=10.0, p=0.5),            # Random rotation: ±10°
        RandomScaling(scale_factor=(0.9, 1.1), p=0.5),    # Random scaling: ±10%
        RandomShift(max_pixels=10, p=0.5),                # Random translation: ±10 pixels
        
        # Photometric augmentations
        RandomBrightness(brightness_factor=(0.8, 1.2), p=0.5),  # Brightness jitter: ±20%
        RandomContrast(contrast_factor=(0.8, 1.2), p=0.5),      # Contrast jitter: ±20%
        RandomGaussianNoise(std_range=(0.001, 0.01), p=0.2),    # Slight Gaussian noise
        RandomBlur(blur_limit=(0, 2), p=0.2)                    # Occasional blur
    ]
    
    if include_horizontal_flip:
        transforms_list.append(RandomHorizontalFlip(p=0.5))
    
    return Compose(transforms_list)


def get_val_transforms():
    """
    Get transforms for validation/testing (no augmentation)
    
    Returns:
        Compose: Composed transforms (empty for validation - just return sample as is)
    """
    return Compose([])


def apply_augmentation(sample, transforms):
    """
    Apply transforms to a sample
    
    Args:
        sample (dict): Sample containing 'image' and 'landmarks'
        transforms (callable): Transforms to apply
        
    Returns:
        dict: Transformed sample
    """
    return transforms(sample) 