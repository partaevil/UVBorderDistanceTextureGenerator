import bpy
import numpy as np
from scipy import ndimage
import bmesh
from mathutils import Vector
import math
import os

"""
UV DISTANCE TEXTURE GENERATOR FOR BLENDER

This script generates UV-based gradient textures for selected mesh objects in Blender.
It creates distance-based gradients that fade from white at the center of UV islands
to black at the edges, useful for creating masks, ambient occlusion-like effects,
or stylized shading textures.

config = TextureGenConfig()
config.resolution = 2048              # Output texture resolution (2048x2048)
config.falloff_rate = 3.0             # Gradient steepness (higher = sharper falloff)
config.smoothness = 1.2               # Gradient curve (>1.0 = slower start, faster end)
config.border_thickness = 1           # Black border width in pixels
config.gamma_correction = 2.2         # Gamma correction for color space
config.blur_amount = 0.5              # Gaussian blur sigma (0 = no blur)
config.edge_contrast = 1.5            # Edge sharpening multiplier
config.flip_vertical = False          # Flip final image vertically
config.uv_index = 0                   # Which UV map to use (0 = first)
config.black_background = True        # True = black bg, False = transparent
config.use_exponential_falloff = True # True = exponential, False = linear falloff
config.save_path = "/custom/path.png" # Custom save location (optional)

OUTPUT:
- Creates a new image in Blender's data
- Saves PNG file to textures folder or custom path
- Image name format: "ObjectName_combined_uv_distance"
"""

# TODO: not delete original if exist, but replace

class TextureGenConfig:
    """
    Configuration class for generating UV-based gradient textures.
    Stores all settings that control the appearance and generation process.
    """
    def __init__(self):
        # Image settings
        self.resolution: int = 1024*4
        """The resolution (width and height) of the generated texture in pixels.
           E.g., 1024 means a 1024x1024 pixel image."""

        self.flip_vertical: bool = False
        """Whether to flip the generated image vertically. Why? idk forgot
           Useful for matching different UV coordinate systems or engine requirements."""

        # Gradient settings
        self.falloff_rate: float = 3.0
        """Controls the steepness of the gradient falloff, especially when
           `use_exponential_falloff` is True. Higher values result in a quicker
           transition from white (center of UV island) towards black."""

        self.smoothness: float = 1.2
        """Adjusts the curve of the gradient.
           - Values > 1.0: The gradient starts slower from the center and accelerates towards the edge.
           - Values < 1.0: The gradient starts faster from the center and decelerates towards the edge.
           - 1.0: No additional curve adjustment (applied after the main falloff_function)."""

        self.border_thickness: int = 1
        """The thickness of the explicit black border created around each UV island, in pixels.
           This border is applied after the gradient calculation.
           A value of 0 effectively means no explicit border is carved out, relying solely on the falloff.
           A value of 1 creates a 1-pixel wide border using a default 3x3 erosion kernel.
           Values greater than 1 increase the border thickness using multiple erosion iterations."""

        # Advanced settings
        self.use_exponential_falloff: bool = True
        """If True, an exponential function (1 - exp(-falloff_rate * x)) is used
           for the primary gradient shape. If False, a linear falloff (x) is used
           as the base (before smoothness and gamma adjustments)."""

        self.gamma_correction: float = 2.2
        """Applies gamma correction to the final gradient values.
           A typical value for sRGB color space is 2.2.
           1.0 means no gamma correction is applied."""

        self.edge_contrast: float = 1.0
        """Controls the contrast enhancement near the edges of the UV islands.
           Values > 1.0 increase contrast (sharper edges), values < 1.0 decrease contrast.
           Applied after the main gradient calculation but before border processing."""

        self.blur_amount: float = 0.0
        """The sigma value for Gaussian blur applied to the gradient map.
           0.0 means no blur. Higher values result in a more blurred/softer gradient."""

        self.uv_index: int = 0
        """The index of the UV map to use from the mesh object(s).
           0 typically refers to the first (default) UV map."""

        self.save_path: str = ""
        """Custom file path (including filename and extension, e.g., '/path/to/image.png')
           where the generated texture will be saved. If left empty, the script will
           attempt to save it in a 'textures' subfolder next to the .blend file."""

        self.black_background: bool = True
        """If True, the generated image will have a black background (RGB values are the gradient, Alpha is 1.0).
           If False, the gradient is placed in all R, G, B, and Alpha channels, meaning areas with
           zero gradient become fully transparent black."""

    @property
    def falloff_function(self):
        if self.use_exponential_falloff:
            return lambda x: 1 - np.exp(-self.falloff_rate * x)
        return lambda x: x

def get_texture_folder(obj_name="combined"):
    """Get or create texture folder in blend file directory"""
    blend_file_path = bpy.data.filepath
    if not blend_file_path:
        return None

    directory = os.path.dirname(blend_file_path)
    texture_folder = os.path.join(directory, "textures")

    if not os.path.exists(texture_folder):
        os.makedirs(texture_folder)

    return texture_folder

def apply_blur(gradient, blur_amount):
    """Apply Gaussian blur to the gradient"""
    if blur_amount > 0:
        return ndimage.gaussian_filter(gradient, sigma=blur_amount)
    return gradient

def apply_edge_contrast(gradient, edge_contrast):
    """Apply edge contrast enhancement to the gradient"""
    if edge_contrast != 1.0:
        # Create edge mask by finding areas with high gradient change
        # Use Sobel filter to detect edges
        edge_x = ndimage.sobel(gradient, axis=1)
        edge_y = ndimage.sobel(gradient, axis=0)
        edge_magnitude = np.sqrt(edge_x**2 + edge_y**2)

        # Normalize edge magnitude
        if np.max(edge_magnitude) > 0:
            edge_magnitude = edge_magnitude / np.max(edge_magnitude)

        # Apply contrast enhancement based on edge strength
        # Areas with high edge magnitude get more contrast adjustment
        contrast_factor = 1.0 + (edge_contrast - 1.0) * edge_magnitude

        # Apply contrast: new_value = (old_value - 0.5) * contrast + 0.5
        enhanced = (gradient - 0.5) * contrast_factor + 0.5

        # Clamp values to [0, 1] range
        enhanced = np.clip(enhanced, 0.0, 1.0)

        return enhanced
    return gradient

def apply_vertical_flip(image_array, flip_vertical):
    """Apply vertical flip to the image array if requested"""
    if flip_vertical:
        return np.flipud(image_array)
    return image_array

def check_uv_map(obj, uv_index):
    """Check if UV map exists at given index"""
    if not obj.data.uv_layers:
        raise ValueError(f"Object '{obj.name}' has no UV maps")

    if uv_index >= len(obj.data.uv_layers):
        raise ValueError(f"UV index {uv_index} out of range. Object has {len(obj.data.uv_layers)} UV maps")

    return obj.data.uv_layers[uv_index]

def create_combined_uv_island_mask(objects, uv_index, resolution):
    """Create a combined mask of UV islands from multiple objects"""
    combined_mask = np.zeros((resolution, resolution), dtype=np.float32)

    for obj in objects:
        if obj.type != 'MESH':
            continue

        try:
            check_uv_map(obj, uv_index)
        except ValueError as e:
            print(f"Skipping {obj.name}: {e}")
            continue

        bm = bmesh.new()
        bm.from_mesh(obj.data)

        if uv_index >= len(bm.loops.layers.uv):
            bm.free()
            continue

        uv_layer = bm.loops.layers.uv[uv_index]

        # Create mask for this object
        obj_mask = create_uv_island_mask_single(bm, uv_layer, resolution)

        # Combine with the main mask (using maximum to preserve all islands)
        combined_mask = np.maximum(combined_mask, obj_mask)

        bm.free()

    return combined_mask

def create_uv_island_mask_single(bm, uv_layer, resolution):
    """Create a mask of UV islands for a single bmesh object"""
    mask = np.zeros((resolution, resolution), dtype=np.float32)

    for face in bm.faces:
        uvs = [l[uv_layer].uv for l in face.loops]

        coords = []
        for uv in uvs:
            x = int(uv.x * (resolution - 1))
            y = int(uv.y * (resolution - 1))
            x = max(0, min(x, resolution - 1))
            y = max(0, min(y, resolution - 1))
            coords.append((x, y))

        for i in range(1, len(coords) - 1):
            p1 = coords[0]
            p2 = coords[i]
            p3 = coords[i + 1]
            fill_triangle(mask, p1, p2, p3)

    return mask

def fill_triangle(mask, p1, p2, p3):
    """Fill a triangle in the mask using a scanline approach"""
    min_x = max(0, min(p1[0], p2[0], p3[0]))
    max_x = min(mask.shape[1] - 1, max(p1[0], p2[0], p3[0]))
    min_y = max(0, min(p1[1], p2[1], p3[1]))
    max_y = min(mask.shape[0] - 1, max(p1[1], p2[1], p3[1]))

    def edge(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    area = edge(p1, p2, p3)
    if area == 0:
        return

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            b1 = edge(p2, p3, (x, y)) / area
            b2 = edge(p3, p1, (x, y)) / area
            b3 = edge(p1, p2, (x, y)) / area

            if b1 >= 0 and b2 >= 0 and b3 >= 0:
                mask[y, x] = 1

def process_gradient(distance, config):
    """Process the distance field according to config settings"""
    max_dist = np.max(distance)
    if max_dist > 0:
        normalized_dist = distance / max_dist
        gradient = config.falloff_function(normalized_dist)

        if config.smoothness != 1.0:
            gradient = np.power(gradient, config.smoothness)

        if config.gamma_correction != 1.0:
            gradient = np.power(gradient, 1/config.gamma_correction)

        # Apply edge contrast enhancement
        gradient = apply_edge_contrast(gradient, config.edge_contrast)

        # Apply blur
        gradient = apply_blur(gradient, config.blur_amount)

        return gradient
    return np.zeros_like(distance)

def generate_combined_uv_border_distance(objects, config=None):
    """Generate a combined UV distance map from multiple objects"""
    if config is None:
        config = TextureGenConfig()

    # Filter out non-mesh objects
    mesh_objects = [obj for obj in objects if obj.type == 'MESH']
    if not mesh_objects:
        print("No mesh objects found in selection")
        return None

    print(f"Processing {len(mesh_objects)} mesh objects: {[obj.name for obj in mesh_objects]}")

    # Create combined UV island mask
    combined_mask = create_combined_uv_island_mask(mesh_objects, config.uv_index, config.resolution)

    if np.max(combined_mask) == 0:
        print("No valid UV data found in selected objects")
        return None

    # Calculate border mask
    kernel = np.ones((3, 3)) if config.border_thickness > 1 else None
    eroded = ndimage.binary_erosion(combined_mask, structure=kernel,
                                  iterations=config.border_thickness)
    border_mask = combined_mask - eroded

    # Calculate distance field
    distance = ndimage.distance_transform_edt(combined_mask)

    # Process gradient according to config
    gradient = process_gradient(distance, config)

    # Ensure borders are black
    gradient[border_mask > 0] = 0

    # Apply vertical flip if requested
    gradient = apply_vertical_flip(gradient, config.flip_vertical)

    # Create combined name
    object_names = [obj.name for obj in mesh_objects]
    if len(object_names) <= 3:
        combined_name = "_".join(object_names)
    else:
        combined_name = f"{object_names[0]}_and_{len(object_names)-1}_others"

    image_name = f"{combined_name}_combined_uv_distance"

    # Remove existing image if it exists
    if image_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[image_name])

    blender_img = bpy.data.images.new(
        name=image_name,
        width=config.resolution,
        height=config.resolution,
        alpha=not config.black_background
    )

    # Convert to Blender's format
    pixels = gradient.ravel()
    if config.black_background:
        pixels_rgba = np.zeros(len(pixels) * 4)
        pixels_rgba[::4] = pixels
        pixels_rgba[1::4] = pixels
        pixels_rgba[2::4] = pixels
        pixels_rgba[3::4] = 1.0
    else:
        pixels_rgba = np.repeat(pixels, 4)

    blender_img.pixels = pixels_rgba

    # Save to texture folder if path is not specified
    if not config.save_path:
        texture_folder = get_texture_folder()
        if texture_folder:
            config.save_path = os.path.join(texture_folder, f"{image_name}.png")

    if config.save_path:
        blender_img.file_format = 'PNG'
        blender_img.save_render(config.save_path)

    return blender_img

def main():
    # Get selected objects
    selected_objects = bpy.context.selected_objects
    if not selected_objects:
        print("No objects selected")
        return

    # Create and customize config
    config = TextureGenConfig()
    # config.resolution = 1024*4
    # config.falloff_rate = 3.0
    # config.smoothness = 1.2
    # config.border_thickness = 1
    # config.gamma_correction = 2.2
    # config.blur_amount = 0.0
    # config.edge_contrast = 1.0
    # config.flip_vertical = False
    # config.uv_index = 0
    # config.black_background = True

    # Process all selected objects as one combined image
    print(f"Combining {len(selected_objects)} selected objects into one UV distance map")
    image = generate_combined_uv_border_distance(selected_objects, config)

    if image:
        print(f"Generated combined UV distance map: {image.name}")
        if config.save_path:
            print(f"Saved to: {config.save_path}")
        print("Settings used:")
        print(f"- Resolution: {config.resolution}")
        print(f"- Falloff rate: {config.falloff_rate}")
        print(f"- Smoothness: {config.smoothness}")
        print(f"- Border thickness: {config.border_thickness}")
        print(f"- Gamma correction: {config.gamma_correction}")
        print(f"- Blur amount: {config.blur_amount}")
        print(f"- Edge contrast: {config.edge_contrast}")
        print(f"- Flip vertical: {config.flip_vertical}")
        print(f"- UV index: {config.uv_index}")
    else:
        print("Failed to generate combined UV distance map")

if __name__ == "__main__":
    main()
