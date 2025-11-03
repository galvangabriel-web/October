"""
Track Map Images Module
========================

Utilities for loading and displaying track map images.

Images are generated from official track sector maps (PDFs) using:
    python convert_track_maps_to_images.py

Available resolutions:
    - Standard: 150 DPI (1275x1650 or 1650x1275)
    - HD: 300 DPI (2550x3300 or 3300x2550)

Usage:
    from src.track_data.track_images import TrackImageLoader

    loader = TrackImageLoader()
    img = loader.get_track_image('barber-motorsports-park', page=1, hd=True)
    img.show()
"""

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class TrackImageLoader:
    """Load and display track map images"""

    def __init__(self, images_dir: Optional[Path] = None):
        """
        Initialize track image loader

        Args:
            images_dir: Directory containing track images
                       Default: track_maps/images/
        """
        if images_dir is None:
            self.images_dir = Path("track_maps/images")
        else:
            self.images_dir = Path(images_dir)

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}\n"
                "Run: python convert_track_maps_to_images.py"
            )

    def get_track_image(
        self,
        track_name: str,
        page: int = 1,
        hd: bool = False
    ) -> Optional[Image.Image]:
        """
        Load track map image

        Args:
            track_name: Track identifier (e.g., 'barber-motorsports-park')
            page: Page number (1-indexed)
            hd: Use high-definition version (300 DPI vs 150 DPI)

        Returns:
            PIL Image object or None if not found

        Example:
            >>> loader = TrackImageLoader()
            >>> img = loader.get_track_image('circuit-of-the-americas', page=1, hd=True)
            >>> print(f"Size: {img.size}")
            Size: (2550, 3300)
        """
        suffix = "_hd" if hd else ""
        filename = f"{track_name}_page_{page}{suffix}.png"
        image_path = self.images_dir / filename

        if not image_path.exists():
            print(f"Image not found: {filename}")
            return None

        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            return None

    def get_all_track_images(
        self,
        track_name: str,
        hd: bool = False
    ) -> list:
        """
        Load all pages for a track

        Args:
            track_name: Track identifier
            hd: Use high-definition versions

        Returns:
            List of PIL Image objects

        Example:
            >>> loader = TrackImageLoader()
            >>> images = loader.get_all_track_images('barber-motorsports-park')
            >>> print(f"Loaded {len(images)} pages")
        """
        images = []
        page = 1

        while True:
            img = self.get_track_image(track_name, page=page, hd=hd)
            if img is None:
                break
            images.append(img)
            page += 1

        return images

    def display_track_map(
        self,
        track_name: str,
        page: int = 1,
        hd: bool = False,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Display track map using matplotlib

        Args:
            track_name: Track identifier
            page: Page number
            hd: Use HD version
            figsize: Figure size (width, height)

        Example:
            >>> loader = TrackImageLoader()
            >>> loader.display_track_map('circuit-of-the-americas', page=1)
        """
        img = self.get_track_image(track_name, page=page, hd=hd)

        if img is None:
            print(f"Cannot display: image not found")
            return

        # Get track metadata for title
        try:
            from src.track_data.track_metadata import get_track_metadata
            metadata = get_track_metadata(track_name)
            title = metadata.get('name', track_name) if metadata else track_name
        except ImportError:
            title = track_name

        # Display
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{title} - Track Map (Page {page})", fontsize=14, pad=10)
        plt.tight_layout()
        plt.show()

    def list_available_images(self) -> dict:
        """
        List all available track images

        Returns:
            Dictionary mapping track names to list of available pages

        Example:
            >>> loader = TrackImageLoader()
            >>> images = loader.list_available_images()
            >>> for track, pages in images.items():
            ...     print(f"{track}: {len(pages)} pages")
        """
        available = {}

        # Scan directory for images
        for image_path in sorted(self.images_dir.glob("*.png")):
            # Skip index file
            if image_path.name == "IMAGE_INDEX.txt":
                continue

            # Parse filename: {track}_page_{n}[_hd].png
            name_parts = image_path.stem.split('_page_')
            if len(name_parts) != 2:
                continue

            track_name = name_parts[0]
            page_info = name_parts[1]

            # Extract page number
            page_num = int(page_info.replace('_hd', ''))
            is_hd = '_hd' in page_info

            if track_name not in available:
                available[track_name] = {'standard': [], 'hd': []}

            if is_hd:
                available[track_name]['hd'].append(page_num)
            else:
                available[track_name]['standard'].append(page_num)

        return available

    def save_with_overlay(
        self,
        track_name: str,
        page: int,
        overlay_func,
        output_path: Path,
        hd: bool = True
    ):
        """
        Save track map with custom overlay

        Args:
            track_name: Track identifier
            page: Page number
            overlay_func: Function that takes (fig, ax) and adds overlays
            output_path: Where to save the result
            hd: Use HD version

        Example:
            >>> def add_telemetry_overlay(fig, ax):
            ...     # Add telemetry visualization
            ...     ax.plot([100, 200], [300, 400], 'r-', linewidth=2)
            >>>
            >>> loader = TrackImageLoader()
            >>> loader.save_with_overlay(
            ...     'barber-motorsports-park',
            ...     page=1,
            ...     overlay_func=add_telemetry_overlay,
            ...     output_path=Path('output.png')
            ... )
        """
        img = self.get_track_image(track_name, page=page, hd=hd)

        if img is None:
            print(f"Cannot create overlay: image not found")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(img)
        ax.axis('off')

        # Apply user overlay
        overlay_func(fig, ax)

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved overlay image: {output_path}")


# ============================================================================
# Convenience Functions
# ============================================================================

def show_track_map(track_name: str, page: int = 1, hd: bool = False):
    """
    Quick function to display a track map

    Args:
        track_name: Track identifier
        page: Page number
        hd: Use HD version

    Example:
        >>> from src.track_data.track_images import show_track_map
        >>> show_track_map('circuit-of-the-americas', page=1, hd=True)
    """
    loader = TrackImageLoader()
    loader.display_track_map(track_name, page=page, hd=hd)


def get_track_image_path(track_name: str, page: int = 1, hd: bool = False) -> Path:
    """
    Get path to track image file

    Args:
        track_name: Track identifier
        page: Page number
        hd: Use HD version

    Returns:
        Path to image file

    Example:
        >>> from src.track_data.track_images import get_track_image_path
        >>> path = get_track_image_path('barber-motorsports-park', page=1, hd=True)
        >>> print(path)
        track_maps/images/barber-motorsports-park_page_1_hd.png
    """
    images_dir = Path("track_maps/images")
    suffix = "_hd" if hd else ""
    filename = f"{track_name}_page_{page}{suffix}.png"
    return images_dir / filename


def list_all_track_images() -> dict:
    """
    List all available track images

    Returns:
        Dictionary of available images by track

    Example:
        >>> from src.track_data.track_images import list_all_track_images
        >>> images = list_all_track_images()
        >>> for track, info in images.items():
        ...     print(f"{track}: {len(info['standard'])} standard, {len(info['hd'])} HD")
    """
    loader = TrackImageLoader()
    return loader.list_available_images()


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("Track Image Loader - Demo")
    print("=" * 60)

    loader = TrackImageLoader()

    # List available images
    print("\nAvailable track images:")
    available = loader.list_available_images()

    for track_name, versions in sorted(available.items()):
        std_pages = len(versions['standard'])
        hd_pages = len(versions['hd'])
        print(f"  {track_name}:")
        print(f"    Standard (150 DPI): {std_pages} pages")
        print(f"    HD (300 DPI): {hd_pages} pages")

    # Load and display sample image
    print("\n" + "=" * 60)
    print("Loading sample image...")

    sample_img = loader.get_track_image('barber-motorsports-park', page=1, hd=False)
    if sample_img:
        print(f"Loaded: barber-motorsports-park_page_1.png")
        print(f"Size: {sample_img.size}")
        print(f"Format: {sample_img.format}")
        print(f"Mode: {sample_img.mode}")

        # Uncomment to display
        # loader.display_track_map('barber-motorsports-park', page=1)

    print("\n" + "=" * 60)
    print("Demo complete")
