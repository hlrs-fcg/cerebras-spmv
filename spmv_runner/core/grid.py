"""
PE Grid Calculation and Validation

Handles calculation of optimal 2D PE grids and validation against fabric constraints.
"""

import logging

logger = logging.getLogger(__name__)


def calculate_pe_grid(num_pes: int,
                     max_width: int = 758,
                     max_height: int = 1171) -> tuple:
    """
    Calculate optimal 2D PE grid that fits within fabric constraints.
    Prefers more square-like grids for better locality.

    Args:
        num_pes: Total number of PEs
        max_width: Maximum fabric width (default: 758 for WSE-3)
        max_height: Maximum fabric height (default: 1171 for WSE-3)

    Returns:
        (width, height) tuple

    Raises:
        ValueError: If num_pes doesn't fit in fabric
    """
    if num_pes == 1:
        return (1, 1)

    if num_pes > max_width * max_height:
        raise ValueError(
            f"Cannot fit {num_pes} PEs in fabric "
            f"(max: {max_width}×{max_height} = {max_width * max_height})"
        )

    # Find all valid factorizations
    factorizations = []
    for w in range(1, min(num_pes, max_width) + 1):
        if num_pes % w == 0:
            h = num_pes // w
            if h <= max_height:
                # Score: prefer square-ish grids
                # Ratio closer to 1.0 is better
                aspect_ratio = max(w, h) / min(w, h)
                factorizations.append((w, h, aspect_ratio))

    if not factorizations:
        raise ValueError(
            f"Cannot fit {num_pes} PEs in fabric constraints "
            f"(max_width={max_width}, max_height={max_height})"
        )

    # Choose factorization with best (lowest) aspect ratio
    factorizations.sort(key=lambda x: x[2])
    width, height, _ = factorizations[0]

    return (width, height)


def print_grid_info(width: int, height: int, num_pes: int,
                   fabric_dims: tuple = (762, 1172),
                   fabric_offsets: tuple = (4, 1)):
    """Print visual representation of PE grid with fabric info."""
    logger.info(f"PE Grid: {width}×{height} ({num_pes} PEs)")
    logger.debug(f"Fabric dimensions: {fabric_dims[0]}×{fabric_dims[1]}")
    logger.debug(f"Fabric offsets: {fabric_offsets}")
    logger.debug(f"Usable area: {fabric_dims[0]-fabric_offsets[0]}×{fabric_dims[1]-fabric_offsets[1]}")
    logger.debug(f"Grid utilization: {(width*height)/(fabric_dims[0]*fabric_dims[1])*100:.2f}%")

    # Visual grid (limit display for large grids)
    if width <= 20 and height <= 20:
        for y in range(height):
            row = ""
            for x in range(width):
                pe_id = y * width + x
                row += f"[PE{pe_id:3d}] "
            logger.debug(row)
    else:
        logger.debug(f"Grid corners: PE 0, PE {width-1}, PE {(height-1)*width}, PE {height*width-1}")


def validate_grid_fits_fabric(width: int, height: int,
                              fabric_dims: tuple = (762, 1172),
                              fabric_offsets: tuple = (4, 1)) -> bool:
    """Check if PE grid fits within fabric constraints."""
    max_width = fabric_dims[0] - fabric_offsets[0]
    max_height = fabric_dims[1] - fabric_offsets[1]

    if width > max_width or height > max_height:
        logger.error("="*60)
        logger.error("FABRIC CONSTRAINT VIOLATION!")
        logger.error("="*60)
        logger.error(f"Requested grid: {width}×{height}")
        logger.error(f"Maximum allowed: {max_width}×{max_height}")
        logger.error(f"Fabric dims: {fabric_dims}, offsets: {fabric_offsets}")
        logger.error("="*60)
        return False

    return True
