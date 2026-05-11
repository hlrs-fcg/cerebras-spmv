"""PE grid calculation and fabric constraint validation."""

from utils.logging import logger


def calculate_pe_grid(num_pes: int,
                     max_width: int = 758,
                     max_height: int = 1171) -> tuple:
    """Calculate optimal 2D PE grid within fabric constraints.

    Prefers square-ish grids for better locality. Returns (width, height).
    """
    if num_pes == 1:
        return (1, 1)

    if num_pes > max_width * max_height:
        raise ValueError(
            f"Cannot fit {num_pes} PEs in fabric "
            f"(max: {max_width}×{max_height} = {max_width * max_height})"
        )

    factorizations = []
    for w in range(1, min(num_pes, max_width) + 1):
        if num_pes % w == 0:
            h = num_pes // w
            if h <= max_height:
                aspect_ratio = max(w, h) / min(w, h)
                factorizations.append((w, h, aspect_ratio))

    if not factorizations:
        raise ValueError(
            f"Cannot fit {num_pes} PEs in fabric constraints "
            f"(max_width={max_width}, max_height={max_height})"
        )

    factorizations.sort(key=lambda x: x[2])
    width, height, _ = factorizations[0]

    return (width, height)


def print_grid_info(width: int, height: int, num_pes: int,
                   fabric_dims: tuple = (762, 1172),
                   fabric_offsets: tuple = (4, 1)):
    logger.info(f"PE Grid: {width}×{height} ({num_pes} PEs)")


def validate_grid_fits_fabric(width: int, height: int,
                              fabric_dims: tuple = (762, 1172),
                              fabric_offsets: tuple = (4, 1)) -> bool:
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
