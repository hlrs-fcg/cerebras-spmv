"""PE count calculation for SpMV on WSE-3."""


def calculate_max_pes_useful(matrix_rows: int, max_width: int = 758, max_height: int = 1171) -> int:
    """Calculate maximum useful PEs (limited by matrix rows and fabric dimensions)."""
    max_fabric_pes = max_width * max_height
    max_useful = min(matrix_rows, max_fabric_pes)

    nice_numbers = [1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 4096, 8192]

    for n in reversed(nice_numbers):
        if n <= max_useful:
            return n

    return max_useful
