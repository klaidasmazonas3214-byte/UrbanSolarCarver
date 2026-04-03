"""Tests for carve_above_columns post-processing."""
import numpy as np
import pytest


def _col_grid(columns: dict, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Build small 3D grids from per-column specs.

    Parameters
    ----------
    columns : dict
        Mapping of (x, y) -> str where each character is '#' (occupied+kept),
        'x' (occupied+carved), or '.' (empty in original envelope).
        String is ordered top-to-bottom (index -1 is z=0, index 0 is z=height-1).
    height : int
        Number of z-levels.

    Returns
    -------
    mask : (X, Y, Z) bool array — True = keep, False = carved
    voxel_grid : (X, Y, Z) bool array — True = occupied in original envelope
    """
    max_x = max(x for x, y in columns) + 1
    max_y = max(y for x, y in columns) + 1
    mask = np.ones((max_x, max_y, height), dtype=bool)
    voxel_grid = np.ones((max_x, max_y, height), dtype=bool)
    for (x, y), col_str in columns.items():
        assert len(col_str) == height, f"Column ({x},{y}) length {len(col_str)} != height {height}"
        for i, ch in enumerate(col_str):
            z = height - 1 - i  # top-to-bottom string -> z index
            if ch == 'x':
                mask[x, y, z] = False       # carved
                voxel_grid[x, y, z] = True   # was occupied
            elif ch == '.':
                mask[x, y, z] = True         # not carved (empty anyway)
                voxel_grid[x, y, z] = False  # not occupied
            elif ch == '#':
                mask[x, y, z] = True         # kept
                voxel_grid[x, y, z] = True   # occupied
            else:
                raise ValueError(f"Unknown char '{ch}' in column ({x},{y})")
    return mask, voxel_grid


class TestCarveAboveColumns:
    """Tests for the carve_above_columns function."""

    def test_spec_example_min2(self):
        """The exact example from the design spec with min_consecutive=2.

        User's original grid (top-to-bottom, reading each column):
            ABC       A: ##xxx#   B: ###x##   C: #xx###
            ###       (top-to-bottom strings)
            ##x
            x#x       A bottom-up: #,x,x,x,#,# — 3 consec >= 2 → carve 2 above
            xx#       B bottom-up: #,#,x,#,#,# — 1 consec < 2  → patched
            x##       C bottom-up: #,#,#,x,x,# — 2 consec >= 2 → carve 1 above
            ###
        """
        from urbansolarcarver.carving import carve_above_columns

        mask, vg = _col_grid({
            (0, 0): "##xxx#",  # A
            (1, 0): "###x##",  # B
            (2, 0): "#xx###",  # C
        }, height=6)

        result = carve_above_columns(mask, vg, min_consecutive=2)

        # A: run of 3 at z1-z3, carve z4,z5
        assert result[0, 0, 4] == False  # was #, now carved
        assert result[0, 0, 5] == False  # was #, now carved
        assert result[0, 0, 0] == True   # bottom # unchanged

        # B: run of 1 at z2, below threshold → patched back to kept
        assert result[1, 0, 2] == True   # x patched back to #
        assert result[1, 0, 3] == True
        assert result[1, 0, 4] == True
        assert result[1, 0, 5] == True

        # C: run of 2 at z3-z4, carve z5
        assert result[2, 0, 5] == False  # was #, now carved
        assert result[2, 0, 0] == True   # bottom # unchanged

    def test_min1_any_carved_triggers(self):
        """With min_consecutive=1, a single carved voxel triggers carve-above."""
        from urbansolarcarver.carving import carve_above_columns

        # Single column bottom-up: #, x, #, #
        mask, vg = _col_grid({(0, 0): "##x#"}, height=4)
        result = carve_above_columns(mask, vg, min_consecutive=1)

        assert result[0, 0, 0] == True   # bottom # kept
        assert result[0, 0, 1] == False  # x stays carved
        assert result[0, 0, 2] == False  # # above → carved
        assert result[0, 0, 3] == False  # # above → carved

    def test_no_carved_voxels_unchanged(self):
        """Column with no carved voxels is untouched."""
        from urbansolarcarver.carving import carve_above_columns

        mask, vg = _col_grid({(0, 0): "####"}, height=4)
        result = carve_above_columns(mask, vg, min_consecutive=1)

        assert result[0, 0, :].all()

    def test_only_carves_occupied_voxels(self):
        """Empty (non-occupied) voxels above a carved run are left alone."""
        from urbansolarcarver.carving import carve_above_columns

        # bottom-up: #, x, x, ., #  ('.' = not occupied)
        mask, vg = _col_grid({(0, 0): "#.xx#"}, height=5)
        result = carve_above_columns(mask, vg, min_consecutive=2)

        assert result[0, 0, 0] == True   # bottom #
        assert result[0, 0, 1] == False  # x
        assert result[0, 0, 2] == False  # x
        assert result[0, 0, 3] == True   # . — not occupied, unchanged
        assert result[0, 0, 4] == False  # # — occupied, carved

    def test_does_not_mutate_input(self):
        """Input mask array must not be modified."""
        from urbansolarcarver.carving import carve_above_columns

        mask, vg = _col_grid({(0, 0): "##xx#"}, height=5)
        original = mask.copy()
        carve_above_columns(mask, vg, min_consecutive=2)
        np.testing.assert_array_equal(mask, original)

    def test_all_carved_column(self):
        """Column already fully carved stays fully carved."""
        from urbansolarcarver.carving import carve_above_columns

        mask, vg = _col_grid({(0, 0): "xxxx"}, height=4)
        result = carve_above_columns(mask, vg, min_consecutive=1)
        assert not result[0, 0, :].any()

    def test_short_runs_patched(self):
        """Carved runs shorter than min_consecutive are patched back."""
        from urbansolarcarver.carving import carve_above_columns

        # bottom-up: #, #, x, #, #  — single carved voxel, min=2
        mask, vg = _col_grid({(0, 0): "##x##"}, height=5)
        result = carve_above_columns(mask, vg, min_consecutive=2)

        # The lone x should be patched back to #
        assert result[0, 0, :].all()  # all kept

    def test_short_run_patched_before_long_run_triggers(self):
        """A short run below a qualifying run is patched, then carve-above fires."""
        from urbansolarcarver.carving import carve_above_columns

        # bottom-up: #, x, #, x, x, #, #  — run of 1 at z1, run of 2 at z3-z4
        mask, vg = _col_grid({(0, 0): "##xx#x#"}, height=7)
        result = carve_above_columns(mask, vg, min_consecutive=2)

        assert result[0, 0, 0] == True   # bottom #
        assert result[0, 0, 1] == True   # x patched (run of 1 < 2)
        assert result[0, 0, 2] == True   # #
        assert result[0, 0, 3] == False  # x (part of qualifying run)
        assert result[0, 0, 4] == False  # x (part of qualifying run)
        assert result[0, 0, 5] == False  # # carved above
        assert result[0, 0, 6] == False  # # carved above

    def test_multiple_columns_independent(self):
        """Each column is processed independently."""
        from urbansolarcarver.carving import carve_above_columns

        mask, vg = _col_grid({
            (0, 0): "##xx#",  # 2 consecutive → carve above
            (1, 0): "#x###",  # 1 consecutive → patched (min=2)
        }, height=5)
        result = carve_above_columns(mask, vg, min_consecutive=2)

        # Column (0,0): bottom-up #,x,x,#,# → #,x,x,carved,carved
        assert result[0, 0, 3] == False
        assert result[0, 0, 4] == False

        # Column (1,0): bottom-up #,#,#,x,# → x patched back to #
        assert result[1, 0, 3] == True   # was x, now patched
        assert result[1, 0, 4] == True
