import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris

from tetris_agent import TargetPlacement, TargetSpace, TetrisAdapter
from tetris_agent.target_space import (
    env_origin_to_target, occupied_offset, orientation_matrices,
    target_to_env_origin, unique_orientations,
)


@pytest.fixture
def setup():
    env = Tetris(gravity=False)
    env.reset(seed=42)
    adapter = TetrisAdapter(env)
    yield env, adapter, TargetSpace(adapter.geometry)
    env.close()


def test_all_action_ids_round_trip(setup):
    _, _, space = setup
    assert space.n == 800
    for action_id in range(space.n):
        assert space.encode(space.decode(action_id)) == action_id


def test_action_slots_cover_only_playable_bbox_coordinates(setup):
    _, adapter, space = setup
    geometry = adapter.geometry
    targets = {space.decode(i) for i in range(space.n)}
    expected = {
        TargetPlacement(x, y, k)
        for y in range(geometry.height)
        for x in range(geometry.padding, geometry.padding + geometry.width)
        for k in range(4)
    }
    assert targets == expected
    assert space.decode(0) == TargetPlacement(4, 0, 0)
    assert space.decode(space.n - 1) == TargetPlacement(13, 19, 3)
    rows, cols = geometry.board_shape
    for y in range(rows):
        for x in range(cols):
            if y >= geometry.height or not geometry.padding <= x < geometry.padding + geometry.width:
                for k in range(4):
                    with pytest.raises(ValueError):
                        space.encode(TargetPlacement(x, y, k))


def test_right_edge_slot_exists_but_piece_overflow_is_masked(setup):
    env, adapter, space = setup
    target = TargetPlacement(13, 18, 0)
    action_id = space.encode(target)
    assert space.decode(action_id) == target
    assert not space.action_mask(adapter.read_board(), env.tetrominoes[1].matrix)[action_id]


def test_geometric_i_placement_with_padding_origin_has_action_id(setup):
    env, adapter, space = setup
    env_x, env_y = adapter.geometry.padding - 1, 16
    rotated = env.rotate(env.tetrominoes[0])
    assert not env.collision(rotated, env_x, env_y)
    assert env.collision(rotated, env_x, env_y + 1)
    matrix = env.tetrominoes[0].matrix
    target = env_origin_to_target(env_x, env_y, 1, matrix)
    assert target == TargetPlacement(4, 16, 1)
    assert space.encode(target) == 641
    assert space.decode(641) == target
    assert target_to_env_origin(target, matrix) == (3, 16)
    assert space.is_valid(adapter.read_board(), matrix, target)
    assert space.action_mask(adapter.read_board(), matrix)[641]


def test_mapping_is_fixed_across_states_and_pieces(setup):
    env, adapter, space = setup
    other = TargetSpace(adapter.geometry)
    target = TargetPlacement(7, 18, 1)
    expected = ((18 * 10) + (7 - 4)) * 4 + 1
    assert space.encode(target) == other.encode(target) == expected
    env.step(env.actions.rotate_clockwise)
    env.step(env.actions.hard_drop)
    assert space.encode(target) == expected
    assert space.decode(expected) == target
    assert space.n == other.n


@pytest.mark.parametrize("action_id", [-1, 800])
def test_outside_action_id_rejected(setup, action_id):
    with pytest.raises(ValueError):
        setup[2].decode(action_id)


@pytest.mark.parametrize("value", [1.2, True, "1"])
def test_non_integer_values_rejected(setup, value):
    with pytest.raises(TypeError):
        setup[2].decode(value)
    with pytest.raises(TypeError):
        TargetPlacement(value, 0, 0)


@pytest.mark.parametrize("target", [
    TargetPlacement(-1, 18, 0), TargetPlacement(18, 18, 0),
    TargetPlacement(3, 18, 0), TargetPlacement(13, 18, 0),
    TargetPlacement(4, -1, 0), TargetPlacement(4, 20, 0),
    TargetPlacement(4, 24, 0), TargetPlacement(4, 18, 4),
    TargetPlacement(4, 18, -1),
])
def test_board_boundary(setup, target):
    env, adapter, space = setup
    o_matrix = env.tetrominoes[1].matrix
    assert not space.is_valid(adapter.read_board(), o_matrix, target)


def test_locked_collision_floating_and_resting(setup):
    env, adapter, space = setup
    board = adapter.read_board()
    matrix = env.tetrominoes[1].matrix
    target = TargetPlacement(adapter.geometry.padding, adapter.geometry.height - 2, 0)
    assert space.is_valid(board, matrix, target)
    assert not space.is_valid(board, matrix, TargetPlacement(target.x, target.y - 1, 0))
    board[target.y, target.x] = 2
    assert not space.is_valid(board, matrix, target)


def test_resting_on_locked_blocks(setup):
    env, adapter, space = setup
    board = adapter.read_board()
    board[10, 7] = 2
    target = TargetPlacement(7, 8, 0)
    assert space.is_valid(board, env.tetrominoes[1].matrix, target)


@pytest.mark.parametrize("index, count", [(0, 2), (1, 1), (2, 4), (3, 2), (4, 2), (5, 4), (6, 4)])
def test_duplicate_shapes_are_masked(setup, index, count):
    env, adapter, space = setup
    matrix = env.tetrominoes[index].matrix
    unique = unique_orientations(matrix)
    assert len(unique) == count
    mask = space.action_mask(adapter.read_board(), matrix)
    for k in range(4):
        assert mask[k::4].any() == (k in unique)
    # Full padded matrices differ for the translated half-turn duplicates.
    if index in (0, 3, 4):
        assert not np.array_equal(matrix, np.rot90(matrix, 2))
        assert unique == (0, 1)


@pytest.mark.parametrize("index", range(7))
@pytest.mark.parametrize("with_blocks", [False, True])
def test_mask_agrees_with_environment_collision(setup, index, with_blocks):
    env, adapter, space = setup
    if with_blocks:
        # Uneven surface and overhang; no path search is involved.
        env.board[19, 4:9] = 2
        env.board[17:20, 11] = 3
        env.board[12, 6:10] = 4
    matrix = env.tetrominoes[index].matrix
    board = adapter.read_board()
    mask = space.action_mask(board, matrix)
    orientations = orientation_matrices(matrix)
    unique = unique_orientations(matrix)
    for action_id in range(space.n):
        target = space.decode(action_id)
        expected = False
        rotated = env.rotate(env.tetrominoes[index])
        rotated.matrix = orientations[target.orientation]
        h, w = rotated.matrix.shape
        rows, cols = np.nonzero(rotated.matrix > 0)
        env_x = target.x - int(cols.min())
        env_y = target.y - int(rows.min())
        if (target.orientation in unique
                and env_x >= 0 and env_y >= 0
                and env_x + w <= board.shape[1]
                and env_y + h + 1 <= board.shape[0]):
            expected = (not env.collision(rotated, env_x, env_y)
                        and env.collision(rotated, env_x, env_y + 1))
        assert bool(mask[action_id]) == bool(expected)
        assert space.is_valid(board, matrix, target) == bool(mask[action_id])


def test_sealed_cavity_is_valid_without_reachability_filter(setup):
    env, adapter, space = setup
    board = adapter.read_board()
    # A solid wall-to-wall roof seals the bottom cavity from spawn.
    board[17, 4:14] = 2
    target = TargetPlacement(7, 18, 0)
    matrix = env.tetrominoes[1].matrix
    assert space.is_valid(board, matrix, target)
    assert space.action_mask(board, matrix)[space.encode(target)]


def test_dangerous_top_placement_is_still_valid(setup):
    env, adapter, space = setup
    board = adapter.read_board()
    board[2, 4:14] = 2
    assert space.is_valid(board, env.tetrominoes[1].matrix, TargetPlacement(8, 0, 0))


def test_real_reset_state_and_no_side_effects(setup):
    _, adapter, space = setup
    before = adapter.read()
    mask = space.action_mask(before.board, before.current_piece.spawn_matrix)
    assert mask.shape == (space.n,)
    assert mask.dtype == np.bool_
    ids = np.flatnonzero(mask)
    assert len(ids) == 17  # S: 8 horizontal and 9 vertical placements.
    targets = [space.decode(i) for i in ids]
    assert all(space.is_valid(before.board, before.current_piece.spawn_matrix, t)
               for t in targets)
    assert all(space.decode(space.encode(t)) == t for t in targets)
    after = adapter.read()
    np.testing.assert_array_equal(before.board, after.board)
    np.testing.assert_array_equal(before.current_piece.matrix, after.current_piece.matrix)
    assert before.visible_queue == after.visible_queue
    assert (before.current_piece.x, before.current_piece.y) == (after.current_piece.x, after.current_piece.y)


def test_rotation_does_not_rebase_target_orientations(setup):
    env, adapter, space = setup
    before = adapter.read()
    mask = space.action_mask(before.board, before.current_piece.spawn_matrix)
    env.step(env.actions.rotate_clockwise)
    after = adapter.read()
    assert after.current_piece.orientation == 1
    np.testing.assert_array_equal(
        mask, space.action_mask(after.board, after.current_piece.spawn_matrix)
    )


def test_board_shape_mismatch_rejected(setup):
    env, _, space = setup
    with pytest.raises(ValueError, match="shape"):
        space.action_mask(np.zeros((20, 10)), env.tetrominoes[1].matrix)


def test_non_default_geometry():
    env = Tetris(width=6, height=12, gravity=False)
    try:
        env.reset(seed=42)
        adapter = TetrisAdapter(env)
        state = adapter.read()
        space = TargetSpace(adapter.geometry)
        assert space.n == 12 * 6 * 4
        assert space.decode(0) == TargetPlacement(env.padding, 0, 0)
        assert space.decode(space.n - 1) == TargetPlacement(env.padding + env.width - 1, env.height - 1, 3)
        assert space.action_mask(state.board, state.current_piece.spawn_matrix).sum() == 9
        assert space.encode(space.decode(space.n - 1)) == space.n - 1
    finally:
        env.close()


def reference_resting_origins(env, matrix):
    """Independent brute-force geometry oracle over raw origins, not action IDs.

    Use the real locked board and actual piece cells. Include origins above and
    beside the board so blank matrix borders cannot hide a valid placement.
    No TargetSpace candidate generation, conversions, or collision helpers.
    """
    cells = [tuple(map(int, cell)) for cell in np.argwhere(matrix > 0)]

    def collision(x, y):
        for row, col in cells:
            bx, by = x + col, y + row
            if not (env.padding <= bx < env.padding + env.width and 0 <= by < env.height):
                return True
            if env.board[by, bx] != 0:
                return True
        return False

    height, width = matrix.shape
    return {
        (x, y)
        for y in range(-height, env.board.shape[0] + 1)
        for x in range(-width, env.board.shape[1] + 1)
        if not collision(x, y) and collision(x, y + 1)
    }


@pytest.mark.parametrize("index, empty_count", [(0, 17), (1, 9), (2, 34), (3, 17), (4, 17), (5, 34), (6, 34)])
@pytest.mark.parametrize("with_blocks", [False, True])
def test_all_raw_placements_covered_by_logical_action_space(setup, index, empty_count, with_blocks):
    env, adapter, space = setup
    if with_blocks:
        env.board[19, 4:9] = 2
        env.board[17:20, 11] = 3
        env.board[12, 6:10] = 4
    spawn = env.tetrominoes[index].matrix
    canonical_by_shape = {}
    reference_targets = set()
    for k in range(4):
        # Actual matrix rotations, entirely independent of target-space helpers.
        matrix = np.rot90(spawn, k)
        rows, cols = np.nonzero(matrix > 0)
        min_row, min_col = int(rows.min()), int(cols.min())
        normalized_cells = frozenset(zip((rows - min_row).tolist(), (cols - min_col).tolist()))
        canonical = canonical_by_shape.setdefault(normalized_cells, k)
        raw_origins = reference_resting_origins(env, matrix)
        assert raw_origins
        shape_width = int(cols.max() - cols.min() + 1)
        # Every raw orientation has both extreme edge placements represented.
        assert any(x + min_col == env.padding for x, _ in raw_origins)
        assert any(x + min_col + shape_width == env.padding + env.width for x, _ in raw_origins)
        for env_x, env_y in raw_origins:
            expected = TargetPlacement(env_x + min_col, env_y + min_row, k)
            assert env_origin_to_target(env_x, env_y, k, spawn) == expected
            action_id = space.encode(expected)
            assert space.decode(action_id) == expected
            assert target_to_env_origin(expected, spawn) == (env_x, env_y)
            # Duplicate slots remain expressible but invalid, with the same
            # physical placement represented by the first shape-equivalent slot.
            assert space.is_valid(env.board, spawn, expected) == (k == canonical)
            reference_targets.add(TargetPlacement(expected.x, expected.y, canonical))
    mask = space.action_mask(adapter.read_board(), spawn)
    actual_targets = {space.decode(i) for i in np.flatnonzero(mask)}
    assert actual_targets == reference_targets
    if not with_blocks:
        assert len(reference_targets) == empty_count


@pytest.mark.parametrize("index", range(7))
@pytest.mark.parametrize("orientation", range(4))
def test_coordinate_conversion_and_offsets_for_every_orientation(setup, index, orientation):
    env, _, _ = setup
    spawn = env.tetrominoes[index].matrix
    rotated = np.rot90(spawn, orientation)
    rows, cols = np.nonzero(rotated > 0)
    expected_offset = (int(cols.min()), int(rows.min()))
    assert occupied_offset(rotated) == expected_offset
    target = TargetPlacement(4, 0, orientation)
    expected_origin = (4 - expected_offset[0], -expected_offset[1])
    assert target_to_env_origin(target, spawn) == expected_origin
    assert env_origin_to_target(*expected_origin, orientation, spawn) == target


@pytest.mark.parametrize("orientation, left_origin, right_origin, bbox_y", [
    (0, (4, 18), (10, 18), 19),
    (1, (3, 16), (12, 16), 16),
])
def test_i_horizontal_and_vertical_extreme_edges(setup, orientation, left_origin, right_origin, bbox_y):
    env, adapter, space = setup
    spawn = env.tetrominoes[0].matrix
    for origin in (left_origin, right_origin):
        target = env_origin_to_target(*origin, orientation, spawn)
        assert target.y == bbox_y
        assert space.decode(space.encode(target)) == target
        assert target_to_env_origin(target, spawn) == origin
        assert space.is_valid(adapter.read_board(), spawn, target)


def test_blank_top_border_does_not_invalidate_geometric_placement(setup):
    env, adapter, space = setup
    env.board[1, 4:14] = 2
    spawn = env.tetrominoes[0].matrix
    target = TargetPlacement(4, 0, 0)
    # Horizontal I has min_row=1. Raw y=-1 is unsupported by NumPy slicing
    # in the environment, but all occupied cells are in row 0 and resting.
    assert target_to_env_origin(target, spawn) == (4, -1)
    assert (4, -1) in reference_resting_origins(env, spawn)
    assert space.is_valid(adapter.read_board(), spawn, target)
    assert space.action_mask(adapter.read_board(), spawn)[space.encode(target)]


def test_seed_42_legacy_set_and_logical_set_match_physically(setup):
    env, adapter, space = setup
    spawn = adapter.read_current_piece().spawn_matrix
    legacy_raw_targets = set()
    logical_reference = set()
    for k in (0, 1):
        matrix = np.rot90(spawn, k)
        rows, cols = np.nonzero(matrix > 0)
        for env_x, env_y in reference_resting_origins(env, matrix):
            if env.padding <= env_x < env.padding + env.width and 0 <= env_y < env.height:
                legacy_raw_targets.add(TargetPlacement(env_x, env_y, k))
                logical_reference.add(TargetPlacement(env_x + int(cols.min()), env_y + int(rows.min()), k))
    actual = {space.decode(i) for i in np.flatnonzero(space.action_mask(adapter.read_board(), spawn))}
    # S's canonical slots 0/1 both have offset (0,0), so coordinates and count
    # happen to be unchanged. Other shapes have different offsets.
    assert actual == logical_reference == legacy_raw_targets
    assert len(actual) == 17


@pytest.mark.parametrize("orientation", [-1, 4])
def test_conversion_rejects_invalid_orientation(setup, orientation):
    spawn = setup[0].tetrominoes[0].matrix
    with pytest.raises(ValueError):
        target_to_env_origin(TargetPlacement(4, 0, orientation), spawn)
    with pytest.raises(ValueError):
        env_origin_to_target(4, 0, orientation, spawn)


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.ones(3)])
def test_offset_rejects_empty_or_non_matrix_piece(matrix):
    with pytest.raises(ValueError):
        occupied_offset(matrix)
