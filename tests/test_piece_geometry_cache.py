import numpy as np
import pytest
from tetris_gymnasium.envs.tetris import Tetris
from tetris_agent._piece_geometry import piece_geometry
from tetris_agent.target_space import orientation_matrices, occupied_offset, unique_orientations


@pytest.mark.parametrize('dtype',[np.uint8,np.int64,np.float64])
def test_geometry_exact_and_owned(dtype):
    for piece in Tetris.TETROMINOES:
        source = piece.matrix.astype(dtype)
        for turn in range(4):
            matrix = np.rot90(source,turn)
            actual = orientation_matrices(matrix)
            expected = tuple(np.rot90(matrix,k).copy() for k in range(4))
            seen, unique = set(), []
            for k, (a,b) in enumerate(zip(actual,expected)):
                assert np.array_equal(a,b) and a.dtype == b.dtype
                assert a.flags.writeable and a.flags.owndata
                rows,cols = np.nonzero(b>0)
                assert occupied_offset(b)==(int(cols.min()),int(rows.min()))
                crop=(b>0)[rows.min():rows.max()+1,cols.min():cols.max()+1]
                key=crop.shape,crop.tobytes()
                if key not in seen: unique.append(k); seen.add(key)
            assert unique_orientations(matrix)==tuple(unique)
            actual[0][:]=0
            assert np.array_equal(orientation_matrices(matrix)[0],matrix)
            with pytest.raises(ValueError):
                piece_geometry(matrix).matrices[0].setflags(write=True)


def test_source_mutation_uses_new_geometry():
    matrix=np.array([[0,1],[1,1]],dtype=np.uint8)
    assert occupied_offset(matrix)==(0,0)
    matrix[0,:]=0; matrix[1,0]=0
    assert occupied_offset(matrix)==(1,1)
    assert np.array_equal(orientation_matrices(matrix)[0],matrix)


def test_large_matrix_preserves_public_fallback():
    matrix=np.zeros((20,20),dtype=np.uint8); matrix[5,7]=2
    assert piece_geometry(matrix) is None
    assert occupied_offset(matrix)==(7,5)
    assert unique_orientations(matrix)==(0,)
