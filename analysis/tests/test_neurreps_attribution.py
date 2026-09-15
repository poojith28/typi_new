import importlib.util
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1] / "neurreps_revision.py"
S = importlib.util.spec_from_file_location("neurreps_revision", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def toy_ref():
    return {"births": np.array([1, 2, 3]), "endpoints": np.array([[0,1],[1,2],[2,3]]),
            "deaths": np.array([.1,.2,.3]), "n_pool": 4}


def test_permutation_inverse_recovers_original_ids():
    rng=np.random.RandomState(20260821); p=rng.permutation(100)
    inv=np.empty(100,int); inv[p]=np.arange(100)
    assert np.array_equal(p[inv],np.arange(100))
    assert np.array_equal(inv[p],np.arange(100))


def test_edge_attribution_symmetric_under_endpoint_swap():
    r=toy_ref(); a=M.touch_metrics(np.array([1]),r,"edge")
    r["endpoints"]=r["endpoints"][:,::-1]
    b=M.touch_metrics(np.array([1]),r,"edge")
    assert a==b


def test_death_multiset_unchanged_by_vertex_relabeling():
    x=M.l2_normalize(np.array([[1.,0.],[1.,1.],[0.,1.],[-1.,2.],[2.,-1.]]))
    p=np.array([2,0,4,1,3])
    _,_,w0=M.prim_mst(x)
    up,vp,w1=M.prim_mst(x[p])
    inv=np.empty(len(p),int); inv[p]=np.arange(len(p))
    # `p` maps permuted row IDs back to original pool IDs.
    mapped_edges=np.sort(p[np.column_stack([up,vp])],axis=1)
    assert mapped_edges.min() >= 0 and mapped_edges.max() < len(x)
    assert np.allclose(np.sort(w0),np.sort(w1),atol=1e-12,rtol=0)


def test_cumulative_coverage_cannot_decrease():
    r=toy_ref(); sets=[np.array([0]),np.array([0,1]),np.array([0,1,2])]
    x=[M.touch_metrics(s,r,"edge")["touched_count"] for s in sets]
    assert np.all(np.diff(x)>=0)


def test_out_of_range_selected_index_rejected():
    try: M.touch_metrics(np.array([4]),toy_ref(),"birth")
    except IndexError: pass
    else: raise AssertionError("out-of-range index accepted")


def test_identical_sets_identical_diagnostics():
    r=toy_ref(); a=M.touch_metrics(np.array([3,1]),r,"birth"); b=M.touch_metrics(np.array([1,3]),r,"birth")
    assert a==b
