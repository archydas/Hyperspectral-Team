"""ROI mosaic / composite / stack of one scene (Data Aggregation > Mosaic / Stack, Source = ROIs of one scene)."""
import numpy as np
import pytest

from hvva.core.roi_mosaic import roi_composite, roi_mosaic, roi_stack
from hvva.core.scene import Scene


def _scene():
    cube = np.arange(20 * 30 * 4, dtype=np.float32).reshape(20, 30, 4)
    sc = Scene("s", cube, np.array([500.0, 600, 700, 800]), unit="reflectance")
    a = sc.add_roi({"kind": "rect", "rect": [0, 0, 5, 4], "class": "A"})       # 5 x 4
    b = sc.add_roi({"kind": "rect", "rect": [10, 10, 13, 16], "class": "B"})   # 3 x 6
    c = sc.add_roi({"kind": "rect", "rect": [2, 20, 4, 22], "class": "A"})     # 2 x 2
    return sc, [a, b, c]


def test_side_by_side_one_row_and_labels():
    sc, rois = _scene()
    res = roi_mosaic(sc, rois, "side", gap=1)
    assert res.shape == (5, 4 + 1 + 6 + 1 + 2, 4)
    assert res.unit == "reflectance" and res.geo is None
    m = res.class_masks()
    assert m["A"].sum() == 20 + 4 and m["B"].sum() == 18
    np.testing.assert_array_equal(res.cube[0, 5], sc.cube[10, 10])              # B starts after A + gap


def test_grid_columns_and_in_place():
    sc, rois = _scene()
    assert roi_mosaic(sc, rois, "side", gap=0, columns=2).shape == (5 + 2, 10, 4)
    res = roi_mosaic(sc, rois, "in_place")
    assert res.shape == sc.shape
    assert np.isnan(res.cube[19, 29]).all()
    np.testing.assert_array_equal(res.cube[11, 12], sc.cube[11, 12])


def test_stack_and_composite():
    sc, rois = _scene()
    arr, masks, meta = roi_stack(sc, rois)
    assert arr.shape == (3, 5, 6, 4) and masks.sum() == 20 + 18 + 4
    assert [m["class"] for m in meta] == ["A", "B", "A"]
    comp = roi_composite(sc, [rois[0], rois[2]], "mean")
    assert comp.shape == (5, 4, 4) and list(comp.class_names.values()) == ["A"]


def test_rejects_foreign_or_empty():
    sc, rois = _scene()
    other, _ = _scene()
    with pytest.raises(ValueError):
        roi_mosaic(other, rois)
    with pytest.raises(ValueError):
        roi_mosaic(sc, [])
