from stlbench.plate_groups import (
    PackablePart,
    build_packable_parts,
    greedy_shelf_plates,
)


def test_greedy_two_parts_one_plate():
    parts = [
        PackablePart("a", height_z=5, footprint_w=10, footprint_h=10),
        PackablePart("b", height_z=5, footprint_w=10, footprint_h=10),
    ]
    plates = greedy_shelf_plates(parts, px=25, py=25)
    assert len(plates) == 1
    assert set(plates[0]) == {"a", "b"}


def test_build_packable_parts_filters_too_tall():
    ok, bad = build_packable_parts(
        ["tall", "ok"],
        [(5.0, 5.0, 200.0), (5.0, 10.0, 10.0)],
        px=100,
        py=100,
        pz=50,
    )
    assert "tall" in bad
    assert len(ok) == 1
    assert ok[0].name == "ok"
