import numpy as np

from parafoil_planner_v3.environment import GridTerrain, NoFlyPolygon, load_no_fly_polygons


def test_no_fly_polygon_signed_distance():
    square = NoFlyPolygon(vertices=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]))
    assert square.signed_distance_m(5.0, 5.0) < 0.0
    assert square.signed_distance_m(15.0, 5.0) > 0.0


def test_grid_terrain_interpolation_and_load(tmp_path):
    height = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    path = tmp_path / "terrain.npz"
    np.savez(path, height_m=height, origin_n=0.0, origin_e=0.0, resolution_m=1.0)

    terrain = GridTerrain.from_file(path)
    assert np.isclose(terrain.height_m(0.0, 0.0), 0.0)
    assert np.isclose(terrain.height_m(1.0, 0.0), 2.0)
    assert np.isclose(terrain.height_m(0.5, 0.5), 1.5)


def test_load_no_fly_polygons_yaml(tmp_path):
    yaml_text = """
polygons:
  - vertices:
      - [0.0, 0.0]
      - [10.0, 0.0]
      - [10.0, 10.0]
      - [0.0, 10.0]
    clearance_m: 1.0
"""
    path = tmp_path / "nofly.yaml"
    path.write_text(yaml_text)

    polys = load_no_fly_polygons(path)
    assert len(polys) == 1
    poly = polys[0]
    assert poly.signed_distance_m(5.0, 5.0) < 0.0
