from scratch.run_s0_audit import (
    audit_annotations,
    parse_labeler,
    parse_site,
    point_distance_to_annotation,
)


def test_parse_site_uses_final_code_only():
    assert parse_site("20140609195854Bh.jpeg") == "Bh"
    assert parse_site("20140609195854Mh.jpeg") == "Mh"
    assert parse_site("unexpected.jpeg") == "unknown"


def test_parse_labeler_preserves_coco_prefix():
    assert parse_labeler("040301-20140609195854Bh") == "040301"
    assert parse_labeler("missing-prefix") == "missing"
    assert parse_labeler("plain") == "unknown"


def test_polygon_distance_supports_tolerant_spine_audit():
    square = [[0, 0, 10, 0, 10, 10, 0, 10]]
    assert point_distance_to_annotation((5, 5), square) > 0
    assert point_distance_to_annotation((11, 5), square) == -1


def test_audit_locks_latest_contiguous_year_and_keeps_zero_category():
    coco = {
        "categories": [{"id": 1, "name": "Left"}, {"id": 2, "name": "Ambiguous"}],
        "images": [
            {"id": "a-old", "file_name": "20200101000000Bh.jpeg", "date_captured": "2020-01-01T00:00:00", "height": 20, "width": 20},
            {"id": "b-new", "file_name": "20220101000000Mh.jpeg", "date_captured": "2022-01-01T00:00:00", "height": 20, "width": 20},
        ],
        "annotations": [
            {"id": 1, "image_id": "a-old", "category_id": 1, "area": 100, "bbox": [0, 0, 10, 10], "spine": [2, 2, 8, 8], "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},
            {"id": 2, "image_id": "b-new", "category_id": 1, "area": 100, "bbox": [0, 0, 10, 10], "spine": [2, 2, 8, 8], "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},
        ],
    }
    folds = [
        {"file_name": "20200101000000Bh.jpeg", "fold": "0"},
        {"file_name": "20220101000000Mh.jpeg", "fold": "1"},
    ]
    report, rows, _ = audit_annotations(coco, folds)
    assert report["chirality"]["instance_counts"]["Ambiguous"] == 0
    assert report["time"]["locked_latest_years"] == [2022]
    assert [row["time_block_role"] for row in rows] == ["development", "locked_evaluation"]
