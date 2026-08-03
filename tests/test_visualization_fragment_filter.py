from visualization_fragment_filter import (
    detect_suspect_fragment_aliases,
    visible_suspect_aliases,
)


def _object(
    object_id,
    points,
    bbox,
    centroid,
    cameras,
    prompt="a magenta block",
    interaction_probability=0.0,
):
    return {
        "id": object_id,
        "points_world": points,
        "point_count": len(points),
        "bbox3d_world": bbox,
        "centroid_world": centroid,
        "visible_camera": cameras,
        "role_evidence": {
            "interaction_part": {"probability": interaction_probability}
        },
        "observations": [
            {
                "provenance": {
                    "prompt_provenance": [{"source_prompt": prompt}]
                }
            }
        ],
    }


def _fragment_frame(frame_id="0", receiver_cameras=None, shift=0.0):
    receiver_cameras = receiver_cameras or ["front", "right_shoulder"]
    donor_points = [
        [shift + 0.020, 0.020, 0.020],
        [shift + 0.030, 0.030, 0.030],
        [shift + 0.040, 0.040, 0.040],
    ]
    receiver_points = donor_points + [
        [shift + 0.000, 0.000, 0.000],
        [shift + 0.010, 0.010, 0.010],
        [shift + 0.050, 0.050, 0.050],
        [shift + 0.060, 0.060, 0.060],
        [shift + 0.070, 0.070, 0.070],
        [shift + 0.080, 0.080, 0.080],
    ]
    return {
        "frame_id": frame_id,
        "objects": [
            _object(
                "O4",
                receiver_points,
                [[shift, 0.0, 0.0], [shift + 0.08, 0.08, 0.08]],
                [shift + 0.030, 0.030, 0.030],
                receiver_cameras,
            ),
            _object(
                "O10",
                donor_points,
                [[shift + 0.02, 0.02, 0.02], [shift + 0.04, 0.04, 0.04]],
                [shift + 0.031, 0.030, 0.030],
                ["front"],
            ),
        ],
    }


def test_multiview_receiver_confirms_fragment_in_one_frame():
    result = detect_suspect_fragment_aliases([_fragment_frame()])

    assert result["aliases"] == {"O10": "O4"}
    assert result["evidence"][0]["has_multiview_receiver"] is True
    assert result["evidence"][0]["evidence_frame_count"] == 1


def test_single_camera_pair_is_not_hidden_from_one_frame_only():
    frame = _fragment_frame(receiver_cameras=["front"])

    assert detect_suspect_fragment_aliases([frame])["aliases"] == {}


def test_stable_single_camera_pair_is_confirmed_temporally():
    frames = [
        _fragment_frame("0", receiver_cameras=["front"], shift=0.0),
        _fragment_frame("1", receiver_cameras=["front"], shift=0.1),
    ]

    result = detect_suspect_fragment_aliases(frames)

    assert result["aliases"] == {"O10": "O4"}
    assert result["evidence"][0]["has_multiview_receiver"] is False
    assert result["evidence"][0]["evidence_frame_count"] == 2


def test_different_semantic_prompts_are_not_aliased():
    frame = _fragment_frame()
    frame["objects"][1]["observations"][0]["provenance"]["prompt_provenance"][0][
        "source_prompt"
    ] = "a different physical object"

    assert detect_suspect_fragment_aliases([frame])["aliases"] == {}


def test_interaction_part_evidence_protects_a_small_real_part():
    frame = _fragment_frame()
    frame["objects"][1]["role_evidence"]["interaction_part"]["probability"] = 0.8

    assert detect_suspect_fragment_aliases([frame])["aliases"] == {}


def test_alias_is_visible_only_while_receiver_coexists():
    frame = _fragment_frame()
    aliases = {"O10": "O4"}

    assert visible_suspect_aliases(frame["objects"], aliases) == aliases
    assert visible_suspect_aliases([frame["objects"][1]], aliases) == {}


def test_alias_selection_never_creates_a_donor_receiver_chain():
    frame = _fragment_frame()
    outer = _object(
        "O9",
        frame["objects"][0]["points_world"]
        + [[-0.01, -0.01, -0.01], [0.09, 0.09, 0.09], [0.1, 0.1, 0.1]]
        + [[0.03, 0.03, 0.03]] * 15,
        [[-0.01, -0.01, -0.01], [0.1, 0.1, 0.1]],
        [0.03, 0.03, 0.03],
        ["front", "left_shoulder"],
    )
    frame["objects"].insert(0, outer)

    aliases = detect_suspect_fragment_aliases([frame])["aliases"]

    assert aliases["O10"] == "O4"
    assert not (set(aliases) & set(aliases.values()))
