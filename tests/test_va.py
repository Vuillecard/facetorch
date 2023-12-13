import pytest


@pytest.mark.integration
@pytest.mark.predictor
@pytest.mark.va
def test_va_in_preds(response):
    for face in response.faces:
        assert "va" in face.preds.keys()


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.va
def test_valence(response, cfg):
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Ony test.jpg is used for this test.")
    assert response.faces[1].preds["va"].other["valence"] > 0


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.va
def test_arousal(response, cfg):
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Ony test.jpg is used for this test.")
    assert response.faces[1].preds["va"].other["arousal"] > 0


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.va
def test_valence_range(response, cfg):
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Ony test.jpg is used for this test.")
    assert response.faces[1].preds["va"].other["valence"] >= -1
    assert response.faces[1].preds["va"].other["valence"] <= 1


@pytest.mark.endtoend
@pytest.mark.predictor
@pytest.mark.va
def test_arousal_range(response, cfg):
    if "test.jpg" not in cfg.path_image:
        pytest.skip("Ony test.jpg is used for this test.")
    assert response.faces[1].preds["va"].other["arousal"] >= -1
    assert response.faces[1].preds["va"].other["arousal"] <= 1
