from latent_error_correcting_representations.experiment import run_seed


def test_same_seed_repeatability(tmp_path):
    a = run_seed(5, tmp_path / "a", epochs=3)
    b = run_seed(5, tmp_path / "b", epochs=3)
    assert a["encoder"] == b["encoder"]
    assert a["random_corruption"] == b["random_corruption"]
    assert a["adversarial_single_flip"] == b["adversarial_single_flip"]
    assert a["coherent_drift"] == b["coherent_drift"]
