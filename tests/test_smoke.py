from latent_error_correcting_representations.experiment import run_seed


def test_small_smoke_run(tmp_path):
    r = run_seed(3, tmp_path, epochs=4)
    assert r["encoder"]["message_bit_accuracy"] > 0.9
    assert r["random_corruption"]["hamming74_repair"]["1"]["task_accuracy"] == 1.0
    assert r["coherent_drift"]["hamming74_repair"]["detection_rate"] == 0.0
