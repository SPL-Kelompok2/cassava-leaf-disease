import pytest

torch = pytest.importorskip("torch")

from cassava_es.losses import TaylorCrossEntropyLoss, TaylorSoftmax  # noqa: E402


def test_taylor_softmax_jumlah_satu_dan_positif():
    x = torch.randn(8, 5) * 10
    p = TaylorSoftmax(dim=1, n=2)(x)
    assert torch.allclose(p.sum(1), torch.ones(8), atol=1e-5)
    assert (p > 0).all()


def test_taylor_softmax_sesuai_rumus_manual():
    z = torch.tensor([[0.0, 1.0, -1.0]])
    f = 1 + z + z**2 / 2
    assert torch.allclose(TaylorSoftmax(n=2)(z), f / f.sum(), atol=1e-6)


def test_orde_harus_genap():
    for bad in (1, 3, 0):
        with pytest.raises(ValueError):
            TaylorSoftmax(n=bad)


def test_loss_sesuai_rumus_manual_dengan_smoothing():
    z = torch.tensor([[2.0, 0.0, -1.0, 0.5, 0.1]])
    y = torch.tensor([0])
    s, k = 0.05, 5
    f = 1 + z + z**2 / 2
    logp = (f / f.sum()).log()
    dist = torch.full((1, 5), s / (k - 1))
    dist[0, 0] = 1 - s
    expected = -(dist * logp).sum()
    got = TaylorCrossEntropyLoss(5, n=2, smoothing=s)(z, y)
    assert got.item() == pytest.approx(expected.item(), abs=1e-6)


def test_loss_lebih_kecil_bila_logit_mendukung_kelas_benar():
    y = torch.tensor([2, 2])
    bagus = torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0]] * 2)
    buruk = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    crit = TaylorCrossEntropyLoss(5)
    assert crit(bagus, y) < crit(buruk, y)


def test_gradien_hingga_dan_aman_untuk_logit_besar():
    z = (torch.randn(16, 5) * 30).requires_grad_()
    loss = TaylorCrossEntropyLoss(5)(z, torch.randint(0, 5, (16,)))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_smoothing_tidak_valid():
    with pytest.raises(ValueError):
        TaylorCrossEntropyLoss(5, smoothing=1.0)
