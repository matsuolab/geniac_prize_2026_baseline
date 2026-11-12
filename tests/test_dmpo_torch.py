"""The two-pass DMPO gradient equals autograd on the full loss. Needs torch (skipped without it; CI runs it)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from geniac_baseline.trainer.losses.dmpo import Pair, phi, two_pass_backward  # noqa: E402


def _setup(beta: float, gamma: float):
    torch.manual_seed(0)
    vocab, dim = 11, 5
    model = torch.nn.Sequential(torch.nn.Embedding(vocab, dim), torch.nn.Tanh(), torch.nn.Linear(dim, vocab)).double()
    lengths = {"w1": 3, "l1": 2, "w2": 1, "l2": 4}
    turns = {
        s: [torch.randint(0, vocab, (int(torch.randint(2, 6, ())),)) for _ in range(T)] for s, T in lengths.items()
    }

    def logp(s: str, k: int) -> torch.Tensor:
        ids = turns[s][k]
        return torch.log_softmax(model(ids[:-1]), -1).gather(1, ids[1:, None]).sum()

    with torch.no_grad():
        ref = {s: [float(logp(s, k)) + float(torch.randn(())) for k in range(T)] for s, T in lengths.items()}
    batch = [Pair("w1", "l1"), Pair("w2", "l2")]
    return model, lengths, ref, logp, batch


@pytest.mark.parametrize(("beta", "gamma"), [(0.5, 0.9), (0.5, 0.5), (0.1, 0.99)])
def test_two_pass_gradient_equals_autograd(beta: float, gamma: float) -> None:
    model, lengths, ref, logp, batch = _setup(beta, gamma)

    # autograd on the whole loss
    model.zero_grad()
    loss = 0
    for p in batch:
        side = {
            s: sum(phi(k, lengths[s], gamma) * (logp(s, k) - ref[s][k]) for k in range(lengths[s]))
            for s in (p.chosen, p.rejected)
        }
        m = beta * (side[p.chosen] - side[p.rejected])
        loss = loss - torch.nn.functional.logsigmoid(m) / len(batch)
    loss.backward()
    want = [q.grad.clone() for q in model.parameters()]

    # two passes, one turn at a time
    model.zero_grad()

    def turn_logp(s: str, k: int, grad: bool):
        if grad:
            return logp(s, k)
        with torch.no_grad():
            return float(logp(s, k))

    two_pass_backward(batch, lengths, ref, turn_logp, beta, gamma)
    got = [q.grad for q in model.parameters()]
    err = max(float((a - b).abs().max()) for a, b in zip(want, got, strict=True))
    assert err < 1e-12


def test_completion_ce_is_the_token_sum_over_the_answer() -> None:
    from geniac_baseline.trainer.loss import LossInputs, compute_loss

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 7, generator=g)
    target = torch.tensor([1, 2, 3, 4])
    mask = torch.tensor([True, True, True, False])
    out = compute_loss(LossInputs(logits, target, mask, 0, torch.arange(10), 6, {"kind": "first_turn"}, "r-t0"))
    want = torch.nn.functional.cross_entropy(logits[:3], target[:3], reduction="sum")
    assert torch.allclose(out.loss, want) and out.normalizer == 3 and out.n_loss_tokens == 3
