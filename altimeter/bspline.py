import torch

def _basis(x: torch.Tensor, k: int, i: int, t: torch.Tensor) -> torch.Tensor:
    """Compute the i-th B-spline basis function of degree ``k``.

    Parameters
    ----------
    x : torch.Tensor
        Points at which to evaluate the basis function.
    k : int
        Polynomial degree.
    i : int
        Basis index.
    t : torch.Tensor
        Knot vector with shape ``(batch, num_knots, out_dim)``.

    Returns
    -------
    torch.Tensor
        Values of the basis function evaluated at ``x``.
    """
    out = torch.zeros_like(x)

    if k == 0:
        return torch.where(
            torch.logical_and(t[:, i, :] <= x, x < t[:, i + 1, :]), 1.0, 0.0
        )

    if t[0, i + k, 0] == t[0, i, 0]:
        c1 = torch.zeros_like(x)
    else:
        c1 = (x - t[:, i, :]) / (t[:, i + k, :] - t[:, i, :]) * _basis(x, k - 1, i, t)

    if t[0, i + k + 1, 0] == t[0, i + 1, 0]:
        c2 = torch.zeros_like(x)
    else:
        c2 = (
            (t[:, i + k + 1, :] - x)
            / (t[:, i + k + 1, :] - t[:, i + 1, :])
            * _basis(x, k - 1, i + 1, t)
        )

    return c1 + c2


def eval_bspline(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int) -> torch.Tensor:
    """Evaluate a B-spline for a batch of inputs."""
    n = t.shape[1] - k - 1
    out = torch.zeros_like(x)
    for idx in range(n):
        out += c[:, idx, :] * _basis(x, k, idx, t)
    return out