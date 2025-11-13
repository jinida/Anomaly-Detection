import torch

def generate_perlin_noise(
    height: int,
    width: int,
    scale: tuple[int, int] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if scale is None:
        min_scale, max_scale = 0, 6
        scalex = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
        scaley = 2 ** torch.randint(min_scale, max_scale, (1,), device=device).item()
    else:
        scalex, scaley = scale

    def nextpow2(value: int) -> int:
        return int(2 ** torch.ceil(torch.log2(torch.tensor(value))).int().item())

    pad_h = nextpow2(height)
    pad_w = nextpow2(width)

    # Generate base grid
    delta = (scalex / pad_h, scaley / pad_w)
    d = (pad_h // scalex, pad_w // scaley)

    grid = (
        torch.stack(
            torch.meshgrid(
                torch.arange(0, scalex, delta[0], device=device),
                torch.arange(0, scaley, delta[1], device=device),
                indexing="ij",
            ),
            dim=-1,
        )
        % 1
    )

    angles = 2 * torch.pi * torch.rand(int(scalex) + 1, int(scaley) + 1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    def tile_grads(slice1: list[int | None], slice2: list[int | None]) -> torch.Tensor:
        return (
            gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]]
            .repeat_interleave(int(d[0]), 0)
            .repeat_interleave(int(d[1]), 1)
        )

    def dot(grad: torch.Tensor, shift: list[float]) -> torch.Tensor:
        return (
            torch.stack(
                (grid[:pad_h, :pad_w, 0] + shift[0], grid[:pad_h, :pad_w, 1] + shift[1]),
                dim=-1,
            )
            * grad[:pad_h, :pad_w]
        ).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])

    def fade(t: torch.Tensor) -> torch.Tensor:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    t = fade(grid[:pad_h, :pad_w])
    noise = torch.sqrt(torch.tensor(2.0, device=device)) * torch.lerp(
        torch.lerp(n00, n10, t[..., 0]),
        torch.lerp(n01, n11, t[..., 0]),
        t[..., 1],
    )
    return noise[:height, :width]