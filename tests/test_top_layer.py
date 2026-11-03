import torch

from DL_modules.custom_layers.top_layers import TwoLevelNormalNormalOneProj


def test_z():
    """Test if the z-value is calculated."""
    m = TwoLevelNormalNormalOneProj(8, 8)
    projection = torch.tensor([[-1], [2], [4], [-4], [3], [-2], [0]], dtype=torch.float32)
    variance = torch.tensor([2], dtype=torch.float32)
    z = m.z(projection, variance)
    assert len(z) == len(projection)


def test_z_with_clip():
    """Test if the z-value is clipped when it needs to be clipped ."""
    model = TwoLevelNormalNormalOneProj(8, 8, n_clip=1, n_ss_train=4)

    projection = torch.tensor([[-1], [2], [4], [-4], [3], [-2], [0]], dtype=torch.float32)
    variance = torch.tensor([2], dtype=torch.float32)

    assert model.z_clip is not None
    assert model.z_clip - 0.01 <= -0.674 <= model.z_clip + 0.01

    # z0 is the reference, z1 should not be clipped and z2 should be clipped.
    z_with_clip = model.z(projection, variance, use_z_clip=True)
    z_no_clip = model.z(projection, variance, use_z_clip=False)

    assert not torch.equal(z_with_clip, z_no_clip)

    assert torch.min(z_with_clip) >= -0.675
    assert torch.max(z_with_clip) <= 0.675
    assert torch.min(z_no_clip) <= -0.675
    assert torch.max(z_no_clip) >= 0.675


def test_z_clip_value():
    """Test if the z-value is calculated."""
    m = TwoLevelNormalNormalOneProj(8, 8)
    assert m.get_clip_value(0, 10) == float('-inf')
    assert m.get_clip_value(1, 2) == 0
    assert m.get_clip_value(1, 4) < 0
    assert m.get_clip_value(1, 4) - 0.01 <= -0.674 <= m.get_clip_value(1, 2) + 0.01
