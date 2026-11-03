import torch
from torch import tensor

from DL_modules.custom_layers import (
    ContractAxes,
    MediumDenseNet,
    PairToOutput,
    SandwichedELU,
    TopLayerSubtractDenseSigmoid,
    TwoLevelNormalNormalOneProj,
    TwoLevelNormalNormalTwoProj,
)
from DL_modules.custom_layers.two_level_normal_normal_sub_modules import LLHT, LLHNormal, Q


def test_small_dense_net():
    model = MediumDenseNet(18, [10, 10, 5], 5)
    output = model(torch.randn(18))
    assert len(output) == 5


def test_sandwichedelu():
    model = SandwichedELU(18, 10, 5)
    output = model(torch.randn(18))
    assert len(output) == 5


def test_sigmoid_subtract_layer():
    susp_emb = torch.randn(5)
    cr_emb = torch.randn(5)
    model = TopLayerSubtractDenseSigmoid(5)
    output = model(susp_emb, cr_emb)
    assert len(output) == 1


def test_tlm_one_proj_layer():
    susp_emb = torch.randn(5)
    cr_emb = torch.randn(5)
    model = TwoLevelNormalNormalOneProj(5, 1000)
    output = model(susp_emb, cr_emb)
    assert len(output) == 4


def test_tlm_two_proj_layer():
    susp_emb = torch.randn(5)
    cr_emb = torch.randn(5)
    model = TwoLevelNormalNormalTwoProj(5, 1000)
    output = model(susp_emb, cr_emb)
    assert len(output) == 4


def test_broadcasting():
    a = torch.tensor([-1, 1])
    b = torch.tensor([[-1, 0], [1, 0]])
    result = torch.matmul(a, b)
    expected = torch.tensor([2, 0])
    a_unsq = a.unsqueeze(0)
    result_sq = torch.matmul(a_unsq, b)
    assert all(result == expected)
    assert all(result_sq.squeeze(0) == expected)


def test_matmul_two_matrices():
    a = torch.tensor([[-1, 1], [1, -1]])
    b = torch.tensor([[-1, 0], [1, 0]])
    result = torch.matmul(a, b)
    expected = torch.tensor([[2, 0], [-2, 0]])
    assert torch.equal(result, expected)


def test_orthonormality_q():
    n_dim = 15
    q_instance = Q(n_dim)
    q = q_instance.forward()
    result = torch.matmul(q.transpose(0, 1), q)
    torch.testing.assert_close(result, torch.eye(n_dim, dtype=torch.float64))


def test_pair_to_output():
    susp_fsw = MediumDenseNet(18, [10, 10, 5], 5)
    cr_fsw = MediumDenseNet(18, [10, 10, 5], 5)
    top_layer = TopLayerSubtractDenseSigmoid(5)
    pair_to_output_model = PairToOutput(susp_fsw, cr_fsw, top_layer)
    output = pair_to_output_model(torch.randn(18), torch.randn(18))
    assert len(output) == 1


def test_pair_to_output_on_batch(data_loader, pair_to_output_model):
    for batch in data_loader:
        susp, cr, labels = batch
        predictions = pair_to_output_model(susp, cr)
        print(f'n_predictions = {len(predictions)}')
        assert len(predictions) == len(susp)
        break


def test_llh_normal():
    llh_normal_instance = LLHNormal()
    z = 2
    var = 3
    result = llh_normal_instance.forward(torch.tensor(z), torch.tensor(var))
    # Create a normal distribution object
    normal_dist = torch.distributions.Normal(0, torch.sqrt(torch.tensor(var)))
    x = torch.tensor(z * torch.sqrt(torch.tensor(var)))
    ln_pdf = normal_dist.log_prob(x)
    log_pdf = ln_pdf / torch.log(torch.tensor(10))
    assert result == log_pdf


def test_llh_normal_extreme():
    llh_normal_instance = LLHNormal()
    z = 1000
    var = 3
    result = llh_normal_instance.forward(torch.tensor(z), torch.tensor(var))

    # Create a normal distribution object
    normal_dist = torch.distributions.Normal(0, torch.sqrt(torch.tensor(var)))
    x = torch.tensor(z * torch.sqrt(torch.tensor(var)))
    ln_pdf = normal_dist.log_prob(x)
    log_pdf = ln_pdf / torch.log(torch.tensor(10))
    assert result == log_pdf


def test_llh_t():
    llh_t_instance = LLHT(100)
    z = 2
    var = torch.tensor(10)
    result = llh_t_instance.forward(torch.tensor(z), var)

    nu = 2 * 52  # value of alpha_post = 52 in llh_t_instance
    est_sd_for_t_distr = torch.tensor(3.2085)  # value in llh_t_instance

    # Create a t-distribution object
    t_dist = torch.distributions.StudentT(nu)
    t = torch.tensor(1.9712)  # value of z_adjusted in llh_t_instance
    ln_pdf = t_dist.log_prob(t) - torch.log(est_sd_for_t_distr)
    log_pdf = ln_pdf / torch.log(torch.tensor(10))
    assert torch.isclose(result, log_pdf, atol=1e-10)


def test_llh_t_extreme():
    llh_t_instance = LLHT(100)
    z = 1000
    var = 100
    result = llh_t_instance.forward(torch.tensor(z), torch.tensor(var))

    nu = 2 * 52  # value of alpha_post = 52 in llh_t_instance
    est_sd_for_t_distr = torch.tensor(10.1460)

    # Create a t-distribution object
    t_dist = torch.distributions.StudentT(nu)

    t = torch.tensor(985.6053)  # value of z_adjusted in llh_t_instance
    ln_pdf = t_dist.log_prob(t) - torch.log(est_sd_for_t_distr)
    log_pdf = ln_pdf / torch.log(torch.tensor(10))
    assert torch.isclose(result, log_pdf, rtol=1e-10)


def test_convergence_student_t_to_normal():
    var = torch.tensor(10)
    llh_t_instance = LLHT(1000)
    z = 2
    z_adjusted = 1.9970  # value of z_adjusted in llh_t_instance

    result = llh_t_instance.forward(torch.tensor(z), var)

    # Create a normal distribution object
    normal_dist = torch.distributions.Normal(0, torch.sqrt(var))
    ln_normal_pdf = normal_dist.log_prob(torch.tensor(z_adjusted) * torch.sqrt(var))
    log_normal_pdf = ln_normal_pdf / torch.log(torch.tensor(10))

    # relative tol = 1e-4 since convergence is not absolute. Note that when increasing the sample size
    # (which is determined by '1000') to huge numbers the difference wil increase due to floating point errors
    # (according to chatgpt). An experimental size of 1000 is already very large.
    assert torch.isclose(result, log_normal_pdf, 1e-4)


def test_convergence_student_t_to_normal_very_large_n():
    var = torch.tensor(10)
    llh_t_instance = LLHT(10**4)
    z = 2
    z_adjusted = 1.9970  # value of z_adjusted in llh_t_instance

    result = llh_t_instance.forward(torch.tensor(z), var)

    # Create a normal distribution object
    normal_dist = torch.distributions.Normal(0, torch.sqrt(var))
    ln_normal_pdf = normal_dist.log_prob(torch.tensor(z_adjusted) * torch.sqrt(var))
    log_normal_pdf = ln_normal_pdf / torch.log(torch.tensor(10))

    # relative tol = 1e-2 since convergence is not absolute. Note that when increasing the sample size
    # (which is determined by '10**4') to huge numbers the difference wil increase due to floating point errors
    # (according to chatgpt). An experimental size of 1000 is already very large.
    assert torch.isclose(result, log_normal_pdf, 1e-2)


def test_contract_axes():
    contract_axes = ContractAxes(2, 1)
    x = torch.Tensor([[1, 0], [0, 0], [-2, -2]])
    a = torch.tensor([1, 2]).unsqueeze(1)
    b = a
    m = a
    result = contract_axes.contract(x, a, b, m)
    assert torch.allclose(result, tensor([[0.620115, 0.0], [0.0, 0.0], [-1.0, -1.662501]]), rtol=1e-6)


def test_contract_axes_3x_2comp():
    contract_axes = ContractAxes(1, 2)
    x = torch.Tensor([[1], [0], [-2]])
    result = contract_axes(x)
    assert result.shape == torch.Size([3, 1])


def test_contract_axes_1x_2comp():
    contract_axes = ContractAxes(1, 2)
    x = torch.Tensor([[1]])
    result = contract_axes(x)
    assert result.shape == torch.Size([1, 1])


def test_contract_axes_1x_2comp_a():
    contract_axes = ContractAxes(1, 2)
    x = torch.Tensor([1])
    result = contract_axes(x)
    assert result.shape == torch.Size([1])


def test_contract_axes_aimed_constraints():
    contract_axes = ContractAxes(1, 1)
    x = torch.linspace(0, 2, 1000).unsqueeze(1)
    a = torch.pow(10, torch.randn(1))
    b = torch.pow(10, torch.randn(1))
    multiplier = torch.pow(10, torch.randn(1)).unsqueeze(1)
    result = contract_axes.contract(x, a, b, multiplier)
    atol = 1e-6
    assert all(x - result >= -atol)
    assert all(x[1:] - x[:-1] - (result[1:] - result[:-1]) >= -atol)
    assert all(result[1:] - result[:-1] >= -atol)


"""
def test_contract_axes_derivative():
    contract_axes = ContractAxes(2, 1)
    x = torch.Tensor([[1, 1], [0, 0], [-2, -2]])
    a = torch.tensor([1, 2]).unsqueeze(1)
    b = a
    c = a
    d = a
    e = a
    m = a
    result = contract_axes.derivative(x, a, b, c, d, e, m)
    assert torch.allclose(result, tensor([[2/3, 0.5], [1.0, 1.0], [0.406155, 0.125668]]), rtol=1e-5)
"""
