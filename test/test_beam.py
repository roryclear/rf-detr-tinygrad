from tinygrad import Tensor
from rfdetr import RFDETR
import numpy as np
from tinygrad.helpers import fetch, Context
if __name__ == "__main__":
    model = RFDETR("nano")
    input = Tensor.randn(model.res, model.res, 3)
    res = model(input).numpy()
    with Context(BEAM=2): res_beam = model(input).numpy()
    np.testing.assert_allclose(res, res_beam, rtol=1e-4)
    print("passed")