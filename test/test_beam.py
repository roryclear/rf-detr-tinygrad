from tinygrad import Tensor
from rfdetr import RFDETR
import numpy as np
from tinygrad.helpers import Context
if __name__ == "__main__":
    Tensor.manual_seed(1337)
    model = RFDETR("nano")
    Tensor.manual_seed(1337)
    res = model().numpy()
    Tensor.manual_seed(1337)
    with Context(BEAM=2): res_beam = model().numpy()
    np.testing.assert_allclose(res, res_beam, rtol=1e-4)
    print("passed")