package org.example.neural_network;

import org.example.entity.torch.Tensor;

public interface Module {

    Tensor forward(Tensor X);

}
