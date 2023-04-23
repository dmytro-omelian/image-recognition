package org.example.neural_network;

import java.util.List;

public interface Module {

    List<List<Double>> forward(List<List<Double>> X);

}
