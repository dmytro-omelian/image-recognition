package org.example.entity.torch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Tensor {

    private final List<List<Double>> tensor;
    private final Random random;

    public Tensor(List<List<Double>> X) {
        this.tensor = X;
        this.random = new Random();
    }

    public Tensor(int height, int width, boolean randomInit) {
        if (randomInit) {
            this.tensor = randInit(height, width);
        } else {
            this.tensor = initZeros(height, width);
        }

        this.random = new Random();
    }

    private List<List<Double>> initZeros(int height, int width) {
        List<List<Double>> result = new ArrayList<>();
        for (int i = 0; i < height; ++ i) {
            result.add(new ArrayList<>(Collections.nCopies(width, 0.0)));
        }
        return result;
    }

    private List<List<Double>> randInit(int height, int width) {
        var result = new ArrayList<List<Double>>();
        for (int i = 0; i < height; ++ i) {
            var temp = new ArrayList<Double>();
            for (int j = 0; j < width; ++ j) {
                var value = random.nextDouble();
                temp.add(value);
            }
            result.add(temp);
        }
        return result;
    }

    public Shape getShape() {
        int height = this.tensor.size();
        int width = (this.tensor.size() > 0) ? this.tensor.get(0).size() : 0;
        return new Shape(height, width);
    }


    public Tensor T() {
        return null;
    }

    public void add(Tensor bias) {

    }
}
