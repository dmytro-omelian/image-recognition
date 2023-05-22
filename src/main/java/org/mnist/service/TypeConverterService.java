package org.mnist.service;

import java.util.List;
import java.util.stream.Collectors;

public class TypeConverterService {

    public TypeConverterService() {
    }

    public static double[][] convertToArrayOfDoubleArrays(List<List<Double>> list) {
        return list.stream()
                .map(row -> row.stream().mapToDouble(Double::doubleValue).toArray())
                .toArray(double[][]::new);
    }

    public static List<double[]> convertToListOfDoubleArrays(List<List<Double>> list) {
        return list.stream()
                .map(innerList -> innerList.stream()
                        .mapToDouble(Double::doubleValue)
                        .toArray())
                .collect(Collectors.toList());
    }

}
