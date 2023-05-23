package org.parallel_mnist.service;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class FileManagerService {

    private final String WEIGHTS_FILENAME = "weights.txt";

    public FileManagerService() {
    }

    public double[][] readWeights() {
        try (BufferedReader reader = new BufferedReader(new FileReader(WEIGHTS_FILENAME))) {
            String line;
            List<List<Double>> weights = new ArrayList<>();
            while ((line = reader.readLine()) != null) {
                List<Double> row;
                String[] values = line.trim().split("\\s+");
                row = Arrays.stream(values).map(Double::parseDouble).collect(Collectors.toList());
                weights.add(row);
            }
            return TypeConverterService.convertToArrayOfDoubleArrays(weights);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void saveWeights(double[][] weights) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(WEIGHTS_FILENAME))) {
            for (double[] row : weights) {
                for (double value : row) {
                    writer.write(String.valueOf(value));
                    writer.write(" ");
                }
                writer.newLine();
            }
            System.out.println("Array data saved to " + WEIGHTS_FILENAME);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
