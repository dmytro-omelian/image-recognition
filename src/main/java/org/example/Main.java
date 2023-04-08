package org.example;

import org.example.entity.data.Dataset;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        Dataset dataset = new Dataset("src/main/java/org/example/data/train.csv");
        List<int[]> train = dataset.load();
        System.out.println(train.size());
    }
}
