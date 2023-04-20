package org.example;

import org.example.entity.data.DataLoader;
import org.example.entity.data.Dataset;
import org.example.neural_network.LogisticRegression;
import org.example.preprocessing.PreprocessService;

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        Dataset train = new Dataset("src/main/java/org/example/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();
        System.out.println(train.headTrainY());

        // FIXME features -> it is double values (especially after normalization)


        var features = train.getFeatures();
        var labels = train.getLabels();

        PreprocessService.TrainTestEntity trainTestEntity = PreprocessService.trainTestSplit(
                train.getFeatures(), train.getLabels(), 0.2);

        var train_X = trainTestEntity.getTrainX();
        var train_y = trainTestEntity.getTrainY();
        var test_X = trainTestEntity.getTestX();
        var test_y = trainTestEntity.getTestY();

        var trainLoader = new DataLoader(train_X, train_y, 8);
        var testLoader = new DataLoader(test_X, test_y, 8);

        // visualize image

        int inputDim = 28 * 28;
        int outputDim = 10;
        LogisticRegression model = new LogisticRegression(inputDim, outputDim);

        // for each epoch
        // for each batch
        // convert train and labels to smth
        // optimizer.zero_grad()
        // calculate model outputs // forward propagation
        // calculate loss with CrossEntropyLoss
        // loss backward()
        // optimizer.step()
        // count += 1
        // calculate prediction
        // - print loss

    }
}
