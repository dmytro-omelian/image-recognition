package org.example;

import org.example.entity.data.Dataset;

public class Main {
    public static void main(String[] args) {
        Dataset train = new Dataset("src/main/java/org/example/data/train.csv"); // FIXME add parameter load on start dataset
        train.load();
        System.out.println(train.headTrainY());

        // get targets
        // get features
        // normalize features

        // train_test_split

        // get train dataloader
        // get test dataloader

        // visualize image

        // create LogisticRegression

        // for each epoch
        // for each batch
        // convert train and labels to smth
        // optimizer.zero_grad()
        // calculate model outputs
        // calculate loss with CrossEntropyLoss
        // loss backward()
        // optimizer.step()
        // count += 1
        // calculate prediction
        // - print loss

    }
}
