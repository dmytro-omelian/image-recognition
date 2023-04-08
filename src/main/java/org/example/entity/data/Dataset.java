package org.example.entity.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Dataset {

    private final String path;

    public Dataset(String path) {
        this.path = path;
    }

    public List<int[]> load() {
        String line = "";
        List<int[]> train = new ArrayList<int[]>();
        try {
            boolean labelsSkipped = false;
            BufferedReader br = new BufferedReader(new FileReader(this.path));
            while ((line = br.readLine()) != null) {
                if (!labelsSkipped) { // FIXME it is bullshit to have smth like that in code
                    labelsSkipped = true;
                    continue;
                }
                String[] strPixels = line.split(",");
                int[] pixels = new int[strPixels.length];
                for (int i = 1; i < strPixels.length; ++ i) { // FIXME extract method
                    pixels[i - 1] = Integer.parseInt(strPixels[i]);
                }
                train.add(pixels);
            }
            return train;
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e); // FIXME handle exceptions in a better way
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
