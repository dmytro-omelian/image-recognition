package org.example.entity.data;

import java.util.List;

public class DataFrame {

    private String[] columnNames;
    private final List<int[]> df;

    public DataFrame(List<int[]> df) {
        this.df = df;
    }

    public int shape(int index) {
        if (index == 0) {
            return df.size();
        } else if (index == 1) {
            return df.get(0).length; // FIXME check if df is not empty
        }
        throw new IllegalStateException("Index is invalid."); // FIXME create custom exception here
    }
    
}
