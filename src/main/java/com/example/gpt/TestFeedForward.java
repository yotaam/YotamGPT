package com.example.gpt;

public class TestFeedForward {
    public static void main(String[] args) {
        int batchSize = 2;
        int embDim = 8;

        Matrix input = Matrix.random(batchSize, embDim, 0.0, 1.0);
        FeedForward ff = new FeedForward(embDim);
        Matrix output = ff.forward(input);

        System.out.println("Input shape: (" + input.getRows() + ", " + input.getCols() + ")");
        System.out.println("Output shape: (" + output.getRows() + ", " + output.getCols() + ")");
    }
}
