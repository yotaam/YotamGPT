package com.example.gpt;

public class GELU {
    public static Matrix forward(Matrix x) {
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);

        Matrix xCubed = x.multiply(x).multiply(x);

        Matrix inner = x.add(xCubed.multiply(0.044715)).multiply(sqrt2OverPi);


        Matrix tanhInner = inner.applyFunction(Math::tanh);


        Matrix result = x.multiply(0.5).multiply(tanhInner.add(1));

        return result;
    }
}
