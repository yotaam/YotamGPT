public class Linear {
    private final Matrix weight;
    private final Matrix bias;   

    public Linear(int inFeatures, int outFeatures) {
        this.weight = Matrix.random(inFeatures, outFeatures, 0.0, 0.02);
        this.bias = new Matrix(1, outFeatures);
    }
    public void setWeights(double[][] weightData) {
        this.weight.setData(weightData);
    }
    
    public void setBias(double[] biasData) {
        if (biasData.length != this.bias.getCols()) {
            throw new IllegalArgumentException("Bias dimensions do not match.");
        }
        this.bias.setData(new double[][] { biasData });
    }
    
    public Matrix forward(Matrix input) {
        Matrix output = input.matMul(this.weight);
        output = output.addRowVector(this.bias);   
        return output;
    }
}
