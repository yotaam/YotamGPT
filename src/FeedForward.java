public class FeedForward {
    public final Linear fc1; 
    public final Linear fc2; 

    public FeedForward(int embDim) {
        this.fc1 = new Linear(embDim, 4 * embDim);
        this.fc2 = new Linear(4 * embDim, embDim);
    }

    public Matrix forward(Matrix x) {
        Matrix out = fc1.forward(x);     
        out = GELU.forward(out);        
        out = fc2.forward(out);      
        return out;
    }
}
