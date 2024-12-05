public class TransformerBlock {
    public final LayerNorm norm1;
    public final MultiHeadAttention attention;
    public final LayerNorm norm2;
    public final FeedForward feedForward;

    public TransformerBlock(int embDim, int numHeads, int contextLength, double dropoutRate) {
        this.norm1 = new LayerNorm(embDim);
        this.attention = new MultiHeadAttention(embDim, embDim, contextLength, dropoutRate, numHeads);
        this.norm2 = new LayerNorm(embDim);
        this.feedForward = new FeedForward(embDim);
    }

    public Matrix forward(Matrix x) {

        Matrix normed1 = norm1.forward(x); 
        Matrix attentionOut = attention.forward(normed1); 
        Matrix add1 = x.add(attentionOut); 
        Matrix normed2 = norm2.forward(add1); 
        Matrix feedForwardOut = feedForward.forward(normed2); 
        Matrix output = add1.add(feedForwardOut); 

        return output; 
    }
}
