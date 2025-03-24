package com.example.gpt;

import java.util.*;
import java.io.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GPTService {

    private GPTModel model;
    private BytePairEncoding.Encoder tokenizer;

    // hyperparameters to play with!
    private final int vocabSize = 50257;   // GPT-2 vocabulary size - keep unless messing around with different pretrained weights
    private final int embDim    = 768;     // GPT-2 embedding dim, "" ""
    private final int contextLength = 1024;
    private final int numLayers = 12;
    private final int numHeads  = 12;
    private final double dropoutRate = 0.0;  // Usually 0 for inference

    private boolean isInitialized = false;

    public GPTService() {
        // Constructor can be empty, real init happens in initModel()
    }

    /**
     * Loads weights and initializes GPTModel + BPE tokenizer.
     * Call this once at server start.
     */
    public void initModel() {
        if (isInitialized) {
            return; // no-op if already loaded
        }

        System.out.println("[GPTService] Loading model weights and tokenizer...");

        try {
            // 1) Load the model weights
            Map<String, Object> weightsMap = WeightsLoader.loadWeights("gpt2_weights.json");

            // 2) Create the GPTModel
            this.model = new GPTModel(
                vocabSize, 
                embDim, 
                contextLength, 
                numLayers, 
                numHeads, 
                dropoutRate
            );

            // 3) Load weights into the model
            this.model.loadWeights(weightsMap);
            System.out.println("[GPTService] Model weights loaded successfully!");

            // 4) Initialize BytePairEncoding
            this.tokenizer = BytePairEncoding.getEncoder("gpt2", "models");
            System.out.println("[GPTService] BPE tokenizer loaded successfully!");

            isInitialized = true;

        } catch (IOException e) {
            throw new RuntimeException("Error loading model files: " + e.getMessage(), e);
        } catch (Exception e) {
            throw new RuntimeException("Unexpected error during model init: " + e.getMessage(), e);
        }
    }

    /**
     * Generate text given a prompt, using the loaded GPTModel.
     *
     * @param prompt       initial text prompt
     * @param maxTokens    how many new tokens to generate
     * @param temperature  temperature scaling
     * @param topK         top-K sampling
     * @return generated text
     */
    public String generateText(String prompt, int maxTokens, double temperature, int topK) {
        if (!isInitialized) {
            throw new IllegalStateException("Model not initialized. Call initModel() first.");
        }

        // 1) Encode the prompt
        List<Integer> inputTokenIndicesList = tokenizer.encode(prompt);
        int[] inputTokenIndices = inputTokenIndicesList
            .stream().mapToInt(Integer::intValue).toArray();

        // 2) Generate new tokens
        List<Integer> generatedTokenIndices = new ArrayList<>();

        for (int i = 0; i < maxTokens; i++) {
            // Combine prompt + previously generated tokens
            int[] currentInput = concatenateArrays(inputTokenIndices, generatedTokenIndices);

            // Enforce context length
            if (currentInput.length > contextLength) {
                System.err.println("Warning: Input exceeded model's context length, truncating.");
                currentInput = Arrays.copyOfRange(
                    currentInput, 
                    currentInput.length - contextLength, 
                    currentInput.length
                );
            }

            // Run forward pass
            Matrix logits = model.forward(currentInput);

            // Take the logits for the last token
            double[] lastLogits = logits.getRow(logits.getRows() - 1);

            // Apply temperature
            for (int j = 0; j < lastLogits.length; j++) {
                lastLogits[j] /= temperature;
            }

            // Softmax
            double[] probabilities = softmax(lastLogits);

            // Top-K sample
            int nextToken = sampleFromTopN(probabilities, topK);

            // Append next token
            generatedTokenIndices.add(nextToken);
        }

        // 3) Decode
        List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
        allTokenIndices.addAll(generatedTokenIndices);
        String output = tokenizer.decode(allTokenIndices);

        return output;
    }

    // -- Helpers --

    private int[] concatenateArrays(int[] array1, List<Integer> list2) {
        int[] array2 = list2.stream().mapToInt(Integer::intValue).toArray();
        int[] result = new int[array1.length + array2.length];
        System.arraycopy(array1, 0, result, 0, array1.length);
        System.arraycopy(array2, 0, result, array1.length, array2.length);
        return result;
    }

    private double[] softmax(double[] logits) {
        double maxLogit = Arrays.stream(logits).max().orElse(0.0);
        double sumExp = 0.0;
        double[] expLogits = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            expLogits[i] = Math.exp(logits[i] - maxLogit);
            sumExp += expLogits[i];
        }
        for (int i = 0; i < logits.length; i++) {
            expLogits[i] /= sumExp;
        }
        return expLogits;
    }

    private int sampleFromTopN(double[] probabilities, int topN) {
        // Sort token IDs by descending probability
        List<Integer> indices = IntStream.range(0, probabilities.length)
                                         .boxed()
                                         .collect(Collectors.toList());
        indices.sort((i, j) -> Double.compare(probabilities[j], probabilities[i]));

        // Keep topN
        List<Integer> topIndices = indices.subList(0, Math.min(topN, indices.size()));
        double[] topProbabilities = topIndices.stream().mapToDouble(i -> probabilities[i]).toArray();

        // Normalize
        double sum = Arrays.stream(topProbabilities).sum();
        for (int i = 0; i < topProbabilities.length; i++) {
            topProbabilities[i] /= sum;
        }

        // Random sample
        double r = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < topIndices.size(); i++) {
            cumulative += topProbabilities[i];
            if (r < cumulative) {
                return topIndices.get(i);
            }
        }
        // fallback
        return topIndices.get(topIndices.size() - 1);
    }
}
