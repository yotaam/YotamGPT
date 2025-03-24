package com.example.gpt;

import java.util.*;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.Scanner;


public class GPTTest {
    public static void main(String[] args) {
        System.out.println("Max Heap Size: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB");

        try {
            System.out.println("Loading the weights");
            Map<String, Object> weightsMap = WeightsLoader.loadWeights("gpt2_weights.json");
            // specs for GPT-2
            int vocabSize = 50257; 
            int embDim = 768;
            int contextLength = 1024;
            int numLayers = 12;
            int numHeads = 12;
            double dropoutRate = 0.0;

            GPTModel model = new GPTModel(vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate);
            model.loadWeights(weightsMap);
            System.out.println("Weights loaded");
            BytePairEncoding.Encoder encoder = BytePairEncoding.getEncoder("gpt2", "models");
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("Prompt: ");
                String inputText = scanner.nextLine();
                System.out.print("# tokens: ");
                String TokensToGenerate = scanner.nextLine(); 
                int numTokensToGenerate;
                try {
                    numTokensToGenerate = Integer.parseInt(TokensToGenerate);
                } catch (NumberFormatException e) {
                    System.out.println("Invalid input. Going with 10.");
                    numTokensToGenerate = 10;
                }
                System.out.println("Input: " + inputText);
                List<Integer> inputTokenIndicesList = encoder.encode(inputText);
                int[] inputTokenIndices = inputTokenIndicesList.stream().mapToInt(Integer::intValue).toArray();
                System.out.println("Producing text:");
                List<Integer> generatedTokenIndices = new ArrayList<>();
                for (int i = 0; i < numTokensToGenerate; i++) {
                    long startTime = System.nanoTime();
                    int[] currentInput = concatenateArrays(inputTokenIndices, generatedTokenIndices);
                    if (currentInput.length > contextLength) {
                        throw new IllegalArgumentException("Input exceeds the model's context length.");
                    }
                    Matrix logits = model.forward(currentInput);
                    double[] lastLogits = logits.getRow(logits.getRows() - 1);
                    double[] probabilities = softmax(lastLogits);
                
                    //deterministic probs
                    /*
                    int nextToken = IntStream.range(0, probabilities.length)
                        .reduce((index1, index2) -> probabilities[index1] > probabilities[index2] ? index1 : index2)
                        .orElse(-1);
                    */
                    //stochastic
                    int nextToken = sampleFromTopN(probabilities, 4);

            
                    generatedTokenIndices.add(nextToken);


                    List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
                    allTokenIndices.addAll(generatedTokenIndices);
                    String currentOutput = encoder.decode(allTokenIndices);
                    System.out.println(currentOutput + "\r");
                    long endTime = System.nanoTime();
                    //long elapsedTime = (endTime - startTime) / 1_000_000;
                }
                System.out.println();
                List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
                allTokenIndices.addAll(generatedTokenIndices);
                String generatedText = encoder.decode(allTokenIndices);
                System.out.println("Final Text:");
                System.out.println(generatedText);
                
            }
        } catch (OutOfMemoryError e) {
            System.err.println("Out of memory error! Consider increasing the heap size.");
        } catch (IOException e) {
            System.err.println("File error while loading model or weights: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            System.err.println("Illegal argument error: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("An unexpected error occurred:");
            e.printStackTrace();
        }
    }

    // Helper method to concatenate two arrays
    private static int[] concatenateArrays(int[] array1, List<Integer> list2) {
        int[] array2 = list2.stream().mapToInt(Integer::intValue).toArray();
        int[] result = new int[array1.length + array2.length];
        System.arraycopy(array1, 0, result, 0, array1.length);
        System.arraycopy(array2, 0, result, array1.length, array2.length);
        return result;
    }

    // Helper method to compute softmax
    private static double[] softmax(double[] logits) {
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
    // Helper function to sample from the top N probabilities
    private static int sampleFromTopN(double[] probabilities, int topN) {
        // Create a list of indices and their associated probabilities
        List<Integer> indices = IntStream.range(0, probabilities.length).boxed().collect(Collectors.toList());
        indices.sort((i, j) -> Double.compare(probabilities[j], probabilities[i])); // Sort descending by probability

        // Retain only the top N indices
        List<Integer> topIndices = indices.subList(0, Math.min(topN, indices.size()));
        double[] topProbabilities = topIndices.stream().mapToDouble(i -> probabilities[i]).toArray();

        // Normalize the top probabilities
        double sum = Arrays.stream(topProbabilities).sum();
        for (int i = 0; i < topProbabilities.length; i++) {
            topProbabilities[i] /= sum;
        }

        // Sample from the top N indices using the normalized probabilities
        double r = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < topIndices.size(); i++) {
            cumulative += topProbabilities[i];
            if (r < cumulative) {
                return topIndices.get(i);
            }
        }

        // Fallback (should not reach here)
        return topIndices.get(topIndices.size() - 1);
    }
/*
    // Helper method to sample an index from a probability distribution
    private static int sampleFromDistribution(double[] probabilities, double temperature, int topK) {
        // Apply temperature
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = Math.pow(probabilities[i], 1.0 / temperature);
        }
    
        // Normalize probabilities after applying temperature
        double sum = Arrays.stream(probabilities).sum();
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
    
        // Top-k filtering
        PriorityQueue<Integer> topKIndices = new PriorityQueue<>(
            Comparator.comparingDouble(i -> probabilities[i])
        );
        for (int i = 0; i < probabilities.length; i++) {
            if (topKIndices.size() < topK) {
                topKIndices.add(i);
            } else if (probabilities[i] > probabilities[topKIndices.peek()]) {
                topKIndices.poll();
                topKIndices.add(i);
            }
        }
    
        // Normalize over the top-k probabilities
        double[] filteredProbabilities = new double[probabilities.length];
        double filteredSum = 0.0;
        for (int idx : topKIndices) {
            filteredProbabilities[idx] = probabilities[idx];
            filteredSum += probabilities[idx];
        }
        for (int i = 0; i < filteredProbabilities.length; i++) {
            filteredProbabilities[i] /= filteredSum;
        }
    
        // Sample from the filtered probabilities
        double r = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < filteredProbabilities.length; i++) {
            cumulative += filteredProbabilities[i];
            if (r < cumulative) {
                return i;
            }
        }
    
        return probabilities.length - 1; // Fallback
    }
    */
}