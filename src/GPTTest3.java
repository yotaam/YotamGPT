import java.util.*;
import java.io.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GPTTest3 {
    public static void main(String[] args) {
        // Display the JVM's max heap size
        System.out.println("Max Heap Size: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + " MB");

        try {
            // Step 1: Load the weights map
            System.out.println("Loading model weights...");
            Map<String, Object> weightsMap = WeightsLoader.loadWeights("gpt2_weights.json");

            // Step 2: Initialize your GPTModel
            int vocabSize = 50257; // GPT-2 uses 50257 tokens
            int embDim = 768;      // For GPT-2 small
            int contextLength = 1024;
            int numLayers = 12;
            int numHeads = 12;
            double dropoutRate = 0.0;

            GPTModel model = new GPTModel(vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate);

            // Step 3: Load weights into the model
            System.out.println("Loading weights into the model...");
            model.loadWeights(weightsMap);
            System.out.println("Model weights loaded successfully!");

            // Step 4: Initialize the BytePairEncoding tokenizer
            System.out.println("Initializing BytePairEncoding tokenizer...");
            BytePairEncoding.Encoder encoder = BytePairEncoding.getEncoder("gpt2", "models");

            // Define the list of prompts
            List<String> prompts = Arrays.asList(
                "Once upon a time, in a land far, far away, there lived a",
                "To be or not to be, that is the",
                "The capital of France is",
                "In 1492, Columbus sailed the ocean",
                "def quicksort(arr):",
                "The meaning of life is",
                "It was the best of times, it was the",
                "She looked at him and said,",
                "E = mc",
                "The mitochondria is the powerhouse of the",
                "The quick brown fox jumps over the",
                "As I walked through the valley of the shadow of death, I",
                "In conclusion,",
                "for i in range(10):",
                "Roses are red, violets are",
                "The stock market crashed today due to",
                "The first law of thermodynamics states that energy cannot be",
                "Knock, knock.",
                "The Pythagorean theorem states that in a right triangle,",
                "In a hole in the ground there lived a"
            );

            // Number of tokens to generate for each prompt
            int numTokensToGenerate = 25; // Adjust as needed

            // Prepare the output file
            String outputFileName = "results2.txt";
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));

            // Define the temperature values for regularization
            double[] temperatures = {.5, .75}; // Higher temperatures flatten the distribution

            // Loop over each prompt
            for (String prompt : prompts) {
                System.out.println("Processing prompt: " + prompt);
                // Encode the prompt
                List<Integer> inputTokenIndicesList = encoder.encode(prompt);
                int[] inputTokenIndices = inputTokenIndicesList.stream().mapToInt(Integer::intValue).toArray();

                // Loop over each temperature value
                for (double temperature : temperatures) {
                    System.out.println("  Using Temperature = " + temperature);
                    // Initialize the list to hold generated tokens
                    List<Integer> generatedTokenIndices = new ArrayList<>();

                    // Record start time
                    long startTime = System.nanoTime();

                    // Generate tokens
                    for (int i = 0; i < numTokensToGenerate; i++) {
                        // Get the current input tokens
                        int[] currentInput = concatenateArrays(inputTokenIndices, generatedTokenIndices);

                        // Ensure the input does not exceed context length
                        if (currentInput.length > contextLength) {
                            throw new IllegalArgumentException("Input exceeds the model's context length.");
                        }

                        // Run the model
                        Matrix logits = model.forward(currentInput);

                        // Get the logits for the last token
                        double[] lastLogits = logits.getRow(logits.getRows() - 1);

                        // Apply temperature scaling to logits
                        for (int j = 0; j < lastLogits.length; j++) {
                            lastLogits[j] /= temperature;
                        }

                        // Convert logits to probabilities using softmax
                        double[] probabilities = softmax(lastLogits);

                        // Sample the next token from the top K probabilities
                        int K = 100; // Fixed K=100 as per instructions
                        int nextToken = sampleFromTopN(probabilities, K);

                        // Add the generated token to the list
                        generatedTokenIndices.add(nextToken);
                    }

                    // Record end time
                    long endTime = System.nanoTime();
                    long elapsedTimeMillis = (endTime - startTime) / 1_000_000;

                    // Decode the generated tokens back into text
                    List<Integer> allTokenIndices = new ArrayList<>(inputTokenIndicesList);
                    allTokenIndices.addAll(generatedTokenIndices);
                    String generatedText = encoder.decode(allTokenIndices);

                    // Prepare the output log
                    String logEntry = "Prompt: " + prompt + "\n" +
                                      "Temperature: " + temperature + "\n" +
                                      "Time Taken: " + elapsedTimeMillis + " ms\n" +
                                      "Generated Text:\n" + generatedText + "\n" +
                                      "----------------------------------------\n";

                    // Write the log entry to the output file
                    writer.write(logEntry);
                    writer.flush(); // Ensure data is written to file

                    // Optional: Print status
                    System.out.println("    Finished Temperature = " + temperature + " in " + elapsedTimeMillis + " ms");
                }
            }

            // Close the writer
            writer.close();
            System.out.println("Results saved to " + outputFileName);

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
}
