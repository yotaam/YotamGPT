package com.example.gpt;

import static spark.Spark.*;
import com.google.gson.Gson;

public class GPTController {

    private final GPTService gptService;
    private final Gson gson = new Gson();

    public GPTController(GPTService service) {
        this.gptService = service;
        initRoutes();
    }

    private void initRoutes() {

        post("/generate", (req, res) -> {
            res.type("application/json");

            GenerateRequest requestBody = gson.fromJson(req.body(), GenerateRequest.class);
            if (requestBody.prompt == null || requestBody.prompt.isEmpty()) {
                res.status(400);
                return gson.toJson(new ErrorResponse("Missing 'prompt' in JSON."));
            }

            // Use defaults if fields are not provided
            int maxTokens    = (requestBody.maxTokens    != null) ? requestBody.maxTokens    : 50;
            double temperature = (requestBody.temperature != null) ? requestBody.temperature : 1.0;
            int topK         = (requestBody.topK         != null) ? requestBody.topK         : 50;

            // Logging start time
            long startTime = System.nanoTime();

            // Generate
            String generatedText = gptService.generateText(
                requestBody.prompt,
                maxTokens,
                temperature,
                topK
            );

            // End time
            long endTime = System.nanoTime();
            double elapsedSeconds = (endTime - startTime) / 1e9;

            // Return JSON
            GenerateResponse responseBody = new GenerateResponse(
                generatedText,
                elapsedSeconds
            );

            return gson.toJson(responseBody);
        });

        // You could add a second endpoint like /evaluate, etc., here
    }

    // Data classes
    private static class GenerateRequest {
        String prompt;
        Integer maxTokens;
        Double temperature;
        Integer topK;
    }

    private static class GenerateResponse {
        String generatedText;
        double inferenceTimeSeconds;

        GenerateResponse(String generatedText, double inferenceTime) {
            this.generatedText = generatedText;
            this.inferenceTimeSeconds = inferenceTime;
        }
    }

    private static class ErrorResponse {
        String error;
        ErrorResponse(String msg) {
            this.error = msg;
        }
    }
}
