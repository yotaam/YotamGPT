package com.example.gpt;

import static spark.Spark.*;

public class GPTServer {
    public static void main(String[] args) {

        port(8080);           // 🟢 MUST come first
        ipAddress("0.0.0.0"); // 🟢 Right after

        // 🔓 CORS setup
        options("/*", (request, response) -> {
            String accessControlRequestHeaders = request.headers("Access-Control-Request-Headers");
            if (accessControlRequestHeaders != null) {
                response.header("Access-Control-Allow-Headers", accessControlRequestHeaders);
            }

            String accessControlRequestMethod = request.headers("Access-Control-Request-Method");
            if (accessControlRequestMethod != null) {
                response.header("Access-Control-Allow-Methods", accessControlRequestMethod);
            }

            return "OK";
        });

        before((request, response) -> {
            response.header("Access-Control-Allow-Origin", "*");
            response.header("Access-Control-Request-Method", "*");
            response.header("Access-Control-Allow-Headers", "*");
            response.type("application/json");
        });

        // ✅ Init model and controller
        GPTService service = new GPTService();
        service.initModel();

        new GPTController(service);

        System.out.println("[GPTServer] Listening on http://0.0.0.0:8080/generate");
    }
}
