package com.example.gpt;

public class GPTServer {
    public static void main(String[] args) {

        port(8080);           // 🟢 MUST come first
        ipAddress("0.0.0.0"); // 🟢 Right after

        GPTService service = new GPTService();
        service.initModel();

        new GPTController(service);

        System.out.println("[GPTServer] Listening on http://0.0.0.0:8080/generate");
    }
}
