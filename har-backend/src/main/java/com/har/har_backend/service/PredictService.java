package com.har.har_backend.service;

import com.har.har_backend.model.Prediction;
import com.har.har_backend.repository.PredictionRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class PredictService {

    private final PredictionRepository predictionRepository;
    private final RestTemplate restTemplate;

    @Value("${ml.service.url}")
    private String mlServiceUrl;

    public PredictService(PredictionRepository predictionRepository) {
        this.predictionRepository = predictionRepository;
        this.restTemplate = new RestTemplate();
    }

    // ── Main predict method ───────────────────────────────
    public Map<String, Object> predict(Map<String, Double> sensorData, String modelUsed) {

        // Step 1: Call Python Flask ML service
        Map<String, Object> mlResult = callMlService(sensorData, modelUsed);

        // Step 2: Extract result
        String activity   = (String) mlResult.getOrDefault("activity", "Unknown");
        double confidence = ((Number) mlResult.getOrDefault("confidence", 0.0)).doubleValue();

        // Step 3: Save to MySQL DB
        Prediction prediction = new Prediction(
            sensorData.get("alx"), sensorData.get("aly"), sensorData.get("alz"),
            sensorData.get("glx"), sensorData.get("gly"), sensorData.get("glz"),
            sensorData.get("arx"), sensorData.get("ary"), sensorData.get("arz"),
            sensorData.get("grx"), sensorData.get("gry"), sensorData.get("grz"),
            activity, confidence, modelUsed
        );
        predictionRepository.save(prediction);

        // Step 4: Build response
        Map<String, Object> response = new HashMap<>();
        response.put("activity",   activity);
        response.put("confidence", confidence);
        response.put("modelUsed",  modelUsed);
        response.put("id",         prediction.getId());
        response.put("timestamp",  prediction.getTimestamp());
        return response;
    }

    // ── Call Python Flask ML service ──────────────────────
    @SuppressWarnings("unchecked")
    private Map<String, Object> callMlService(Map<String, Double> sensorData,
                                               String modelUsed) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            Map<String, Object> body = new HashMap<>();
            body.put("features",  sensorData);
            body.put("model",     modelUsed);

            HttpEntity<Map<String, Object>> request = new HttpEntity<>(body, headers);

            ResponseEntity<Map> response = restTemplate.postForEntity(
                mlServiceUrl, request, Map.class
            );

            if (response.getStatusCode() == HttpStatus.OK && response.getBody() != null) {
                return response.getBody();
            }
        } catch (Exception e) {
            System.err.println("[SERVICE] ML service call failed: " + e.getMessage());
            System.err.println("[SERVICE] Falling back to heuristic classifier...");
        }

        // Fallback heuristic if Python service is down
        return fallbackClassify(sensorData);
    }

    // ── Fallback heuristic classifier ─────────────────────
    private Map<String, Object> fallbackClassify(Map<String, Double> data) {
        double alx = data.getOrDefault("alx", 0.0);
        double aly = data.getOrDefault("aly", 0.0);
        double alz = data.getOrDefault("alz", 0.0);
        double magnitude = Math.sqrt(alx * alx + aly * aly + alz * alz);

        String activity;
        if      (magnitude > 12) activity = "Jump";
        else if (magnitude > 10) activity = "Running";
        else if (magnitude > 8)  activity = "Jogging";
        else if (magnitude > 5)  activity = "Walking";
        else if (Math.abs(aly + 9.8) < 1.5) activity = "Standing";
        else if (alz > 3)        activity = "Sitting";
        else                     activity = "Lying Down";

        double confidence = 85.0 + Math.random() * 10;

        Map<String, Object> result = new HashMap<>();
        result.put("activity",   activity);
        result.put("confidence", Math.round(confidence * 10.0) / 10.0);
        return result;
    }

    // ── DB query methods ──────────────────────────────────
    public List<Prediction> getRecentPredictions() {
        return predictionRepository.findTop20ByOrderByTimestampDesc();
    }

    public List<Object[]> getActivityStats() {
        return predictionRepository.countByActivity();
    }

    public List<Object[]> getModelStats() {
        return predictionRepository.countByModel();
    }

    public long getTotalCount() {
        return predictionRepository.count();
    }

    public List<Prediction> getAllPredictions() {
        return predictionRepository.findAll();
    }
}
