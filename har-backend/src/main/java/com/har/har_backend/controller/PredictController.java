package com.har.har_backend.controller;

import com.har.har_backend.model.Prediction;
import com.har.har_backend.service.PredictService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")   // allows frontend to call this API
public class PredictController {

    private final PredictService predictService;

    public PredictController(PredictService predictService) {
        this.predictService = predictService;
    }

    // ── POST /api/predict ─────────────────────────────────
    // Frontend sends sensor values → returns predicted activity
    @PostMapping("/predict")
    public ResponseEntity<Map<String, Object>> predict(
            @RequestBody Map<String, Object> requestBody) {

        try {
            // Extract sensor values
            @SuppressWarnings("unchecked")
            Map<String, Double> features = (Map<String, Double>)
                    requestBody.getOrDefault("features", new HashMap<>());

            String modelUsed = (String)
                    requestBody.getOrDefault("model", "1D CNN");

            // Run prediction + save to DB
            Map<String, Object> result = predictService.predict(features, modelUsed);

            return ResponseEntity.ok(result);

        } catch (Exception e) {
            Map<String, Object> error = new HashMap<>();
            error.put("error", "Prediction failed: " + e.getMessage());
            return ResponseEntity.internalServerError().body(error);
        }
    }

    // ── GET /api/history ──────────────────────────────────
    // Returns last 20 predictions for live feed
    @GetMapping("/history")
    public ResponseEntity<List<Prediction>> getHistory() {
        return ResponseEntity.ok(predictService.getRecentPredictions());
    }

    // ── GET /api/stats ────────────────────────────────────
    // Returns activity counts + model usage + total count
    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        Map<String, Object> stats = new HashMap<>();

        // Activity breakdown
        Map<String, Long> activityCounts = new HashMap<>();
        for (Object[] row : predictService.getActivityStats()) {
            activityCounts.put((String) row[0], (Long) row[1]);
        }

        // Model breakdown
        Map<String, Long> modelCounts = new HashMap<>();
        for (Object[] row : predictService.getModelStats()) {
            modelCounts.put((String) row[0], (Long) row[1]);
        }

        stats.put("totalPredictions", predictService.getTotalCount());
        stats.put("activityBreakdown", activityCounts);
        stats.put("modelBreakdown",    modelCounts);

        return ResponseEntity.ok(stats);
    }

    // ── GET /api/health ───────────────────────────────────
    // Simple health check endpoint
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> status = new HashMap<>();
        status.put("status",  "UP");
        status.put("service", "HAR Backend");
        status.put("total_predictions", predictService.getTotalCount());
        return ResponseEntity.ok(status);
    }
}