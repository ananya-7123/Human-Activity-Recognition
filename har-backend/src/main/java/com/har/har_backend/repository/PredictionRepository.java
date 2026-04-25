package com.har.har_backend.repository;

import com.har.har_backend.model.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    // Get last 20 predictions (for live feed)
    List<Prediction> findTop20ByOrderByTimestampDesc();

    // Get all predictions by model used
    List<Prediction> findByModelUsed(String modelUsed);

    // Get all predictions by activity
    List<Prediction> findByPredictedActivity(String activity);

    // Count predictions per activity (for pie chart)
    @Query("SELECT p.predictedActivity, COUNT(p) FROM Prediction p GROUP BY p.predictedActivity")
    List<Object[]> countByActivity();

    // Count predictions per model
    @Query("SELECT p.modelUsed, COUNT(p) FROM Prediction p GROUP BY p.modelUsed")
    List<Object[]> countByModel();

    // Total prediction count
    long count();
}