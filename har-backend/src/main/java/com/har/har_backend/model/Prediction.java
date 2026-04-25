package com.har.har_backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "predictions")
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // ── Sensor inputs ─────────────────────────────────────
    private double alx;
    private double aly;
    private double alz;
    private double glx;
    private double gly;
    private double glz;
    private double arx;
    private double ary;
    private double arz;
    private double grx;
    private double gry;
    private double grz;

    // ── Prediction output ─────────────────────────────────
    private String predictedActivity;
    private double confidence;
    private String modelUsed;

    // ── Metadata ──────────────────────────────────────────
    private LocalDateTime timestamp;

    // ── Lifecycle ─────────────────────────────────────────
    @PrePersist
    public void prePersist() {
        this.timestamp = LocalDateTime.now();
    }

    // ── Constructors ──────────────────────────────────────
    public Prediction() {}

    public Prediction(double alx, double aly, double alz,
                      double glx, double gly, double glz,
                      double arx, double ary, double arz,
                      double grx, double gry, double grz,
                      String predictedActivity, double confidence,
                      String modelUsed) {
        this.alx = alx; this.aly = aly; this.alz = alz;
        this.glx = glx; this.gly = gly; this.glz = glz;
        this.arx = arx; this.ary = ary; this.arz = arz;
        this.grx = grx; this.gry = gry; this.grz = grz;
        this.predictedActivity = predictedActivity;
        this.confidence = confidence;
        this.modelUsed = modelUsed;
    }

    // ── Getters & Setters ─────────────────────────────────
    public Long getId() { return id; }

    public double getAlx() { return alx; }
    public void setAlx(double alx) { this.alx = alx; }

    public double getAly() { return aly; }
    public void setAly(double aly) { this.aly = aly; }

    public double getAlz() { return alz; }
    public void setAlz(double alz) { this.alz = alz; }

    public double getGlx() { return glx; }
    public void setGlx(double glx) { this.glx = glx; }

    public double getGly() { return gly; }
    public void setGly(double gly) { this.gly = gly; }

    public double getGlz() { return glz; }
    public void setGlz(double glz) { this.glz = glz; }

    public double getArx() { return arx; }
    public void setArx(double arx) { this.arx = arx; }

    public double getAry() { return ary; }
    public void setAry(double ary) { this.ary = ary; }

    public double getArz() { return arz; }
    public void setArz(double arz) { this.arz = arz; }

    public double getGrx() { return grx; }
    public void setGrx(double grx) { this.grx = grx; }

    public double getGry() { return gry; }
    public void setGry(double gry) { this.gry = gry; }

    public double getGrz() { return grz; }
    public void setGrz(double grz) { this.grz = grz; }

    public String getPredictedActivity() { return predictedActivity; }
    public void setPredictedActivity(String predictedActivity) { this.predictedActivity = predictedActivity; }

    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }

    public String getModelUsed() { return modelUsed; }
    public void setModelUsed(String modelUsed) { this.modelUsed = modelUsed; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
}