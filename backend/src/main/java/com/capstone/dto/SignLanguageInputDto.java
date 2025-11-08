package com.capstone.dto;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;

public class SignLanguageInputDto {
    @NotNull(message = "Sign language data cannot be null")
    @NotEmpty(message = "Sign language data cannot be empty")
    private String signLanguageData;

    @NotNull(message = "Recognition target cannot be null")
    @NotEmpty(message = "Recognition target cannot be empty")
    private String recognitionTarget; // "DEPARTURE" or "ARRIVAL"

    public String getSignLanguageData() {
        return signLanguageData;
    }

    public void setSignLanguageData(String signLanguageData) {
        this.signLanguageData = signLanguageData;
    }

    public String getRecognitionTarget() {
        return recognitionTarget;
    }

    public void setRecognitionTarget(String recognitionTarget) {
        this.recognitionTarget = recognitionTarget;
    }
}
