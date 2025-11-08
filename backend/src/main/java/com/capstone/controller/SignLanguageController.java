package com.capstone.controller;

import com.capstone.dto.CityRecognitionResponseDto;
import com.capstone.dto.DateTimeRecognitionResponseDto;
import com.capstone.dto.PassengerRecognitionResponseDto;
import com.capstone.dto.SeatClassRecognitionResponseDto;
import com.capstone.dto.SignLanguageInputDto;
import com.capstone.dto.TripTypeRecognitionResponseDto;
import com.capstone.service.SignLanguageService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/signlanguage")
public class SignLanguageController {

    private static final Logger logger = LoggerFactory.getLogger(SignLanguageController.class);
    private final SignLanguageService signLanguageService;

    @Autowired
    public SignLanguageController(SignLanguageService signLanguageService) {
        this.signLanguageService = signLanguageService;
    }

    @PostMapping("/recognize/city")
    public CityRecognitionResponseDto recognizeCity(@Valid @RequestBody SignLanguageInputDto signLanguageInputDto) {
        logger.info("Received sign language recognition request: {}", signLanguageInputDto.getSignLanguageData());
        CityRecognitionResponseDto response = signLanguageService.recognizeCity(signLanguageInputDto);
        logger.info("Returning city recognition response: Departure={}, Arrival={}", response.getDepartureCity(), response.getArrivalCity());
        return response;
    }

    @PostMapping("/recognize/datetime")
    public DateTimeRecognitionResponseDto recognizeDateTime(@Valid @RequestBody SignLanguageInputDto signLanguageInputDto) {
        logger.info("날짜/시간 수어 인식 요청 수신: {}", signLanguageInputDto.getSignLanguageData());
        DateTimeRecognitionResponseDto response = signLanguageService.recognizeDateTime(signLanguageInputDto);
        logger.info("더미 날짜/시간 인식 응답 반환: 날짜={}, 시간={}", response.getRecognizedDate(), response.getRecognizedTime());
        return response;
    }

    @PostMapping("/recognize/passengers")
    public PassengerRecognitionResponseDto recognizePassengers(@Valid @RequestBody SignLanguageInputDto signLanguageInputDto) {
        logger.info("승객 수 수어 인식 요청 수신: {}", signLanguageInputDto.getSignLanguageData());
        PassengerRecognitionResponseDto response = signLanguageService.recognizePassengers(signLanguageInputDto);
        logger.info("더미 승객 수 인식 응답 반환: 승객 수={}", response.getRecognizedPassengers());
        return response;
    }

    @PostMapping("/recognize/triptype")
    public TripTypeRecognitionResponseDto recognizeTripType(@Valid @RequestBody SignLanguageInputDto signLanguageInputDto) {
        logger.info("편도/왕복 수어 인식 요청 수신: {}", signLanguageInputDto.getSignLanguageData());
        TripTypeRecognitionResponseDto response = signLanguageService.recognizeTripType(signLanguageInputDto);
        logger.info("더미 편도/왕복 인식 응답 반환: 편도/왕복 여부={}", response.getRecognizedTripType());
        return response;
    }

    @PostMapping("/recognize/seatclass")
    public SeatClassRecognitionResponseDto recognizeSeatClass(@Valid @RequestBody SignLanguageInputDto signLanguageInputDto) {
        logger.info("좌석 등급 수어 인식 요청 수신: {}", signLanguageInputDto.getSignLanguageData());
        SeatClassRecognitionResponseDto response = signLanguageService.recognizeSeatClass(signLanguageInputDto);
        logger.info("더미 좌석 등급 인식 응답 반환: 좌석 등급={}", response.getRecognizedSeatClass());
        return response;
    }
}
