package com.capstone.service;

import com.capstone.dto.CityRecognitionResponseDto;
import com.capstone.dto.DateTimeRecognitionResponseDto;
import com.capstone.dto.PassengerRecognitionResponseDto;
import com.capstone.dto.SeatClassRecognitionResponseDto; // 추가
import com.capstone.dto.SignLanguageInputDto;
import com.capstone.dto.TripTypeRecognitionResponseDto; // 추가
import com.capstone.dto.ErrorResponseDto;  //오류 코드 처리를 위해 추가한 Dto
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//추론 서버와 통신을 위한 추가 import
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;
import org.springframework.web.client.ResourceAccessException;
import org.springframework.web.client.RestTemplate;
//오류코드 처리용
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class SignLanguageService {

    private static final Logger logger = LoggerFactory.getLogger(SignLanguageService.class);
    // Flask 서버와 HTTP 통신을 하기위한 RestTemplate
    private RestTemplate restTemplate;
    // ObjectMapper는 final 필드로 선언하고 생성자나 선언 시점에 초기화합니다.
    private final ObjectMapper objectMapper = new ObjectMapper();
    // Flask 서버의 CSV 엔드포인트
    private final String flaskUrl = "http://localhost:5000/predict_csv";

    //생성자 주입(RestTemplate)
    public SignLanguageService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public CityRecognitionResponseDto recognizeCity_with_AI(SignLanguageInputDto signLanguageInputDto, String csvFile) {
        logger.info("서비스에서 수어 데이터를 처리 중입니다: {}", signLanguageInputDto.getSignLanguageData());
        logger.info("인식 대상: {}", signLanguageInputDto.getRecognitionTarget());

        CityRecognitionResponseDto response = new CityRecognitionResponseDto();

        // 1. http 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        // 핵심: 컨텐트 타입을 MULTIPART_FORM_DATA로 설정(csv 파일 전송을 위함)
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        // 2. 요청 본문(Body)에 출발지, 도착지 구분과 파일 데이터 추가
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("recognitionTarget", signLanguageInputDto.getRecognitionTarget());
        // 핵심: Flask에서 'csv_file'로 접근할 수 있도록 key를 설정합니다.
        body.add("csv_file", new FileSystemResource(csvFile));

        // 3. HttpEntity: 헤더와 본문을 결합
        HttpEntity<MultiValueMap<String, Object>> entity = new HttpEntity<>(body, headers);

        // 4. Flask 서버에 POST 요청 전송
        try {
            ResponseEntity<CityRecognitionResponseDto> server_response = restTemplate.exchange(
                    flaskUrl,
                    HttpMethod.POST,
                    entity,
                    CityRecognitionResponseDto.class // 응답을 Dto (JSON)으로 받음
            );

            response = server_response.getBody();
        } catch (HttpClientErrorException e) {
            // 4xx Client Error (예: 400 Bad Request) 처리
            String errorResponseJson = e.getResponseBodyAsString();
            logger.warn("클라이언트 오류 발생. Code={}, JSON={}", e.getStatusCode(), errorResponseJson);
            handleErrorResponse(response, e.getStatusCode().value(), errorResponseJson);

        } catch (HttpServerErrorException e){
            // 5xx Server Error (예: 500 Internal Server Error) 처리
            String errorResponseJson = e.getResponseBodyAsString();
            logger.error("서버 오류 발생. Code={}, JSON={}", e.getStatusCode(), errorResponseJson);
            handleErrorResponse(response, e.getStatusCode().value(), errorResponseJson);

        } catch (ResourceAccessException e) {
            // 서버 연결 실패(서버 다운) 등 HTTP 통신 중 에러 처리
            logger.error("HTTP 통신 중 에러가 발생하였습니다. {}", e.getMessage());
            // 기본 오류 응답 설정
            response.setDepartureCity("연결 오류 발생");
            response.setArrivalCity("연결 오류 발생");
        } catch (Exception e) {
            // 기타 예상치 못한 예외 처리
            logger.error("예상치 못한 오류 발생: {}", e.getMessage(), e);
            response.setDepartureCity("알 수 없는 오류");
            response.setArrivalCity("알 수 없는 오류");
        }
        return response;
    }

    // 오류 JSON 파싱 및 로직 설정 헬퍼 메서드
    private void handleErrorResponse(CityRecognitionResponseDto response, int statusCode, String errorJson) {
        try {
            // 오류 JSON을 ErrorResponseDto로 변환 시도
            ErrorResponseDto errorDto = objectMapper.readValue(errorJson, ErrorResponseDto.class);
            logger.warn("오류 상세: [{}], {}", errorDto.getErrorCode(), errorDto.getErrorMessage());

            // 실제 API 오류 발생 시 알림이나 특정 로직을 수행할 수 있습니다.
            response.setDepartureCity("API 오류: " + statusCode);
            response.setArrivalCity("오류 원인: " + errorDto.getErrorMessage());

        } catch (Exception jsonEx) {
            // 오류 응답 JSON 형식이 ErrorResponseDto와 맞지 않을 때
            logger.error("오류 응답 본문 파싱 실패: 원본 JSON = {}", errorJson);
            response.setDepartureCity("알 수 없는 오류");
            response.setArrivalCity("알 수 없는 오류");
        }
    }

    public CityRecognitionResponseDto recognizeCity(SignLanguageInputDto signLanguageInputDto) {
        logger.info("서비스에서 수어 데이터를 처리 중입니다: {}", signLanguageInputDto.getSignLanguageData());
        logger.info("인식 대상: {}", signLanguageInputDto.getRecognitionTarget());

        CityRecognitionResponseDto response = new CityRecognitionResponseDto();

        // 현재는 recognitionTarget을 기반으로 더미 데이터를 반환합니다.
        if ("DEPARTURE".equalsIgnoreCase(signLanguageInputDto.getRecognitionTarget())) {
            response.setDepartureCity("서울"); // 출발지에 대한 예시 데이터
            response.setArrivalCity(null); // 이번 단계에서는 도착 도시를 인식하지 않습니다.
        } else if ("ARRIVAL".equalsIgnoreCase(signLanguageInputDto.getRecognitionTarget())) {
            response.setDepartureCity(null); // 이번 단계에서는 출발 도시를 인식하지 않습니다.
            response.setArrivalCity("부산");   // 도착지에 대한 예시 데이터
        } else {
            // 유효하지 않은 recognitionTarget을 처리하거나 기본값으로 설정합니다.
            response.setDepartureCity("알 수 없는 출발지");
            response.setArrivalCity("알 수 없는 도착지");
            logger.warn("유효하지 않은 인식 대상이 수신되었습니다: {}", signLanguageInputDto.getRecognitionTarget());
        }

        logger.info("더미 도시 인식 응답 생성됨: 출발지={}, 도착지={}", response.getDepartureCity(), response.getArrivalCity());
        return response;
    }

    public DateTimeRecognitionResponseDto recognizeDateTime(SignLanguageInputDto signLanguageInputDto) {
        logger.info("서비스에서 수어 데이터를 처리 중입니다 (날짜/시간): {}", signLanguageInputDto.getSignLanguageData());

        DateTimeRecognitionResponseDto response = new DateTimeRecognitionResponseDto();
        // TODO: 실제 날짜/시간 수어 인식 로직을 여기에 구현하세요.
        // 현재는 더미 데이터를 반환합니다.
        response.setRecognizedDate("2025-11-05"); // 예시 날짜
        response.setRecognizedTime("14:30");      // 예시 시간

        logger.info("더미 날짜/시간 인식 응답 생성됨: 날짜={}, 시간={}", response.getRecognizedDate(), response.getRecognizedTime());
        return response;
    }

    public PassengerRecognitionResponseDto recognizePassengers(SignLanguageInputDto signLanguageInputDto) {
        logger.info("서비스에서 수어 데이터를 처리 중입니다 (승객 수): {}", signLanguageInputDto.getSignLanguageData());

        PassengerRecognitionResponseDto response = new PassengerRecognitionResponseDto();
        // TODO: 실제 승객 수 수어 인식 로직을 여기에 구현하세요.
        // 현재는 더미 데이터를 반환합니다.
        response.setRecognizedPassengers(2); // 예시 승객 수

        logger.info("더미 승객 수 인식 응답 생성됨: 승객 수={}", response.getRecognizedPassengers());
        return response;
    }

    // 새로운 메서드 추가
    public TripTypeRecognitionResponseDto recognizeTripType(SignLanguageInputDto signLanguageInputDto) {
        logger.info("서비스에서 수어 데이터를 처리 중입니다 (편도/왕복): {}", signLanguageInputDto.getSignLanguageData());

        TripTypeRecognitionResponseDto response = new TripTypeRecognitionResponseDto();
        // TODO: 실제 편도/왕복 수어 인식 로직을 여기에 구현하세요.
        // 현재는 더미 데이터를 반환합니다.
        response.setRecognizedTripType("왕복"); // 예시: 왕복 또는 편도

        logger.info("더미 편도/왕복 인식 응답 생성됨: 편도/왕복 여부={}", response.getRecognizedTripType());
        return response;
    }

    // 새로운 메서드 추가
    public SeatClassRecognitionResponseDto recognizeSeatClass(SignLanguageInputDto signLanguageInputDto) {
        logger.info("서비스에서 수어 데이터를 처리 중입니다 (좌석 등급): {}", signLanguageInputDto.getSignLanguageData());

        SeatClassRecognitionResponseDto response = new SeatClassRecognitionResponseDto();
        // TODO: 실제 좌석 등급 수어 인식 로직을 여기에 구현하세요.
        // 현재는 더미 데이터를 반환합니다.
        response.setRecognizedSeatClass("일반실"); // 예시: 일반실, 특실

        logger.info("더미 좌석 등급 인식 응답 생성됨: 좌석 등급={}", response.getRecognizedSeatClass());
        return response;
    }
}
