package com.capstone.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;

import java.time.LocalDateTime;

@Entity
@Table(name = "bookings")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Booking {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(nullable = false)
    private String bookingNumber;
    
    @Column(nullable = false)
    private String trainNumber;
    
    @Column(nullable = false)
    private String departureStation;
    
    @Column(nullable = false)
    private String arrivalStation;
    
    @Column(nullable = false)
    private LocalDateTime departureTime;
    
    @Column(nullable = false)
    private LocalDateTime arrivalTime;
    
    @Column(nullable = false)
    private Integer passengers;
    
    @Column(nullable = false)
    private String seatType;
    
    @Column(nullable = false)
    private Integer totalPrice;
    
    @Column(nullable = false)
    private String paymentMethod;
    
    @Column(nullable = false)
    private String status; // "pending", "confirmed", "cancelled"
    
    @Column(nullable = false)
    private String tripType;
    
    @Column(nullable = false)
    private LocalDateTime createdAt;
    
    @Column
    private LocalDateTime updatedAt;
    
    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }
    
    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
