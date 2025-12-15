// SentinEV Frontend Types

// ==================== Vehicle Types ====================

export interface Vehicle {
    vehicle_id: string;
    driver_profile: 'normal' | 'aggressive' | 'eco';
    status: 'initialized' | 'loaded';
    training_samples: number;
    model_trained: boolean;
}

// ==================== Telemetry Types ====================

export interface TelemetryData {
    timestamp: string;
    vehicle_id: string;
    driver_profile: string;
    // Motion data
    speed_kmh: number;
    speed_kph: number;  // Alias for ML compatibility
    acceleration_ms2: number;
    jerk_ms3: number;
    // Motor data
    motor_rpm: number;
    motor_temp_c: number;
    inverter_temp_c: number;
    // Power data
    power_draw_kw: number;
    power_kw: number;  // Alias for ML
    net_power_kw: number;
    regen_power_kw: number;
    regen_efficiency: number;
    regen_pct: number;  // Percentage for ML
    // Battery data
    battery_soc_pct: number;
    battery_temp_c: number;
    battery_voltage_v: number;
    battery_current_a: number;
    cell_voltage_avg_v: number;
    cell_voltage_diff_v: number;
    battery_cell_delta_v: number;  // Alias for ML
    // Thermal data
    brake_temp_c: number;
    coolant_temp_c: number;
    ambient_temp_c: number;
    // HVAC
    hvac_power_kw: number;
    // Control inputs
    throttle_pct: number;
    brake_pct: number;
    // IMU/Accelerometer
    accel_x: number;
    accel_y: number;
    accel_z: number;
    // Wear data
    wear_index: number;
    odometer_km: number;
    // Fault indicators
    active_faults: string[];
    fault_count: number;
    // Anomaly labels (from backend)
    is_anomaly?: boolean;
    anomaly_type?: string;
}

export interface AnomalyData {
    is_anomaly: boolean;
    score: number;  // Anomaly score from ML models
    type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    failure_risk_pct: number;
}

export interface ScoringData {
    delta: number;
    total: number;
    feedback: string;
}

// ==================== Scenario Types ====================

export interface Scenario {
    id: string;
    name: string;
    description: string;
    duration_seconds: number;
    component: string;
    severity: string;
    requires_service: boolean;
}

export interface ScenarioState {
    phase: 'normal' | 'building' | 'warning' | 'critical' | null;
    event: string;
    description: string;
    complete?: boolean;
}

// ==================== Prediction Types ====================

export interface Prediction {
    prediction_id: string;
    vehicle_id: string;
    component: string;
    anomaly_type: string;
    severity: string;
    days_to_failure: number;
    message: string;
    requires_service: boolean;
    status: 'pending' | 'accepted' | 'rejected';
    created_at: string;
    actions?: string[];
    timeout_seconds?: number;
}

// ==================== Safety Agent Types ====================

export interface SafetyAdvice {
    type: 'safety_advice';
    prediction_id: string;
    title: string;
    message: string;
    tips: string[];
    points_awarded: number;
    severity: string;
    rag_sources: string[];
    timestamp: string;
}

// ==================== Diagnosis Types ====================

export interface DiagnosisStep {
    step: number;
    title: string;
    finding: string;
    status: 'ok' | 'warning' | 'critical' | 'pending';
    icon: string;
}

export interface DiagnosisResult {
    type: 'diagnosis';
    diagnosis_id: string;
    title: string;
    summary: string;
    steps: DiagnosisStep[];
    root_cause: string;
    repair_action: string;
    estimated_cost: string;
    urgency: 'immediate' | 'soon' | 'scheduled';
    service_required: boolean;
    rag_sources: string[];
    actions?: string[];
    timestamp: string;
}

// ==================== Notification Types ====================

export interface Notification {
    id: string;
    type: string;
    variant?: 'positive' | 'negative' | 'info' | 'warning';
    points?: number;
    message: string;
    timestamp: string;
    read: boolean;
}

// ==================== WebSocket Types ====================

export interface WebSocketMessage {
    type: string;
    timestamp?: string;
    vehicle_id?: string;
    status?: string;
    available_commands?: string[];
    data?: {
        telemetry: TelemetryData;
        anomaly: AnomalyData;
        scoring: ScoringData;
    };
    scenario?: ScenarioState;
    prediction?: Prediction;
    notifications?: Notification[];
    // Command responses
    routed_to?: string;
    advice?: SafetyAdvice;
    points_awarded?: number;
    diagnosis?: DiagnosisResult;
    rejection_count?: number;
    warning?: string;
}

// ==================== Chat Types ====================

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: string;
}

// ==================== API Response Types ====================

export interface ApiResponse<T> {
    data?: T;
    error?: string;
    status: number;
}

// ==================== Service Center Types ====================

export interface ServiceCenter {
    id: string;
    name: string;
    location: string;
    phone: string;
    specialties: string[];
    capacity_per_day: number;
    operating_hours: Record<string, string>;
    rating: number;
}

export interface TimeSlot {
    id: string;
    center_id: string;
    center_name?: string;
    location?: string;
    date: string;
    start_time: string;
    end_time: string;
    available: boolean;
    component_type: string;
    rating?: number;
}

// ==================== Appointment Types ====================

export interface Appointment {
    id: string;
    vehicle_id: string;
    center_id: string;
    center_name?: string;
    slot_id: string;
    component: string;
    diagnosis_summary: string;
    estimated_cost: string;
    urgency: 'low' | 'medium' | 'high' | 'critical';
    status: 'scheduled' | 'confirmed' | 'in_progress' | 'completed' | 'cancelled';
    created_at: string;
    scheduled_date: string;
    scheduled_time: string;
    completed_at?: string;
    notes: string;
}

export interface SlotProposal {
    success: boolean;
    message: string;
    slots: {
        option: number;
        slot_id: string;
        center_id: string;
        center_name: string;
        location: string;
        date: string;
        time: string;
        rating: number;
    }[];
    diagnosis_summary?: string;
    estimated_cost?: string;
    component?: string;
    urgency?: string;
}

// ==================== Feedback Types ====================

export interface ServiceFeedback {
    id: string;
    appointment_id: string;
    vehicle_id: string;
    rating: number;
    comments: string;
    submitted_at: string;
}

export interface FeedbackResponse {
    success: boolean;
    message: string;
    feedback_id?: string;
    rating?: number;
    next_service?: {
        recommendation: string;
        due_date: string;
        interval_days: number;
    };
    notification?: {
        type: string;
        title: string;
        body: string;
    };
}

// ==================== Service History Types ====================

export interface MaintenanceRecord {
    id: string;
    vehicle_id: string;
    appointment_id?: string;
    service_type: string;
    component: string;
    description: string;
    cost: string;
    performed_at: string;
    next_service_due?: string;
    mileage?: number;
}

export interface ServiceHistory {
    success: boolean;
    vehicle_id: string;
    total_services: number;
    average_rating: number;
    history: MaintenanceRecord[];
    feedback: ServiceFeedback[];
    upcoming_service?: {
        type: string;
        due_date: string;
    };
    message: string;
}

// ==================== Demand Forecasting Types ====================

export interface DemandForecast {
    center_id: string;
    center_name: string;
    capacity_per_day: number;
    predictions: {
        date: string;
        day: string;
        predicted_appointments: number;
        confidence: number;
    }[];
    average_daily_demand: number;
    utilization_forecast: number;
}

