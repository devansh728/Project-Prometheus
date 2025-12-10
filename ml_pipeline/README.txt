# ML Pipeline - EV Telematics Anomaly & Failure Prediction

This standalone module provides end-to-end ML capabilities for:
1. Real-time anomaly detection from EV telemetry
2. Failure prediction with configurable horizons
3. Severity classification
4. Digital twin personalization

## Project Structure

```
ml_pipeline/
├── telemetry_schema.json    # EV telemetry field definitions
├── labeling_spec.txt        # Labeling rules and failure modes
├── evaluation_metrics.txt   # Success criteria and targets
├── config.json              # All tunable parameters
├── synthetic_data/          # Generated data (Phase 2)
│   ├── raw_frames/
│   └── aggregates/
├── datasets/                # Processed train/val/test (Phase 3)
├── models/                  # Trained models (Phase 4)
├── scripts/                 # Training and inference scripts
├── services/                # FastAPI orchestrator (Phase 6)
└── rag/                     # RAG vector index (Phase 7)
```

## Quick Start

```bash
# Phase 2: Generate synthetic data
python scripts/generate_synthetic.py --config config.json

# Phase 3: Preprocess and create datasets
python scripts/preprocessing.py --config config.json

# Phase 4: Train models
python scripts/train_lgbm.py --config config.json
python scripts/train_lstm_ae.py --config config.json

# Phase 6: Start real-time service
python services/fastapi_orchestrator.py
```

## Configuration

All parameters are in `config.json`. Key settings:
- `num_vehicles`: 10 vehicles for synthetic data
- `duration_days_per_vehicle`: 7 days per vehicle
- `failure_horizons_days`: [7, 14, 30] day prediction windows
- `anomaly_threshold_percentile`: 99th percentile for anomaly detection

## Phase Deliverables

- Phase 1: Schema, labeling spec, metrics (COMPLETE)
- Phase 2: Synthetic data generator
- Phase 3: Preprocessing pipeline
- Phase 4: LightGBM + LSTM-AE models
- Phase 5: Evaluation and thresholds
- Phase 6: Real-time FastAPI service
- Phase 7: RAG explanations
- Phase 8: Monitoring and final report
