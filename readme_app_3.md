berlin-environment-app/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
│
├── data/
│   ├── boundaries/
│   │   ├── berlin_bezirke.geojson
│   │   └── berlin_boundary.geojson
│   └── ndvi/
│       ├── nvdi_001.png
│       └── nvdi_009.png
│
├── utils/
│   ├── h3_grid_generator.py       # H3 grid creation utilities
│   ├── change_detector.py         # AI-powered change detection
│   └── visualization.py           # Map visualization helpers
│
└── reports/
    └── templates/
        └── report_template.md