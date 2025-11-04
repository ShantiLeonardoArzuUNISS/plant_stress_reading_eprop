# Plant Stress Reading - Spiking Neural Networks

Detection of water stress and iron deficiency in tomato plants using spiking neural networks (SRNN)

================================================================================
PROJECT DESCRIPTION
================================================================================

This project adapts a recurrent spiking neural network (SRNN) with CUBA LIF neurons for the analysis of bioimpedance data from tomato plants under stress conditions.

================================================================================
SYSTEM REQUIREMENTS
================================================================================

- Python 3.8+
- pip

================================================================================
QUICK INSTALLATION
================================================================================

1. CLONE THE REPOSITORY
----------------------
git clone https://github.com/[your-username]/plant_stress_reading.git
cd plant_stress_reading


2. CREATE THE VIRTUAL ENVIRONMENT
---------------------------------

Windows:
    python -m venv venv
    venv\Scriptsctivate

macOS / Linux:
    python3 -m venv venv
    source venv/bin/activate


3. INSTALL DEPENDENCIES
----------------------
pip install --upgrade pip
pip install -r requirements.txt


================================================================================
PROJECT STRUCTURE
================================================================================

plant_stress_reading/
├── scripts/              # Training and testing scripts
├── utils/               # Utility functions
├── data/                # Dataset folder (empty)
├── figures_plants/      # Generated plots
├── plots/               # Visual results
├── .gitignore           # Files to exclude from git
├── requirements.txt     # Python dependencies
└── README.md            # This file
