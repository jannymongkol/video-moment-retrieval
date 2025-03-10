# video-moment-retrieval

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jannymongkol/video-moment-retrieval.git
   cd video-moment-retrieval
   ```

2. **Create a virtual environment and activate it:**
   
    - With venv:

        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    
    - With conda:
    
        ```bash
        conda create -n env_name
        conda activate env_name
        conda install pip
        ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the necessary data and place it in the appropriate directories:**
   - Ensure the data is structured as described below.

## Expected Data Folder Structure

```
video-moment-retrieval/
└── data/
    ├── vu17_charades/
    ├── Charades-CD/
    ├── Charades_v1_480/
    └── Charades_v1_480_16/
```

- **data/vu17_charades/**: 
    - Download from https://prior.allenai.org/projects/charades (Annotations and Evaluation Code)
- **data/Charades-CD/**: 
    - Contains the Charades-CD dataset, for mapping natural langauge queries to time-stamps within Charades Videos.
    - Download from https://github.com/yytzsy/grounding_changing_distribution/tree/main 
- **data/Charades_v1_480/**: 
    - Contains the videos in MP4 format
    - Download from https://prior.allenai.org/projects/charades (Data scaled to 480p)
- **data/Charades_v1_480_16/**: 
    - Contains the videos in Charades dataset, standardized to 16 fps
    - Run ./generate_16fps.sh, to generate this folder

Make sure to follow this structure to ensure the code runs correctly.