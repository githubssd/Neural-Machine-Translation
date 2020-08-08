## Readme

``` Every folder is for different NMT model, and they contain their own README.md files for reference and setup.```
```bash
Project Structure for Every Model.

│   DockerBuild.cmd
│   Dockerfile
│   manage.py
│   README.md
│   requirements.txt
│
├───app
│   │   config.py
│   │   __init__.py
│   │
│   ├───controllers
│   │       __init__.py
│   │
│   ├───helpers
│   │       __init__.py
│   │
│   ├───main
│   │       __init__.py
│   │
│   ├───models
│   │       __init__.py
│   │
│   ├───resources
│   │       __init__.py
│   │
│   ├───services
│   │       __init__.py
│   │
│   └───utils
│           __init__.py
│
└───Training
    │   Playground.ipynb
    │
    ├───data_source
    ├───helpers
    │       __init__.py
    │
    ├───models
    │       model.py
    │       __init__.py
    │
    ├───model_weights
    ├───training_summary
    └───utils
            utils.py
            __init__.py
```