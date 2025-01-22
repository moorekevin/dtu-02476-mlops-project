# MLOPS Group 46: Final project

Final project for DTU 02476 Machine Learning Operations course

By: Kevin Moore, Mario Medoni, Angeliki-Artemis Doumeni and Konstantina Freri

## Project description

### Overall Goal of the Project
Our project leverages natural language processing (NLP) to identify mental health disorders from Reddit posts across five major mental health subreddits. By building and fine-tuning a machine learning model, we aim to classify Reddit posts into categories that reflect what the author might suffer from or seek help with. This approach showcases the potential of NLP in analyzing user-generated content and offers a step toward developing tools to enhance mental health awareness and support systems.

### What Framework Are You Going to Use?
We will use the Hugging Face Transformers library, a powerful tool for NLP tasks. It offers pre-trained transformer models such as BERT and RoBERTa, known for their strong performance in understanding textual data. These models will be fine-tuned using both the title and selftext (body) of Reddit posts to classify them into five specific mental health disorder categories. This framework enables us to utilize state-of-the-art NLP techniques efficiently while focusing on our specific classification problem.

Additionally, we will incorporate PyTorch Lightning to streamline code and reduce boilerplate, enabling faster experimentation and improved project organization.

### What Data Are You Going to Run On?
The project will utilize the “Mental Disorders Identification (Reddit)” dataset from Kaggle. This dataset contains approximately 700k rows, including:
- **Post title** (string)
- **Post body (selftext)** (string)
- **Creation time** (timestamp)
- **Over 18 tag** (boolean)
- **Subreddit name** (mental disorder label, string)

**Note:** We will exclude the `mentalillness` subreddit, as it represents a generalized disorder, reducing the classification task to five subreddits.

**Link to the Kaggle Dataset:** [Mental Disorders Identification (Reddit)](https://www.kaggle.com/datasets/kamaruladha/mental-disorders-identification-reddit-nlp/data)

### What Models Do You Expect to Use?
We will fine-tune transformer models such as BERT and RoBERTa. As a baseline, we will use random guessing, assuming a balanced dataset where each category has a 20% probability of being predicted. This baseline will provide a clear comparison to demonstrate the performance of our models.

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps)
