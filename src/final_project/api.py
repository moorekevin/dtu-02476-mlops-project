from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from src.final_project.model import AwesomeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextItem(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load and clean up model on startup and shutdown.
    """
    global model, tokenizer

    print("Loading model")

    # ---------------------------------------------------------
    # (1) Create a minimal config dict or adapt from your Hydra cfg
    #     with the necessary parameters (num_labels, model name, etc.)
    # ---------------------------------------------------------
    config_path = ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    # ---------------------------------------------------------
    # (2) Instantiate your AwesomeModel and load its weights
    # ---------------------------------------------------------
    model = AwesomeModel(cfg)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # (3) Load the tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])

    yield

    # Cleanup if needed
    print("Cleaning up")
    del model
    del tokenizer


app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict_text(data: TextItem):
    """
    Accept a JSON body containing text, runs inference and return predicted class.
    """
    text = data.text

    # ---------------------------------------
    # Tokenize
    # ---------------------------------------
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    # ---------------------------------------
    # Forward pass
    # ---------------------------------------
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # ---------------------------------------
    # Convert to probabilities and get the predicted class
    # ---------------------------------------
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return {
        "input_text": text,
        "predicted_class": predicted_class,
        "probabilities": probabilities.tolist()
    }
