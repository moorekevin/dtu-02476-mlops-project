from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from src.final_project.model import AwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu")
LABELS = ["BPD", "Anxiety", "depression", "bipolar", "schizophrenia"]


class TextItem(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load and clean up model on startup and shutdown.
    """
    global model, tokenizer

    print("Loading model")

    config_path = ".hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    # ---------------------------------------------------------
    # Instantiate AwesomeModel and load its weights
    # ---------------------------------------------------------
    model = AwesomeModel(cfg)
    model.load_state_dict(torch.load("models/model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # ---------------------------------------------------------
    # Load the tokenizer
    # ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])

    yield

    print("Cleaning up")
    del model
    del tokenizer


app = FastAPI(lifespan=lifespan)


def run_inference(text: str):
    """Local helper function that does the tokenization and forward pass."""
    tokenized = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokenized["input_ids"].to(DEVICE)
    attention_mask = tokenized["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = LABELS[torch.argmax(probabilities, dim=1).item()]

    probabilities_dict = {}
    for idx, label in enumerate(LABELS):
        # Multiply by 100 for a percentage, round to 2 decimals
        percent_value = probabilities[0][idx].item() * 100
        probabilities_dict[label] = f"{percent_value:.2f}%"

    return predicted_class, probabilities_dict


@app.post("/predict/")
async def predict_text(data: TextItem):
    """JSON endpoint for programmatic access."""
    predicted_class, probabilities = run_inference(data.text)
    return {
        "input_text": data.text,
        "predicted_class": predicted_class,
        "probabilities": probabilities
    }


@app.get("/", response_class=HTMLResponse)
async def show_form():
    """Simple form to input title and selftext from reddit post."""
    html_content = """
    <html>
    <head>
        <title>Predict mental health subreddit based on reddit post title and selftext</title>
    </head>
    <body>
        <h1>Enter Title and Selftext from reddit post</h1>
        <form action="/ui_predict" method="post">
            <label for="title">Title:</label><br>
            <input type="text" id="title" name="title" size="60"><br><br>

            <label for="selftext">Selftext:</label><br>
            <textarea id="selftext" name="selftext" rows="5" cols="60"></textarea><br><br>

            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/ui_predict", response_class=HTMLResponse)
async def ui_predict(title: str = Form(...), selftext: str = Form(...)):
    """
    Combine 'title' + 'selftext' into one text string,
    run prediction, display the result in HTML.
    """
    combined_text = f"{title}\n\n{selftext}"

    predicted_label, probabilities_dict = run_inference(combined_text)

    probs_html = "".join([
        f"<tr><td>{label}</td><td>{probabilities_dict[label]}</td></tr>"
        for label in LABELS
    ])

    result_html = f"""
    <html>
    <head>
        <title>Prediction Result</title>
    </head>
    <body>
        <h2>Your Input</h2>
        <p><strong>Title:</strong> {title}</p>
        <p><strong>Selftext:</strong> {selftext}</p>
        <hr/>
        <h2>Prediction</h2>
        <p>Predicted class: <strong>{predicted_label}</strong></p>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><th>Class</th><th>Probability</th></tr>
            {probs_html}
        </table>
        <br/>
        <a href="/">Go Back</a>
    </body>
    </html>
    """
    return HTMLResponse(content=result_html, status_code=200)
