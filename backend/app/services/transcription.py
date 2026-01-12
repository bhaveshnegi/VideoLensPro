from pathlib import Path
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer
import torch

class TranscriptionService:
    def __init__(self, model_size="small"):
        # Initialize Whisper
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        # Toxicity model
        model_name = "unitary/toxic-bert"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.toxicity_model = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def _chunk_text(self, text, max_tokens=500):
        """
        Split text into token-aware chunks for BERT.
        """
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False
        )["input_ids"]

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            yield self.tokenizer.decode(chunk_tokens)

    def transcribe(self, audio_path: Path):
        """
        Transcribe audio and detect abusive/toxic content.
        """
        segments, info = self.model.transcribe(str(audio_path))
        text = " ".join(seg.text for seg in segments).strip()

        if not text:
            return {
                "text": "",
                "toxicity": None,
                "language": info.language,
                "is_toxic": False
            }

        toxicity_results = []

        # ✅ Chunked inference
        for chunk in self._chunk_text(text):
            chunk_results = self.toxicity_model(
                chunk,
                truncation=True,
                max_length=512
            )
            toxicity_results.extend(chunk_results)

        # Filter toxic labels
        toxic_labels = [r for r in toxicity_results if r["score"] > 0.5]

        return {
            "text": text,
            "language": info.language,
            "toxicity": toxic_labels,
            "is_toxic": len(toxic_labels) > 0,
            "confidence": {
                r["label"]: round(r["score"], 3)
                for r in toxic_labels
            }
        }
