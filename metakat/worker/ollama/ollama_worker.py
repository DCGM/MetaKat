import base64
import argparse
import logging
from pathlib import Path
from typing import Optional

from ollama import Client


logger = logging.getLogger(__name__)


class OllamaWorker:
    """Worker that communicates with a locally hosted Ollama instance."""

    def __init__(
        self,
        model: str,
        host: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        """Initialize OllamaWorker.

        Args:
            model: Model name to use for completions.
            host: Ollama server URL.
            max_tokens: Maximum number of tokens in the response.
            temperature: Sampling temperature (0-2). Lower values are more
                deterministic.
        """
        self.client = Client(host=host)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prompt(
        self,
        text: str,
        images: Optional[list[str | Path]] = None,
        system_message: str = "You are an expert document analyst. Respond only in JSON.",
        json_mode: bool = False,
    ) -> str:
        """Send a prompt (with optional images) to Ollama and return the
        response text.

        Args:
            text: The text prompt to send.
            images: Optional list of image file paths to include in the
                prompt.  Each entry should be a path to a local image file
                (JPEG, PNG, GIF, or WEBP).
            system_message: Optional system-level instruction for the model.
            json_mode: If True, force the model to output valid JSON by
                setting ``format`` to ``"json"``.

        Returns:
            The assistant's reply as a string.
        """
        image_data = self._load_images(images or [])

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": text, "images": image_data} if image_data
            else {"role": "user", "content": text},
        ]

        logger.debug(
            "Sending prompt to %s (images: %d)", self.model, len(image_data)
        )

        kwargs = dict(
            model=self.model,
            messages=messages,
            options={
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        if json_mode:
            kwargs["format"] = "json"

        response = self.client.chat(**kwargs)

        reply = response.message.content
        logger.debug("Received response (%d chars)", len(reply))
        return reply

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_images(images: list[str | Path]) -> list[bytes]:
        """Read image files and return their raw bytes.

        Ollama accepts images as raw bytes or base64 strings in the
        ``images`` field of a message.

        Args:
            images: List of file paths to local images.

        Returns:
            List of raw image bytes.
        """
        SUPPORTED = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

        image_data: list[bytes] = []
        for img_path in images:
            path = Path(img_path)
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED:
                raise ValueError(
                    f"Unsupported image format '{suffix}' for file {path}. "
                    f"Supported formats: {', '.join(SUPPORTED)}"
                )
            image_data.append(path.read_bytes())

        return image_data


def main():
    parser = argparse.ArgumentParser(
        description="Send a prompt (with optional images) to a locally hosted Ollama model.",
    )

    # Worker setup
    parser.add_argument(
        "--host",
        required=True,
        help="Ollama server URL (e.g. http://localhost:11434).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for completions (e.g. llama3.2-vision).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens in the response (default: 4096).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature 0-2 (default: 0.0).",
    )

    # Prompt
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        help="Text prompt to send to the model.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Path to a file containing the text prompt.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Optional list of image file paths to include in the prompt.",
    )
    parser.add_argument(
        "--system-message",
        default="You are an expert document analyst. Respond only in JSON.",
        help="System-level instruction for the model.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help="Force the model to output valid JSON.",
    )

    # Output
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to a file where the response will be saved. If not provided, prints to stdout.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve prompt
    prompt_text = args.prompt if args.prompt else args.prompt_file.read_text()

    worker = OllamaWorker(
        model=args.model,
        host=args.host,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    reply = worker.prompt(
        text=prompt_text,
        images=args.images,
        system_message=args.system_message,
        json_mode=args.json_mode,
    )

    if args.output_file:
        args.output_file.write_text(reply)
        logger.info("Response saved to %s", args.output_file)
    else:
        print(reply)


if __name__ == "__main__":
    main()
