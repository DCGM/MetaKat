# filepath: /home/ikohut/Projects/MetaKat/metakat/worker/gpt_worker.py

import base64
import argparse
import logging
from pathlib import Path
from typing import Optional

from openai import OpenAI


logger = logging.getLogger(__name__)


class GPTWorker:
    """Worker that communicates with the OpenAI ChatGPT API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        """Initialize GPTWorker with OpenAI API credentials.

        Args:
            api_key: OpenAI API key.
            model: Model name to use for completions.
            max_tokens: Maximum number of tokens in the response.
            temperature: Sampling temperature (0-2). Lower values are more
                deterministic.
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_completion_tokens = max_tokens
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
        """Send a prompt (with optional images) to ChatGPT and return the
        response text.

        Args:
            text: The text prompt to send.
            images: Optional list of image file paths to include in the
                prompt.  Each entry should be a path to a local image file
                (JPEG, PNG, GIF, or WEBP).
            system_message: Optional system-level instruction for the model.
            json_mode: If True, force the model to output valid JSON by
                setting ``response_format`` to ``{"type": "json_object"}``.

        Returns:
            The assistant's reply as a string.
        """
        user_content: list[dict] = [{"type": "text", "text": text}]

        for image_url in self._prepare_images(images or []):
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]

        logger.debug(
            "Sending prompt to %s (images: %d)", self.model, len(images or [])
        )

        kwargs = dict(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        reply = response.choices[0].message.content
        logger.debug("Received response (%d chars)", len(reply))
        return reply

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_images(images: list[str | Path]) -> list[str]:
        """Convert a list of image paths to base64-encoded data URLs.

        Args:
            images: List of file paths to local images.

        Returns:
            List of ``data:image/...;base64,...`` URLs.
        """
        MIME_TYPES = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        data_urls: list[str] = []
        for img_path in images:
            path = Path(img_path)
            suffix = path.suffix.lower()
            mime = MIME_TYPES.get(suffix)
            if mime is None:
                raise ValueError(
                    f"Unsupported image format '{suffix}' for file {path}. "
                    f"Supported formats: {', '.join(MIME_TYPES)}"
                )
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            data_urls.append(f"data:{mime};base64,{encoded}")

        return data_urls


def main():
    parser = argparse.ArgumentParser(
        description="Send a prompt (with optional images) to the OpenAI ChatGPT API.",
    )

    # Worker setup
    api_key_group = parser.add_mutually_exclusive_group(required=True)
    api_key_group.add_argument(
        "--api-key",
        help="OpenAI API key.",
    )
    api_key_group.add_argument(
        "--api-key-file",
        type=Path,
        help="Path to a file containing the OpenAI API key.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Model name to use for completions (default: gpt-5.4).",
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
        help="Force the model to output valid JSON (sets response_format to json_object).",
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

    # Resolve API key
    api_key = args.api_key if args.api_key else args.api_key_file.read_text().strip()

    # Resolve prompt
    prompt_text = args.prompt if args.prompt else args.prompt_file.read_text()

    worker = GPTWorker(
        api_key=api_key,
        model=args.model,
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
