# VMT - Discord Voice Message Transcriber

[![License: MIT](https://img.shields.io/badge/license-MIT-blueviolet.svg)](https://github.com/dromzeh/vmt)
[![Formatter: Black](https://img.shields.io/badge/formatter-Black-lightgrey.svg)](https://black.readthedocs.io/en/stable/)

A Discord bot that locally transcribes Discord Voice Messages (voice notes) to text using the tiny `faster-whisper` model (runs on CPU, auto‑detects better hardware if available).

## Requirements

- Python 3.8+
- discord.py
- pydub + ffmpeg (for audio format conversion)
- faster-whisper

Run the following:

```bash
pip install -r requirements.txt
```

## Installation

> **Note**
> Bot requires the proper Gateway Intents to be enabled in the Discord developer portal (Message Content + Guild Messages for most setups) and `ffmpeg` available (installed automatically in the provided Docker image).

- Clone the repository: `git clone https://github.com/dromzeh/vmt.git`
- Rename the `config.example.json` file to `config.json` inside `src/config` and fill in the necessary values (See [Configuration](#configuration)) 

Once the configuration file has been filled, you may run the bot using one of the following methods:

### Native Installation

- Install the requirements `pip install -r requirements.txt`
- Run the bot inside the `/src` folder: `python main.py`

### Docker Installation (Easier)

Assuming `docker` and `docker-compose` are installed

```bash
docker-compose -f compose.yaml up
```

## Usage

- Reply to a voice message with the `transcribe` command (e.g. `vmt transcribe`).
- If you don't reply to a specific message, the bot searches recent channel history for the most recent voice note and transcribes that instead.
- The bot also auto-transcribes any new voice message (adds an ⌛ reaction while processing, then posts an embed with the text).

### Configuration

The `config.json` file contains the following fields:

- `token`: Your Discord bot token.
- `prefix`: Command prefix (default example uses `vmt ` including a space).

Example:

```json
{
    "token": "YOUR_DISCORD_BOT_TOKEN",
    "prefix": "vmt "
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
