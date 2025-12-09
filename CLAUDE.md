# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an LLM tool plugin that wraps the `imagemage` Go binary to provide Google Gemini image generation and editing capabilities within the llm ecosystem. It exposes a single tool `generate_image` that can create images from text prompts or edit existing images.

## Dependencies

- **Runtime**: `imagemage` binary must be in PATH (https://github.com/quinnypig/imagemage)
- **Python**: >= 3.10
- **Package**: `llm` library

## Installation

```bash
# Install as llm plugin
llm install git+https://github.com/c0ffee0wl/llm-tools-imagemage

# Or for development
llm install -e /path/to/llm-tools-imagemage
```

## Architecture

Single-file plugin (`llm_tools_imagemage.py`) with:

- **URL handling**: Downloads HTTP/HTTPS URLs to temp files, supports `file://` URLs
- **Subprocess wrapper**: Calls `imagemage generate` or `imagemage edit` with appropriate flags
- **Dual output**: Returns both file path and `llm.ToolOutput` with `Attachment` for inline display
- **Cleanup**: Temp files cleaned up after imagemage processes them

## Key Functions

| Function | Purpose |
|----------|---------|
| `_download_url_to_temp()` | Downloads image URL to temp file with MIME type detection |
| `_resolve_image_path()` | Resolves path/URL to local file, returns (path, temp_file_or_none) |
| `_cleanup_temp_files()` | Cleans up temp files on success or error |
| `generate_image()` | Main tool function - generate or edit images |

## Timeouts

- URL download: 30 seconds
- imagemage subprocess: 60 seconds

## Validation

```bash
# Check Python syntax
python3 -m py_compile llm_tools_imagemage.py

# Verify tool registration
llm tools | grep generate_image
```

## imagemage CLI Mapping

| Tool Parameter | imagemage Flag |
|----------------|----------------|
| `output_path` | `-o` |
| `aspect_ratio` | `-a` |
| `resolution` | `-r` |
| `model="flash"` | `--frugal` |
| `style` | `-s` (generate only) |
| Additional images | `-i` (edit only) |
