"""
LLM tool for Gemini image generation and editing via imagemage.

Wraps the imagemage Go binary to provide AI-powered image generation
and editing capabilities within the llm ecosystem.
"""

import llm
import subprocess
import shutil
import os
import re
import tempfile
import urllib.request


def _download_url_to_temp(url: str) -> str:
    """Download URL to a temporary file, return the path.

    Preserves the file extension from the URL for proper MIME type handling.
    Falls back to .png if no extension can be determined.
    """
    # Extract extension from URL (before query params and fragments)
    url_path = url.split('?')[0].split('#')[0]
    ext = os.path.splitext(url_path)[1].lower()

    # Validate extension is a known image type, otherwise default to .png
    if ext not in ('.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp'):
        ext = '.png'

    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    )
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        with urllib.request.urlopen(request, timeout=60) as response:
            # Check content-type header as fallback for extension
            content_type = response.headers.get('Content-Type', '')
            if ext == '.png' and content_type:
                # Map content-type to extension if we defaulted
                type_to_ext = {
                    'image/jpeg': '.jpg',
                    'image/png': '.png',
                    'image/webp': '.webp',
                    'image/gif': '.gif',
                }
                for mime, detected_ext in type_to_ext.items():
                    if mime in content_type:
                        # Need to create a new temp file with correct extension
                        tmp.close()
                        os.unlink(tmp.name)
                        with tempfile.NamedTemporaryFile(suffix=detected_ext, delete=False) as tmp2:
                            tmp2.write(response.read())
                            return tmp2.name
            tmp.write(response.read())
        return tmp.name


def _resolve_image_path(path_or_url: str) -> tuple[str, str | None]:
    """Resolve an image path or URL to a local file path.

    Supports:
    - Local paths: /path/to/image.png
    - HTTP/HTTPS URLs: https://example.com/image.png
    - File URLs: file:///path/to/image.png

    Returns:
        (local_path, temp_file_or_none) - temp_file is set if we downloaded
    """
    if path_or_url.startswith(('http://', 'https://')):
        temp_file = _download_url_to_temp(path_or_url)
        return temp_file, temp_file
    if path_or_url.startswith('file://'):
        # Convert file:// URL to local path
        local_path = path_or_url[7:]  # Remove 'file://'
        return local_path, None
    return path_or_url, None


def _cleanup_temp_files(temp_files: list[str]) -> None:
    """Clean up temporary files, ignoring errors."""
    for tf in temp_files:
        try:
            os.unlink(tf)
        except Exception:
            pass


def _open_image_viewer(image_path: str) -> None:
    """Open image in default viewer, detached from terminal process.

    Uses start_new_session=True to create a new process group, which:
    - Prevents SIGHUP when terminal closes
    - Keeps viewer open after sidechat exits
    """
    try:
        subprocess.Popen(
            ["xdg-open", image_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True
        )
    except Exception:
        # Silently ignore viewer errors - image is still saved
        pass


def generate_image(
    prompt: str,
    mode: str = "generate",
    input_images: str = "",
    output_path: str = "",
    aspect_ratio: str = "",
    resolution: str = "",
    model: str = "pro",
    style: str = "",
    auto_open: bool = True,
) -> llm.ToolOutput:
    """
    Generate or edit images using Google Gemini via imagemage.

    IMPORTANT: When the user asks to modify, edit, or change an existing image,
    ALWAYS use this tool with mode="edit". Gemini is capable of precise edits
    including removing elements, adding elements, changing colors, modifying text,
    and redrawing portions of images. Do not refuse edit requests - try them.

    MODES:
    - generate: Create image from text description
    - edit: Modify existing image(s) - can add, remove, change, or redraw elements

    MODELS:
    - pro (default): gemini-3-pro-image-preview - High quality, up to 4K, complex reasoning
    - flash: gemini-2.5-flash-image - Faster and cheaper, good for drafts/iterations

    ASPECT RATIOS: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9

    RESOLUTIONS (pro only): 1K, 2K (default), 4K

    PROMPTING GUIDE:

    1. DESCRIBE SCENES NARRATIVELY (don't list keywords)
       Bad: "sunset, mountain, lake, reflection"
       Good: "A serene mountain lake at sunset with snow-capped peaks reflected in still water"

    2. PHOTOREALISTIC - Include photography terminology:
       - Shots: wide shot, close-up, aerial view, macro, over-the-shoulder
       - Lighting: golden hour, studio lighting, backlit, soft diffused light
       - Lens effects: 35mm, telephoto, shallow depth of field, bokeh

    3. ILLUSTRATIONS - Specify art style explicitly:
       - Styles: watercolor, pixel art, vector, oil painting, anime, pencil sketch
       - Details: line weight, shading approach, color palette
       - For icons/stickers: include "transparent background"

    4. TEXT IN IMAGES - Be very explicit:
       - Specify exact text content, font style description, placement
       - Pro model excels at rendering legible, styled text

    5. PRODUCT/MOCKUP SHOTS - Use professional terms:
       - Background: white seamless, gradient, contextual setting
       - Lighting: three-point lighting, rim light, soft box
       - Angles: hero shot, flat lay, 45-degree, eye-level

    6. EDITING INSTRUCTIONS - Be specific about what to change:
       - REMOVE: "Remove the essos.local section from this diagram"
       - ADD: "Add a cloud icon in the top right corner"
       - CHANGE: "Change all blue elements to green"
       - REDRAW: "Redraw this diagram without the bottom section"
       - TEXT: "Change the title text to say 'New Title'"

    USAGE TIPS:
    - Style guidance belongs in the style parameter, not the prompt
    - Always use aspect_ratio parameter - model ignores dimensions in prompt text
    - Multi-image composition works best with 3 or fewer images (max 14 supported)
    - Specify output_path to control where files are saved
    - All generated images include invisible SynthID watermark

    Args:
        prompt: Descriptive text for generation, or editing instruction for edit mode
        mode: "generate" for text-to-image, "edit" for modifying existing images
        input_images: Comma-separated paths or URLs for edit mode (first is base image)
        output_path: Output directory for generate, or file path for edit mode
        aspect_ratio: Output aspect ratio (use parameter, not prompt text)
        resolution: 1K, 2K (default), or 4K (pro model only)
        model: "pro" for quality or "flash" for speed/cost
        style: Style guidance separate from main prompt (e.g., "watercolor, muted palette")
        auto_open: Open generated image in viewer (default: True)

    Returns:
        ToolOutput with image path and attachment for inline display

    Examples:
        # Generate - describe the scene narratively
        generate_image(
            "A cozy coffee shop interior with morning light streaming through large windows",
            aspect_ratio="16:9"
        )

        # Edit - add elements
        generate_image(
            "Add dramatic storm clouds gathering in the sky",
            mode="edit",
            input_images="https://example.com/landscape.png"
        )

        # Edit - remove elements from diagram
        generate_image(
            "Remove the essos.local section from this network diagram",
            mode="edit",
            input_images="https://example.com/diagram.png"
        )

        # Multi-image composition (best with 2-3 images)
        generate_image(
            "Place the person naturally on the left side of the outdoor scene",
            mode="edit",
            input_images="/path/to/background.png,/path/to/person.png"
        )

        # Styled generation - style in parameter, scene in prompt
        generate_image(
            "A mountain vista with a winding river through a valley",
            style="watercolor painting, soft edges, muted earth tones"
        )
    """
    # Check if imagemage is available in PATH
    if not shutil.which("imagemage"):
        return llm.ToolOutput(
            "Error: imagemage not found in PATH. Install with:\n"
            "  git clone https://github.com/quinnypig/imagemage.git\n"
            "  cd imagemage && go build -o imagemage\n"
            "  # Then add to PATH or: go install"
        )

    # Track temp files for cleanup
    temp_files = []

    # Build command based on mode
    if mode == "edit":
        if not input_images:
            return llm.ToolOutput(
                "Error: edit mode requires input_images parameter with path(s) to image(s)"
            )
        image_sources = [p.strip() for p in input_images.split(",") if p.strip()]

        if not image_sources:
            return llm.ToolOutput(
                "Error: edit mode requires at least one image path or URL"
            )

        # Resolve paths/URLs to local files
        resolved_images = []
        for img_source in image_sources:
            try:
                local_path, temp_file = _resolve_image_path(img_source)
                if temp_file:
                    temp_files.append(temp_file)
                resolved_images.append(local_path)
            except Exception as e:
                _cleanup_temp_files(temp_files)
                return llm.ToolOutput(f"Error downloading image from {img_source}: {e}")

        # Validate resolved images exist
        for img in resolved_images:
            if not os.path.isfile(img):
                _cleanup_temp_files(temp_files)
                return llm.ToolOutput(f"Error: input image not found: {img}")

        base_image = resolved_images[0]
        cmd = ["imagemage", "edit", base_image, prompt]

        # Add additional images for composition
        for img in resolved_images[1:]:
            cmd.extend(["-i", img])
    else:
        # generate mode
        cmd = ["imagemage", "generate", prompt]

    # Add optional parameters
    if output_path:
        cmd.extend(["-o", output_path])

    if aspect_ratio:
        cmd.extend(["-a", aspect_ratio])

    # Resolution: default to 2K for pro model, flash has fixed 1024px
    if model != "flash":
        res = resolution if resolution else "2K"
        cmd.extend(["-r", res])
    else:
        cmd.append("--frugal")

    # Style flag only available for generate mode
    if style and mode != "edit":
        cmd.extend(["-s", style])

    # Run imagemage
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for image generation
        )
    except subprocess.TimeoutExpired:
        _cleanup_temp_files(temp_files)
        return llm.ToolOutput(
            "Error: Image generation timed out after 5 minutes. "
            "Try a simpler prompt or use model='flash' for faster generation."
        )
    except Exception as e:
        _cleanup_temp_files(temp_files)
        return llm.ToolOutput(f"Error running imagemage: {e}")

    # Cleanup temp files after imagemage has processed them
    _cleanup_temp_files(temp_files)

    # Check for errors
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout or "Unknown error"
        return llm.ToolOutput(f"Error: {error_msg}")

    # Extract output path from stdout
    # imagemage prints "✓ Saved to: /path/to/file.png" (or .jpg, etc. for edit mode)
    output = result.stdout
    match = re.search(r"Saved to: (.+\.(png|jpg|jpeg|webp))", output, re.IGNORECASE)

    if not match:
        return llm.ToolOutput(
            f"Image may have been generated but could not find output path:\n{output}"
        )

    image_path = match.group(1).strip()
    extension = match.group(2).lower()

    # Determine MIME type from extension
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }
    mime_type = mime_types.get(extension, "image/png")

    # Auto-open in viewer if requested
    if auto_open:
        _open_image_viewer(image_path)

    # Load image and return as attachment
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()

        return llm.ToolOutput(
            f"Generated image saved to: {image_path}",
            attachments=[
                llm.Attachment(
                    content=image_data,
                    type=mime_type
                )
            ]
        )
    except Exception as e:
        return llm.ToolOutput(
            f"Image saved to {image_path} but failed to load for preview: {e}"
        )


@llm.hookimpl
def register_tools(register):
    """Register the image generation tool."""
    register(generate_image)
