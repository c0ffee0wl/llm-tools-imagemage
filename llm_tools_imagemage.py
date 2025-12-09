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

    MODES:
    - generate: Create image from text description
    - edit: Modify existing image(s) based on instruction

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

    6. EDITING INSTRUCTIONS - Be clear about changes:
       - "Add [element] to the [position] of the image"
       - "Change the [aspect] to [new value]"
       - "Remove [element] while preserving [surrounding context]"

    USAGE TIPS:
    - Style guidance belongs in the style parameter, not the prompt
    - Always use aspect_ratio parameter - model ignores dimensions in prompt text
    - Multi-image composition works best with 3 or fewer images (max 14 supported)
    - Specify output_path to control where files are saved
    - All generated images include invisible SynthID watermark

    Args:
        prompt: Descriptive text for generation, or editing instruction for edit mode
        mode: "generate" for text-to-image, "edit" for modifying existing images
        input_images: Comma-separated paths for edit mode (first is base image)
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

        # Edit an existing image
        generate_image(
            "Add dramatic storm clouds gathering in the sky",
            mode="edit",
            input_images="/path/to/landscape.png"
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

    # Build command based on mode
    if mode == "edit":
        if not input_images:
            return llm.ToolOutput(
                "Error: edit mode requires input_images parameter with path(s) to image(s)"
            )
        images = [p.strip() for p in input_images.split(",")]

        # Validate input images exist
        for img in images:
            if not os.path.isfile(img):
                return llm.ToolOutput(f"Error: input image not found: {img}")

        base_image = images[0]
        cmd = ["imagemage", "edit", base_image, prompt]

        # Add additional images for composition
        for img in images[1:]:
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
        return llm.ToolOutput(
            "Error: Image generation timed out after 5 minutes. "
            "Try a simpler prompt or use model='flash' for faster generation."
        )
    except Exception as e:
        return llm.ToolOutput(f"Error running imagemage: {e}")

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
