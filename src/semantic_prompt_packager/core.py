from typing import Any, Dict, Optional
from pathlib import Path
import json
import sys

from .embedding import get_embeddings
from .manifest import PromptManifest

def init_prompt(name: str, output_dir: str = ".") -> Path:
    """Initialize a new prompt template."""
    template_content = f"""# {name}

Write your prompt template here. Use {{variables}} for placeholders.

Example 1:
Input: example input
Output: example output

Example 2:
Input: another input
Output: another output
"""
    
    output_path = Path(output_dir) / f"{name}.prompt"
    output_path.write_text(template_content)
    
    # Create a default manifest without embeddings initially
    # User can run `prompt pack` to add embeddings
    manifest = PromptManifest(
        content=template_content,
        embeddings=[],  # Empty embeddings initially
        version="0.1.0",
        description=f"Initial version of {name} prompt"
    )
    
    manifest_path = output_path.with_suffix(".pp.json")
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    
    return output_path

def pack_prompt(
    prompt_file: Path,
    output_file: Optional[Path] = None,
    provider: str = "sentence-transformers",
) -> Dict[str, Any]:
    """Package a prompt file into a versioned manifest with embeddings."""
    # Read prompt content
    content = prompt_file.read_text()
    
    # Generate embeddings
    embeddings = get_embeddings(content, provider=provider)
    
    # Create manifest
    manifest = PromptManifest(
        content=content,
        embeddings=embeddings,
        version="0.1.0",  # TODO: Implement version bumping
    )
    
    # Save manifest
    output_path = output_file or prompt_file.with_suffix(".pp.json")
    output_path.write_text(manifest.model_dump_json(indent=2))
    
    return manifest.model_dump()

def diff_prompts(
    version_a: Path,
    version_b: Path,
    threshold: float = 0.8,
    json_output: bool = False,
) -> Dict[str, Any]:
    """Compare two versions of a prompt using semantic diffing."""
    # Load manifests
    manifest_a = PromptManifest.model_validate_json(version_a.read_text())
    manifest_b = PromptManifest.model_validate_json(version_b.read_text())
    
    # Calculate similarity
    similarity = calculate_similarity(manifest_a.embeddings, manifest_b.embeddings)
    similarity_percent = float(similarity * 100)  # Ensure it's a Python float
    
    result = {
        "similarity": float(similarity),  # Ensure it's a Python float
        "similarity_percent": similarity_percent,
        "above_threshold": bool(similarity >= threshold),  # Ensure it's a Python bool
        "threshold": float(threshold),
        "versions": {
            "a": manifest_a.version,
            "b": manifest_b.version,
        },
        "content_diff": {
            "a": manifest_a.content,
            "b": manifest_b.content,
        }
    }
    
    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"Semantic similarity: {similarity_percent:.1f}%")
        print(f"Threshold: {threshold * 100:.1f}%")
        print(f"Above threshold: {'Yes' if result['above_threshold'] else 'No'}")
        print(f"Version A: {result['versions']['a']}")
        print(f"Version B: {result['versions']['b']}")
        
        # Show content diff
        print("\nContent comparison:")
        print("=" * 50)
        print("Version A:")
        print(manifest_a.content[:200] + ("..." if len(manifest_a.content) > 200 else ""))
        print("\nVersion B:")
        print(manifest_b.content[:200] + ("..." if len(manifest_b.content) > 200 else ""))
    
    # Exit with code 1 if below threshold
    if similarity < threshold:
        sys.exit(1)
    
    return result

def validate_manifest(manifest_path: Path) -> bool:
    """Validate a manifest against the JSON schema."""
    import json
    from pathlib import Path
    
    try:
        import jsonschema
    except ImportError as e:
        print(f"✗ jsonschema is required for validation. Install with: pip install jsonschema")
        return False
    
    # Load schema
    schema_path = Path(__file__).parent / "schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
    
    # Load and validate manifest
    try:
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        
        jsonschema.validate(manifest_data, schema)
        print(f"✓ Manifest {manifest_path} is valid")
        return True
    
    except jsonschema.ValidationError as e:
        print(f"✗ Validation error in {manifest_path}: {e.message}")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in {manifest_path}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error validating {manifest_path}: {e}")
        return False

def calculate_similarity(embeddings_a: list[float], embeddings_b: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    from scipy.spatial.distance import cosine
    
    return 1 - cosine(embeddings_a, embeddings_b) 