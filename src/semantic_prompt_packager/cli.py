"""Command-line interface for LLM Prompt Semantic Diff."""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table

from . import __version__
from .core import init_prompt, pack_prompt, diff_prompts, validate_manifest

app = typer.Typer(help="A CLI tool for managing and comparing LLM prompts using semantic diffing")
console = Console()

@app.command()
def version():
    """Show the version of LLM Prompt Semantic Diff."""
    console.print(f"LLM Prompt Semantic Diff version: {__version__}")

@app.command()
def init(
    name: str = typer.Argument(..., help="Name of the prompt template to create"),
    output_dir: str = typer.Option(".", help="Directory to create the prompt template in"),
) -> None:
    """Initialize a new prompt template."""
    try:
        prompt_path = init_prompt(name, output_dir)
        console.print(f"[green]✓ Created prompt template: {prompt_path}[/green]")
        console.print(f"[green]✓ Created manifest: {prompt_path.with_suffix('.pp.json')}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error creating prompt template: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def pack(
    file: str = typer.Argument(..., help="Path to the prompt file to package"),
    output: Optional[str] = typer.Option(None, help="Output path for the packaged prompt"),
    provider: str = typer.Option("sentence-transformers", help="Embedding provider (openai or sentence-transformers)"),
) -> None:
    """Package a prompt template into a versioned manifest."""
    try:
        prompt_file = Path(file)
        output_file = Path(output) if output else None
        
        if not prompt_file.exists():
            console.print(f"[red]✗ Prompt file not found: {file}[/red]")
            raise typer.Exit(1)
        
        result = pack_prompt(prompt_file, output_file, provider)
        output_path = output_file or prompt_file.with_suffix(".pp.json")
        
        console.print(f"[green]✓ Packaged prompt: {file}[/green]")
        console.print(f"[green]✓ Created manifest: {output_path}[/green]")
        console.print(f"[blue]Version: {result['version']}[/blue]")
        console.print(f"[blue]Embeddings: {len(result['embeddings'])} dimensions[/blue]")
        
    except Exception as e:
        console.print(f"[red]✗ Error packaging prompt: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def diff(
    version_a: str = typer.Argument(..., help="First version to compare"),
    version_b: str = typer.Argument(..., help="Second version to compare"),
    threshold: float = typer.Option(0.8, help="Similarity threshold for exit code"),
    json: bool = typer.Option(False, help="Output in JSON format"),
    provider: str = typer.Option("sentence-transformers", help="Embedding provider (openai or sentence-transformers)"),
) -> None:
    """Compare two versions of a prompt template."""
    try:
        path_a = Path(version_a)
        path_b = Path(version_b)
        
        if not path_a.exists():
            console.print(f"[red]✗ Version A not found: {version_a}[/red]")
            raise typer.Exit(1)
        
        if not path_b.exists():
            console.print(f"[red]✗ Version B not found: {version_b}[/red]")
            raise typer.Exit(1)
        
        # This will handle output and exit codes internally
        diff_prompts(path_a, path_b, threshold, json)
        
    except typer.Exit:
        # Re-raise typer exits (from threshold checks)
        raise
    except Exception as e:
        console.print(f"[red]✗ Error comparing prompts: {e}[/red]")
        raise typer.Exit(1)

@app.command()
def validate(
    file: str = typer.Argument(..., help="Path to the manifest file to validate"),
) -> None:
    """Validate a prompt manifest against the schema."""
    try:
        manifest_path = Path(file)
        
        if not manifest_path.exists():
            console.print(f"[red]✗ Manifest file not found: {file}[/red]")
            raise typer.Exit(1)
        
        is_valid = validate_manifest(manifest_path)
        
        if not is_valid:
            raise typer.Exit(1)
            
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]✗ Error validating manifest: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app() 