import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from semantic_prompt_packager.core import (
    init_prompt,
    pack_prompt,
    diff_prompts,
    validate_manifest,
    calculate_similarity,
)
from semantic_prompt_packager.manifest import PromptManifest


def test_init_prompt():
    """Test prompt initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('semantic_prompt_packager.core.get_embeddings') as mock_embeddings:
            mock_embeddings.return_value = [0.1, 0.2, 0.3]
            
            result_path = init_prompt("test-prompt", temp_dir)
            
            # Check files were created
            assert result_path.exists()
            assert result_path.with_suffix(".pp.json").exists()
            
            # Check content
            content = result_path.read_text()
            assert "test-prompt" in content
            assert "Write your prompt template here" in content


def test_pack_prompt():
    """Test prompt packing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('semantic_prompt_packager.core.get_embeddings') as mock_embeddings:
            mock_embeddings.return_value = [0.1, 0.2, 0.3]
            
            # Create test prompt file
            prompt_file = Path(temp_dir) / "test.prompt"
            prompt_file.write_text("Test prompt content")
            
            result = pack_prompt(prompt_file)
            
            # Check result structure
            assert "content" in result
            assert "version" in result
            assert "embeddings" in result
            assert result["content"] == "Test prompt content"
            assert result["version"] == "0.1.0"
            assert result["embeddings"] == [0.1, 0.2, 0.3]
            
            # Check manifest file was created
            manifest_file = prompt_file.with_suffix(".pp.json")
            assert manifest_file.exists()


def test_pack_prompt_with_custom_output():
    """Test prompt packing with custom output path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('semantic_prompt_packager.core.get_embeddings') as mock_embeddings:
            mock_embeddings.return_value = [0.1, 0.2, 0.3]
            
            prompt_file = Path(temp_dir) / "test.prompt"
            prompt_file.write_text("Test prompt content")
            
            output_file = Path(temp_dir) / "custom.pp.json"
            
            result = pack_prompt(prompt_file, output_file)
            
            assert output_file.exists()
            assert result["content"] == "Test prompt content"


def test_diff_prompts():
    """Test prompt diffing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test manifests
        manifest1 = PromptManifest(
            content="Hello world",
            version="0.1.0",
            embeddings=[1.0, 0.0, 0.0]
        )
        
        manifest2 = PromptManifest(
            content="Hello universe", 
            version="0.2.0",
            embeddings=[0.8, 0.6, 0.0]  # Similar but different
        )
        
        file1 = Path(temp_dir) / "v1.pp.json"
        file2 = Path(temp_dir) / "v2.pp.json"
        
        file1.write_text(manifest1.model_dump_json())
        file2.write_text(manifest2.model_dump_json())
        
        with patch('sys.exit') as mock_exit:
            result = diff_prompts(file1, file2, threshold=0.9, json_output=True)
            
            assert "similarity" in result
            assert "similarity_percent" in result
            assert "above_threshold" in result
            assert "versions" in result
            assert result["versions"]["a"] == "0.1.0"
            assert result["versions"]["b"] == "0.2.0"
            
            # Should exit with 1 since similarity likely below 0.9
            mock_exit.assert_called_with(1)


def test_calculate_similarity():
    """Test similarity calculation."""
    # Identical vectors should have similarity 1.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    similarity = calculate_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 0.001
    
    # Orthogonal vectors should have similarity 0.0
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    similarity = calculate_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 0.001


def test_validate_manifest_valid():
    """Test manifest validation with valid manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest = {
            "content": "Test prompt",
            "version": "1.0.0",
            "embeddings": [0.1, 0.2, 0.3],
            "model": "gpt-4"
        }
        
        manifest_file = Path(temp_dir) / "test.pp.json"
        manifest_file.write_text(json.dumps(manifest))
        
        with patch('builtins.print') as mock_print:
            result = validate_manifest(manifest_file)
            assert result is True
            mock_print.assert_called_with(f"✓ Manifest {manifest_file} is valid")


def test_validate_manifest_invalid():
    """Test manifest validation with invalid manifest."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Missing required field
        manifest = {
            "content": "Test prompt",
            "version": "1.0.0"
            # Missing embeddings
        }
        
        manifest_file = Path(temp_dir) / "test.pp.json"
        manifest_file.write_text(json.dumps(manifest))
        
        with patch('builtins.print') as mock_print:
            result = validate_manifest(manifest_file)
            assert result is False


def test_validate_manifest_invalid_json():
    """Test manifest validation with invalid JSON."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest_file = Path(temp_dir) / "test.pp.json"
        manifest_file.write_text("invalid json content")
        
        with patch('builtins.print') as mock_print:
            result = validate_manifest(manifest_file)
            assert result is False


def test_pack_prompt_with_provider():
    """Test pack prompt with specific provider."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch('semantic_prompt_packager.core.get_embeddings') as mock_embeddings:
            mock_embeddings.return_value = [0.1, 0.2, 0.3]
            
            prompt_file = Path(temp_dir) / "test.prompt"
            prompt_file.write_text("Test prompt content")
            
            result = pack_prompt(prompt_file, provider="openai")
            
            mock_embeddings.assert_called_once_with("Test prompt content", provider="openai")
            assert result["embeddings"] == [0.1, 0.2, 0.3] 