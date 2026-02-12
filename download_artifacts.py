"""
Download model artifacts from Weights & Biases

This script downloads the latest model artifacts from W&B
and extracts them to the artifacts directory.
"""

import os
import wandb
import shutil
from pathlib import Path

# Configuration
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "mindsync-model")
WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
ARTIFACT_NAME = "mindsync-model"
ARTIFACT_VERSION = os.getenv("ARTIFACT_VERSION", "latest")  # 'latest' or specific version like 'v0', 'v1'


def download_artifacts(output_dir="artifacts"):
    """
    Download model artifacts from Weights & Biases.
    
    Args:
        output_dir: Directory to save downloaded artifacts
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print("=" * 60)
        print("☁️ Downloading artifacts from Weights & Biases...")
        print("=" * 60)
        
        # Initialize W&B API
        api = wandb.Api()
        
        # Construct artifact path
        if WANDB_ENTITY:
            artifact_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{ARTIFACT_NAME}:{ARTIFACT_VERSION}"
        else:
            artifact_path = f"{WANDB_PROJECT}/{ARTIFACT_NAME}:{ARTIFACT_VERSION}"
        
        print(f"📦 Fetching artifact: {artifact_path}")
        
        # Download artifact
        artifact = api.artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        
        print(f"✅ Artifact downloaded to: {artifact_dir}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy files to output directory
        print(f"📂 Copying files to {output_dir}/...")
        
        # Files to preserve (don't overwrite if they exist locally)
        preserve_files = ["healthy_cluster_avg.csv"]
        
        files_copied = 0
        files_skipped = 0
        for file in Path(artifact_dir).glob("*"):
            if file.is_file():
                dest = output_path / file.name
                
                # Skip if file should be preserved and already exists locally
                if file.name in preserve_files and dest.exists():
                    print(f"  ⏭️  {file.name} (preserved local version)")
                    files_skipped += 1
                    continue
                
                shutil.copy2(file, dest)
                print(f"  ✓ {file.name}")
                files_copied += 1
        
        print(f"\n✅ Successfully copied {files_copied} files")
        if files_skipped > 0:
            print(f"⏭️  Preserved {files_skipped} local files (not overwritten)")
        
        # Display artifact metadata
        print("\n📊 Artifact metadata:")
        metadata = artifact.metadata
        for key, value in metadata.items():
            print(f"  • {key}: {value}")
        
        print("\n" + "=" * 60)
        print("✅ Artifacts download complete!")
        print("=" * 60)
        
        return True
        
    except wandb.errors.CommError as e:
        print(f"❌ W&B Communication Error: {e}")
        print("   Make sure you're logged in: wandb login")
        return False
        
    except Exception as e:
        print(f"❌ Error downloading artifacts: {e}")
        return False


def verify_artifacts(artifacts_dir="artifacts"):
    """
    Verify that all required artifacts are present.
    
    Args:
        artifacts_dir: Directory containing artifacts
    
    Returns:
        bool: True if all artifacts present, False otherwise
    """
    required_files = [
        "model.pkl",
        "preprocessor.pkl",
        "model_coefficients.csv",
    ]
    
    # healthy_cluster_avg.csv is maintained locally, not from W&B
    # So we don't require it to be downloaded
    
    artifacts_path = Path(artifacts_dir)
    
    if not artifacts_path.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return False
    
    missing_files = []
    for file in required_files:
        if not (artifacts_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing artifacts: {', '.join(missing_files)}")
        return False
    
    print(f"✅ All required artifacts present in {artifacts_dir}/")
    return True


def main():
    """Main function to download and verify artifacts."""
    # Determine artifacts directory relative to this script
    script_dir = Path(__file__).parent
    artifacts_dir = script_dir / "artifacts"
    
    # Download artifacts
    success = download_artifacts(str(artifacts_dir))
    
    if not success:
        print("\n⚠️ Failed to download artifacts from W&B")
        print("   Checking if local artifacts exist...")
        
        if verify_artifacts(str(artifacts_dir)):
            print("   Using existing local artifacts")
            return True
        else:
            print("   No valid artifacts found!")
            return False
    
    # Verify downloaded artifacts
    return verify_artifacts(str(artifacts_dir))


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
