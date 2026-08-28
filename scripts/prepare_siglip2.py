"""Download the pinned public SigLIP2 model for automatic-crop diagnostics."""
from prepare_grounding_dino import download_release
from prepare_keen_components import ROOT

if __name__ == "__main__":
    download_release("google/siglip2-base-patch16-naflex",
        "b53b807d3a2d5e2b3911292f2d69e5341cdc064c",ROOT/"weights/siglip2-base-naflex-b53b807",
        ROOT/"runtime/target_sources/siglip2_base_api.json")
