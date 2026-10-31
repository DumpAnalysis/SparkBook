"""Validate laboratory metadata, not checkpoint origin or authenticity."""
BATCH_RULE = "step_seeded_equal_mask_count_v1"

def validate_checkpoint(checkpoint, world):
    if checkpoint.get("world") != world:
        raise ValueError("Checkpoint world size differs")
    if checkpoint.get("batch_rule") != BATCH_RULE:
        raise ValueError("Checkpoint batch rule differs")
