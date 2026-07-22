#!/bin/bash

# Define the list of tasks
tasks=(
    "close_jar"
    "insert_onto_square_peg"
    "light_bulb_in"
    "meat_off_grill"
    "open_drawer"
    "place_cups"
    "place_shape_in_shape_sorter"
    "push_buttons"
    "put_groceries_in_cupboard"
    "put_item_in_drawer"
    "put_money_in_safe"
    "reach_and_drag"
    "stack_blocks"
    "stack_cups"
    "turn_tap"
    "place_wine_at_rack_location"
    "slide_block_to_color_target"
    "sweep_to_dustpan_of_size"
)

# Loop through each task
for task in "${tasks[@]}"; do
    echo "=========================================="
    echo "Processing task: $task"
    echo "=========================================="
    
    python qwen3vl_rlbench_episode_grounding.py \
        --episode-dir "data/BridgeVLA_RLBench_EVAL_DATA/${task}/all_variations/episodes/episode0/" \
        --output-dir "./role_grounding_output/${task}" --stride 20

    echo "Completed task: $task"
    echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="