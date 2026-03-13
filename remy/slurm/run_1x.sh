#!/bin/bash
#SBATCH --job-name=remy-cca
#SBATCH --account=iris
#SBATCH --partition=sc-freecpu
#SBATCH --time=200:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2G
#SBATCH --output=logs/remy_%j.out
#SBATCH --error=logs/remy_%j.err


# Constants - Configure these for your specific run
WORK_DIR="."
OUTPUT_DIR="1x-2src"
CONFIG_FILE="link-1x.cfg"
OUTPUT_PREFIX="cca"

# Change to working directory
cd "$WORK_DIR" || exit 1

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Find the CCA file with the highest index in the output directory
HIGHEST_CCA=""
HIGHEST_INDEX=-1

# Look for files matching the pattern OUTPUT_PREFIX.N in the output directory
for file in "$OUTPUT_DIR"/"$OUTPUT_PREFIX".*; do
    # Check if file exists (pattern might not match anything)
    if [[ -f "$file" ]]; then
        # Extract the numeric suffix
        suffix="${file##*.}"
        # Check if suffix is a number
        if [[ "$suffix" =~ ^[0-9]+$ ]]; then
            # Compare with current highest
            if (( suffix > HIGHEST_INDEX )); then
                HIGHEST_INDEX=$suffix
                HIGHEST_CCA="$file"
            fi
        fi
    fi
done

# Build the remy command
REMY_CMD="./remy cf=$CONFIG_FILE of=$OUTPUT_DIR/$OUTPUT_PREFIX"

# Add initial CCA file if found
if [[ -n "$HIGHEST_CCA" ]]; then
    echo "Found existing CCA file: $HIGHEST_CCA (index: $HIGHEST_INDEX)"
    echo "Continuing from run $((HIGHEST_INDEX + 1))"
    REMY_CMD="$REMY_CMD if=$HIGHEST_CCA"
else
    echo "No existing CCA files found in $OUTPUT_DIR/"
    echo "Starting fresh run from index 0"
fi

# Log the command being executed
echo "Executing: $REMY_CMD"
echo "Starting time: $(date)"

# Execute remy
$REMY_CMD

# Log completion
echo "Completed at: $(date)"
echo "Exit code: $?"
