#!/bin/bash
#SBATCH --job-name=model_opt            # Job name
#SBATCH --array=1-10                    # Job array with 10 tasks
#SBATCH --output=logs/output_%A_%a.out  # Standard output log
#SBATCH --error=logs/error_%A_%a.err    # Standard error log
#SBATCH --time=04:00:00                 # Time limit hrs:min:sec
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --mem=500G                      # Memory per CPU

# Load necessary modules
module load matlab/latest               # Load MATLAB module (adjust version as needed)

# Create an array of initial alpha_income values
alpha_values=(-33.5 -33.4 -33.3 -33.2 -33.1 -33.0 -32.9 -32.8 -32.7 -32.6)

# Get the initial value for this sub-job
alpha_income_init=${alpha_values[$SLURM_ARRAY_TASK_ID-1]}

echo "Execute optimize_model_SOL with alpha_income_init=${alpha_income_init}"

# Run the MATLAB optimization script
matlab -nodisplay -r "optimize_model_SOL(${alpha_income_init}); exit;"
