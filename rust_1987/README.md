# Rust 1987

## How to Run the Code
The environment is still preserved in the main ECN_753 folder. To run it, please download the whole branch, change the working directory to the main folder i.e. ECN_753. Then activate the environment (e.g. anaconda) and use an IDE with the activated environment python compiler to run the code.

## Code Details

### Code Structure
The code is compartmentalized. The scripts hold related function that are called through the main script.

### Speed and Efficiency
The speed is about 5 minutes with initial values of (1, 1) but is faster through closer values to the "pseudo" source of truth given the pseudo data. The speed can be improved with efficient function coding, which unfortunately suffers in these scripts.

### Improvement Ideas
Specific scripts may not require call of other scripts, due to all being called in main. I did not have time to check it to imrpove time.

Further, the text for the part b is called through config, which does not sit so right, and needs to be improved.