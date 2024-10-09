# ECN753 - Homework 1

### Content
- Data: Contains csv data files for agents and products
- Figures: Contains the figures produced with the python code
- HW1.py: Python program for generating results. Run time is about 40 minutes
- HW1.pdf: PDF file with results for homework 1
- homework_1.pdf: PDF file with problem description and quastions for homework 1

### Environment
The following packages were used
- numpy: version 1.21.6
- pandas: version 0.25.1
- linearmodels: version 4.25
- sklearn: version 0.0
- matplotlib: version 3.1.3
- scipy: version 1.3.1
- time
  
I found that more recent versions of the linearmodels package may result in conflicts with numpy when using the panel module.

###### Note:
For qualitative results, there was no much difference between using a tolerance level of 1e-5 and 1e-12 in line 198, and the former is much faster (it takes about 10 min). However, there are small changes in estimated coefficients of the BLP model although unlikely to be significant.
