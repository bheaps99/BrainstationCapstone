# BrainstationCapstone

By Ben Heaps

Matrix Inversion Investigation

There is a tremendous amount of research being done in this area at this time to improve
many algorithms for matrix inversion. 

I will attempt to find a data science solution to finding a better matrix inversion algorithm.
I will run tests with Hermitian (symetric) matrices and non symetric matrices. I will work with
3x3 matrices for now and will possibly work with larger matrices in future steps. 




I will work with linear regression with various tests, I will include the following tests:

Test 1: linear regression between matrix elements and inverse matrix elements

Test 2: linear regression with test 1 and all possible combinations of one matrix element 
multiplied by every other matrix element

Test 3: linear regression with test 2 and the square of each matrix element

Test 4: linear regression with test 1 and all possible combinations of one matrix element 
squared and multiplied by every other matrix element and the cube of each matrix element

Jupyter Notebook LinearRegressionMatrixInverse.ipynb will be used for this Analysis

Results will be summarized in 3x3TestResults.pdf

In AutomatedTesting.ipynb perform the right number of iterations to generate the mean of r squared value
for each inverse cell to be statistically different than the others with an acceptable p value 
with an ANOVA Test for Test 4

Results will be summarized in AutomatedTesting.pdf

------------------------------------------------------------------------------------------------
I found that with Test 4 I could generate a Linear Regression model with an R Squared value of
0.044 averaged across all 9 cells of a 3x3 matrix. This is not sufficient for determining
the inverse of a 3x3 matrix but it is interesting nonetheless
-----------------------------------------------------------------------------------------------



I will work with Neural Networks to model matrix eigenvectors and eigenvalues and sum of 
eigenvalues and sum of eigenvector values as they can play a part in matrix inverse by the 
following formula: Av = A^    ^ is lambda or eigenvalues

details of my model and work can be found in EigenVectorsEigenValuesNeuralNetwork.ipynb

-----------------------------------------------------------------------------------------------

the results are for a test with 3501 matrices with values between 0-6 for Hermitian Matrices

[['MAE for: eigen1', '0.0855'],    eigenvalue 1 Mean Average error
 ['MAE for: eigen2', '0.0529'],
 ['MAE for: eigen3', '0.0474'],
 ['MAE for: eigenv1', '0.0171'],   eigenvector element 1 of 9 Mean Average Error
 ['MAE for: eigenv2', '0.0294'],
 ['MAE for: eigenv3', '0.0266'],
 ['MAE for: eigenv4', '0.0163'],
 ['MAE for: eigenv5', '0.0281'],
 ['MAE for: eigenv6', '0.0245'],
 ['MAE for: eigenv7', '0.0152'],
 ['MAE for: eigenv8', '0.0245'],
 ['MAE for: eigenv9', '0.0271'],
 ['sumEigenvectorMAE', '0.0525'],
 ['sumEigenvaluesMAE', '0.0499']]

 [['%MAE for: eigen1', '0.0122'],    Mean Average Error as a percentage against range of values of the terms
 ['%MAE for: eigen2', '0.0640'],
 ['%MAE for: eigen3', '0.0424'],
 ['%MAE for: eigenv1', '0.0086'],
 ['%MAE for: eigenv2', '0.0147'],
 ['%MAE for: eigenv3', '0.0134'],
 ['%MAE for: eigenv4', '0.0083'],
 ['%MAE for: eigenv5', '0.0141'],
 ['%MAE for: eigenv6', '0.0123'],
 ['%MAE for: eigenv7', '0.0077'],
 ['%MAE for: eigenv8', '0.0123'],
 ['%MAE for: eigenv9', '0.0136'],
 ['sumEigenvectorMAE%', '0.0067'],
 ['sumEigenvaluesMAE%', '0.0028']]

 ---------------------------------------------

the results are for a test with 3501 matrices with values between 0-6 for Non Hermitian Matrices

resultsList
[['MAE for: eigen1', '0.0898'],
 ['MAE for: eigen2', '0.0486'],
 ['MAE for: eigen3', '0.0530'],
 ['MAE for: eigenv1', '0.0183'],
 ['MAE for: eigenv2', '0.0324'],
 ['MAE for: eigenv3', '0.0288'],
 ['MAE for: eigenv4', '0.0167'],
 ['MAE for: eigenv5', '0.0247'],
 ['MAE for: eigenv6', '0.0274'],
 ['MAE for: eigenv7', '0.0145'],
 ['MAE for: eigenv8', '0.0244'],
 ['MAE for: eigenv9', '0.0290'],
 ['sumEigenvectorMAE', '0.0574'],
 ['sumEigenvaluesMAE', '0.0454']]
))
total=0 
for i in resultsList:
    total+=float(i[1])
print("Mean: "+ str(total/len(resultsList)))
    
Mean: 0.036457142857142855
resultsListMAE
[['%MAE for: eigen1', '0.0128'],
 ['%MAE for: eigen2', '0.0587'],
 ['%MAE for: eigen3', '0.0474'],
 ['%MAE for: eigenv1', '0.0092'],
 ['%MAE for: eigenv2', '0.0162'],
 ['%MAE for: eigenv3', '0.0144'],
 ['%MAE for: eigenv4', '0.0084'],
 ['%MAE for: eigenv5', '0.0124'],
 ['%MAE for: eigenv6', '0.0137'],
 ['%MAE for: eigenv7', '0.0073'],
 ['%MAE for: eigenv8', '0.0122'],
 ['%MAE for: eigenv9', '0.0146'],
 ['sumEigenvectorMAE%', '0.0074'],
 ['sumEigenvaluesMAE%', '0.0025']]
total=0 
for i in resultsListMAE:
    total+=float(i[1])
print("Mean: "+str(total/len(resultsListMAE)))
Mean: 0.016942857142857138



LinearRegressionAdjoint.ipynb shows how the adjoint matrix can be perfectly approximated by terms of 
each matrix element multiplied by every other matrix element but this is just a property of adjoint 
matrices for 3x3 matrices

the formula for the inverse of a matrix is adjoint matrix/determinant


NEXT STEPS:

more investigation for larger adjoint matrices
more investigation for neural netoworks for direct inverse calculation, larger matrices eigenvectors eigenvalues
plug in calcualte models to numpy and see how fast they are compared to current formulas
more mathematical investigation







