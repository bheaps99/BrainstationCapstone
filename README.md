# BrainstationCapstone

By Ben Heaps

Matrix Inversion Investigation

There is a tremendous amount of research being done in this area at this time to improve
many algorithms for matrix inversion. Speed of computation is a big issue and better algorithms
can invert matrices that are too complex to currently invert or take too much time

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

R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent 
variable that’s explained by an independent variable in a regression model.

a good linear regression will have a r squared value of 0.2, a perfect fit will have a value of 1

So I have not found a suitable linear regresssion model
-----------------------------------------------------------------------------------------------






I will work with Neural Networks to model matrix eigenvectors and eigenvalues and sum of 
eigenvalues and sum of eigenvector values as they can play a part in matrix inverse by the 
following formula: Av = A^    ^ is lambda or eigenvalues
modelling the eigenvectors and eigenvalues can be used for future study of matrix
inversion with machine learning

details of my model and work can be found in EigenVectorsEigenValuesNeuralNetwork.ipynb

I have evaluated my neural networks with Mean Average Error for the terms usring the test 
dataset of randomly generated 3x3 matrices and the corresponding eigenvalues/eigenvectors

-----------------------------------------------------------------------------------------------

the results are for a test with 3501 matrices with values between 0-6 for Hermitian Matrices
REMEMBER WE ARE WORKING WITH 3x3 matrices

[['MAE for: eigen1', '0.0855'],    eigenvalue 1 Mean Average error
 ['MAE for: eigen2', '0.0529'],    eigenvalue 2 Mean Average error
 ['MAE for: eigen3', '0.0474']     eigenvalue 3 Mean Average error
 ['MAE for: eigenv1', '0.0171'],   eigenvector element 1 of 9 Mean Average Error
 ['MAE for: eigenv2', '0.0294'],   eigenvector element 2 of 9 Mean Average Error
 ['MAE for: eigenv3', '0.0266'],   eigenvector element 3 of 9 Mean Average Error
 ['MAE for: eigenv4', '0.0163'],   eigenvector element 4 of 9 Mean Average Error
 ['MAE for: eigenv5', '0.0281'],   eigenvector element 5 of 9 Mean Average Error
 ['MAE for: eigenv6', '0.0245'],   eigenvector element 6 of 9 Mean Average Error
 ['MAE for: eigenv7', '0.0152'],   eigenvector element 7 of 9 Mean Average Error
 ['MAE for: eigenv8', '0.0245'],   eigenvector element 8 of 9 Mean Average Error
 ['MAE for: eigenv9', '0.0271'],   eigenvector element 9 of 9 Mean Average Error
 ['sumEigenvectorMAE', '0.0525'],  sum of eigen vector elements Mean Average Error
 ['sumEigenvaluesMAE', '0.0499']]  sum of all 9 eigenvalue elements from 3 eigenvectors Mean Average Error

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
[['MAE for: eigen1', '0.0898'],   eigenvalue 1 Mean Average error
 ['MAE for: eigen2', '0.0486'],   eigenvalue 2 Mean Average error
 ['MAE for: eigen3', '0.0530'],   eigenvalue 3 Mean Average error
 ['MAE for: eigenv1', '0.0183'],  eigenvector element 1 of 9 Mean Average Error
 ['MAE for: eigenv2', '0.0324'],  eigenvector element 2 of 9 Mean Average Error
 ['MAE for: eigenv3', '0.0288'],  eigenvector element 3 of 9 Mean Average Error
 ['MAE for: eigenv4', '0.0167'],  eigenvector element 4 of 9 Mean Average Error
 ['MAE for: eigenv5', '0.0247'],  eigenvector element 5 of 9 Mean Average Error
 ['MAE for: eigenv6', '0.0274'],  eigenvector element 6 of 9 Mean Average Error
 ['MAE for: eigenv7', '0.0145'],  eigenvector element 7 of 9 Mean Average Error
 ['MAE for: eigenv8', '0.0244'],  eigenvector element 8 of 9 Mean Average Error
 ['MAE for: eigenv9', '0.0290'],  eigenvector element 9 of 9 Mean Average Error
 ['sumEigenvectorMAE', '0.0574'], sum of eigen vector elements Mean Average Error
 ['sumEigenvaluesMAE', '0.0454']] sum of all 9 eigenvalue elements from 3 eigenvectors Mean Average Error

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

My model for Hermitian Matrices is better, however % error based on term ranges is very similar

For Hermitian matrices to an error of 0.03 for matrices with values between 0-6 and 1.7% error
For Non Hermitian matrice to an error of 0.036 for matrices with values between 0-6 and 1.69% error







LinearRegressionAdjoint.ipynb shows how the adjoint matrix can be perfectly approximated by terms of 
each matrix element multiplied by every other matrix element but this is just a property of adjoint 
matrices for 3x3 matrices

a formula for the inverse of a matrix is adjoint matrix/determinant

determinant formula is |A| = a11 (a22 a33 – a23 a32) – a12 (a21 a33 – a23 a31) + a13 (a21 a32 – a22 a31)
The adjoint of a square matrix A = [aij]n×n is defined as the transpose of the matrix [Aij]n×n , where Aij 
is the cofactor of the element aij. In other words, the transpose of a cofactor matrix of the square matrix 
is called the adjoint of the matrix. 

a formlua for the adjoint of a matrix is as follows

import math as mth
# get cofactors matrix 
def getcofat(x):
    eps = 1e-6
    detx = np.linalg.det(x)
    if (mth.fabs(detx) < eps):
        print("No possible to get cofactors for singular matrix with this method")
        x = None
        return x
    invx = np.linalg.pinv(x)
    invxT = invx.T
    x = invxT * detx
    return x
# get adj matrix
def getadj(x):
    eps = 1e-6
    detx = np.linalg.det(x)
    if (mth.fabs(detx) < eps):
        print("No possible to get adj matrix for singular matrix with this method")
        adjx = None
        return adjx
    cofatx = getcofat(x)
    adjx = cofatx.T
    return adjx


NEXT STEPS:

More investigation into tweaking my neural network model parametets to get a lower Mean Average Error
More investigation for larger adjoint matrices
More investigation for neural networks for inverse of a matrix, larger matrices for eigenvectors eigenvalues
neural networks
Plug in calcualte models to numpy and see how fast they are compared to current formulas
More mathematical investigation into current formulas for matrix inversion and applying terms to machine learning







