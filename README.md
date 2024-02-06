##A Data Science/Machine Learning Approach to Matrix Inversion
#Brainstation Data Science Bootcamp - Capstone Project 
#Nov 2023-February 2024
#By Ben Heaps

There is a tremendous amount of research being done in the area of Matrix Inversion to improve
the algorithms for Matrix Inversion. Speed of computation is a big issue as well as time complexity.

Purpose of this project is to attempt to find a better matrix inversion algorithm. Modelling
for inverse of Hermitian (symetric) matrices and non Hermitian matrices. By Applying Linear Regression 
to better understand which mathematical combinations of matrix terms are best for modelling the inverse of
a matrix. Use Neural Networks to model Eigenvectors and Eigenvalues which are a property of matrix 
by the formula Av = λv where v=eigenvectors and λ=eigenvalues. 
The matrix inverse relationship with eigenvectors and eigenvalues is (A^-1)v = v/λ

Matrix Inverse can also be calculated by the Adjoint Matrix divided by the Determinant. 
The formula for the determinant of a 2x2 matrix is: |A| = (a11 a22) - (a12 a21)
The formula for the determinant of a 3x3 matrix is:  
|A| = a11 (a22 a33 – a23 a32) – a12 (a21 a33 – a23 a31) + a13 (a21 a32 – a22 a31)
The adjoint of a square matrix A = [aij]n×n is defined as the transpose of the matrix [Aij]n×n , where Aij 
is the cofactor of the element aij. In other words, the transpose of a cofactor matrix of the square matrix 
is called the adjoint of the matrix. 

#################################################################################################

#Linear Regression Investigation

A selection of models attempted to demonstrate the progression of matrix inverse element dependence progression
The following tests demonstrate the progression of combination of matrix elements dependence for the inverse

Test 1: linear regression between matrix elements and inverse matrix elements

Test 2: linear regression with test 1 and all possible combinations of one matrix element 
multiplied by every other matrix element

Test 3: linear regression with test 2 and the square of each matrix element

Test 4: linear regression with test 1 and all possible combinations of one matrix element 
squared and multiplied by every other matrix element and the cube of each matrix element

LinearRegressionMatrixInverse.ipynb 3x3 randomly generated matrices code Linear Regression Investigation
Results summarized in 3x3TestResults.pdf

Test 4 generated a Linear Regression model with an R Squared value of 0.044 averaged across all 9 cells 
of the inverse of a 3x3 matrix. <b>This is not sufficient for determining
the inverse of a 3x3 matrix but the progression of dependence on terms is demonstrated</b

R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent 
variable that’s explained by an independent variable in a regression model.

AutomatedTesting.ipynb performs the right number of iterations of Test4 to generate the mean of r squared value
for each inverse cell to be statistically different than other cells with an acceptable p value 
with an ANOVA Test for Test 4.
Analysis of variance (ANOVA) is a collection of statistical models and their associated estimation procedures 
(such as the "variation" among and between groups) used to analyze the differences among means

Results summarized in AutomatedTesting.pdf

##################################################################################################

#Neural Network Modelling

Work with Neural Networks to model matrix Eigenvectors and Eigenvalues and sum of Eigenvalues and 
sum of Eigenvector elements. 

Investigate Sum of Eigenvalues and Eigenvectors to see if Neural Network can be found and possibly use
this sum in future Neural Network modelling as a term.

Jupyter Notebook EigenVectorsEigenValuesNeuralNetwork.ipynb

Evaluate my Neural Networks with Mean Average Error for the terms using 
datasets of randomly generated 3x3 matrices and corresponding eigenvalues/eigenvectors

------------------------------------------------------------------------------------------------------------------------

Results for 3501 randomly generated matrices with values between 0-6 for <b>Hermitian</b> 3x3 Matrices

['MAE for: eigen1', '0.0855']    eigenvalue 1 Mean Average error<br>
['MAE for: eigen2', '0.0529']    eigenvalue 2 Mean Average error<br>
['MAE for: eigen3', '0.0474']    eigenvalue 3 Mean Average error<br>
['MAE for: eigenv1', '0.0171']   eigenvector element 1 of 9 Mean Average Error<br>
['MAE for: eigenv2', '0.0294']   eigenvector element 2 of 9 Mean Average Error<br>
['MAE for: eigenv3', '0.0266']   eigenvector element 3 of 9 Mean Average Error<br>
['MAE for: eigenv4', '0.0163']   eigenvector element 4 of 9 Mean Average Error<br>
['MAE for: eigenv5', '0.0281']   eigenvector element 5 of 9 Mean Average Error<br>
['MAE for: eigenv6', '0.0245']   eigenvector element 6 of 9 Mean Average Error<br>
['MAE for: eigenv7', '0.0152']   eigenvector element 7 of 9 Mean Average Error<br>
['MAE for: eigenv8', '0.0245']   eigenvector element 8 of 9 Mean Average Error<br>
['MAE for: eigenv9', '0.0271']   eigenvector element 9 of 9 Mean Average Error<br>
['sumEigenvectorMAE', '0.0525']  sum of eigen vector elements Mean Average Error<br>
['sumEigenvaluesMAE', '0.0499']  sum of all 9 eigenvalue elements from 3 eigenvectors Mean Average Error<br><br>

Mean Average Error as a percentage against range of values in the terms <br>

['%MAE for: eigen1', '0.0122']   eigenvalue 1 Mean Average error As Percentage versus range of values<br>
['%MAE for: eigen2', '0.0640']   eigenvalue 2 Mean Average error As Percentage versus range of values<br>
['%MAE for: eigen3', '0.0424']   eigenvalue 3 Mean Average error As Percentage versus range of values<br>
['%MAE for: eigenv1', '0.0086']  eigenvector element 1 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv2', '0.0147']  eigenvector element 2 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv3', '0.0134']  eigenvector element 3 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv4', '0.0163']  eigenvector element 4 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv5', '0.0141']  eigenvector element 5 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv6', '0.0123']  eigenvector element 6 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv7', '0.0077']  eigenvector element 7 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv8', '0.0123']  eigenvector element 8 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv9', '0.0136']  eigenvector element 9 of 9 Mean Average Error As Percentage versus range of values<br>
['sumEigenvectorMAE%', '0.0067'] sum of eigen vector elements Mean Average Error As Percentage versus range of values<br>
['sumEigenvaluesMAE%', '0.0028'] sum of all 9 eigenvalue elements from 3 eigenvectors Mean Average Error As Percentage versus range of values<br>

---------------------------------------------------------------------------------------------------------------------------------

Results for 3501 randomly generated matrices with values between 0-6 for <b>Non Hermitian Matrices</b> 3x3 matrices

['MAE for: eigen1', '0.0898']   eigenvalue 1 Mean Average error<br>
['MAE for: eigen2', '0.0486']   eigenvalue 2 Mean Average error<br>
['MAE for: eigen3', '0.0530']   eigenvalue 3 Mean Average error<br>
['MAE for: eigenv1', '0.0183']  eigenvector element 1 of 9 Mean Average Error<br>
['MAE for: eigenv2', '0.0324']  eigenvector element 2 of 9 Mean Average Error<br>
['MAE for: eigenv3', '0.0288']  eigenvector element 3 of 9 Mean Average Error<br>
['MAE for: eigenv4', '0.0167']  eigenvector element 4 of 9 Mean Average Error <br>
['MAE for: eigenv5', '0.0247']  eigenvector element 5 of 9 Mean Average Error<br>
['MAE for: eigenv6', '0.0274']  eigenvector element 6 of 9 Mean Average Error<br>
['MAE for: eigenv7', '0.0145']  eigenvector element 7 of 9 Mean Average Error<br>
['MAE for: eigenv8', '0.0244']  eigenvector element 8 of 9 Mean Average Error<br>
['MAE for: eigenv9', '0.0290']  eigenvector element 9 of 9 Mean Average Error<br>
['sumEigenvectorMAE', '0.0574'] sum of eigen vector elements Mean Average Error<br>
['sumEigenvaluesMAE', '0.0454'] sum of all 9 eigenvalue elements from 3 eigenvectors Mean Average Error<br>

['%MAE for: eigen1', '0.0128']  eigenvalue 1 Mean Average error As Percentage versus range of values<br>
['%MAE for: eigen2', '0.0587']  eigenvalue 2 Mean Average error As Percentage versus range of values<br>
['%MAE for: eigen3', '0.0474']  eigenvalue 3 Mean Average error As Percentage versus range of values<br>
['%MAE for: eigenv1', '0.0092'] eigenvector element 1 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv2', '0.0162'] eigenvector element 2 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv3', '0.0144'] eigenvector element 3 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv4', '0.0084'] eigenvector element 4 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv5', '0.0124'] eigenvector element 5 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv6', '0.0137'] eigenvector element 6 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv7', '0.0073'] eigenvector element 7 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv8', '0.0122'] eigenvector element 8 of 9 Mean Average Error As Percentage versus range of values<br>
['%MAE for: eigenv9', '0.0146'] eigenvector element 9 of 9 Mean Average Error As Percentage versus range of values<br>
['sumEigenvectorMAE%', '0.0074']sum of eigen vector elements Mean Average Error As Percentage versus range of values<br>
['sumEigenvaluesMAE%', '0.0025']sum of all 9 eigenvalue elements from 3 eigenvectors Mean Average Error As Percentage versus range of values<br>

Hermitian Matrices Model is better however % error based on term ranges are very similar

<b>Considering the average of all Neural Networks results from this notebook, Hermitian matrices 
had an error of 0.03 or 1.7% error considering the range of values
Non Hermitian matrices to an error of 0.036 for matrices with values and 1.69% error</b>

###########################################################################################################################################

LinearRegressionAdjoint.ipynb 
3x3 Adjoint of a matrix can be perfectly approximated by terms of each matrix element multiplied by every other matrix element 
but this is just a property of adjoint matrices for 3x3 matrices.

############################################################################################################################################

NEXT STEPS:

More investigation into tweaking my neural network model parameters to get a lower Mean Average Error
More investigation for larger adjoint matrices
More investigation for neural networks for inverse of a matrix, larger matrices for eigenvectors eigenvalues
neural networks
More investigation on combining multiple machine learning techniques to a single solution
Plug in calculaalte models to numpy and see how fast they are compared to current formulas
More mathematical investigation into current formulas for matrix inversion and applying terms to machine learning







