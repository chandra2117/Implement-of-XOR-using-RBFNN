<H3>NAME : CHANDRAPRIYADHARSHINI C</H3>
<H3>REGISTER NO : 212223240019</H3>
<H3>EX. NO.5</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

```

import numpy as np
import matplotlib.pyplot as plt
# Gaussian RBF
def gaussian_rbf(x, landmark, gamma=1):
    return np.exp(-gamma * np.linalg.norm(x - landmark)**2)
# End-to-end XOR solution
def end_to_end(X1, X2, ys, mu1, mu2):
    # Transform inputs using RBFs
    from_1 = [gaussian_rbf(np.array([x1, x2]), mu1) for x1, x2 in zip(X1, X2)]
    from_2 = [gaussian_rbf(np.array([x1, x2]), mu2) for x1, x2 in zip(X1, X2)]

    # Plot original XOR points
    plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.scatter([X1[0], X1[3]], [X2[0], X2[3]], label="Class_0")
    plt.scatter([X1[1], X1[2]], [X2[1], X2[2]], label="Class_1")
    plt.xlabel("$X1$", fontsize=15)
    plt.ylabel("$X2$", fontsize=15)
    plt.title("XOR: Linearly Inseparable", fontsize=15)
    plt.legend()

    # Plot transformed RBF space
    plt.subplot(1, 2, 2)
    plt.scatter(from_1[0], from_2[0], label="Class_0")
    plt.scatter(from_1[1], from_2[1], label="Class_1")
    plt.scatter(from_1[2], from_2[2], label="Class_1")
    plt.scatter(from_1[3], from_2[3], label="Class_0")
    plt.plot([0, 0.95], [0.95, 0], "k--")
    plt.annotate("Separating hyperplane", xy=(0.4, 0.55), xytext=(0.55, 0.66),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xlabel(f"$mu1$: {mu1}", fontsize=15)
    plt.ylabel(f"$mu2$: {mu2}", fontsize=15)
    plt.title("Transformed Inputs: Linearly Separable", fontsize=15)
    plt.legend()

    # Solve for weights in matrix form: A W = Y
    A = np.array([[f1, f2, 1] for f1, f2 in zip(from_1, from_2)])
    W = np.linalg.pinv(A).dot(ys)  # Pseudo-inverse to handle non-square A
    print(f"Weights: {W}")
    return W, from_1, from_2
# Prediction
def predict_matrix(points, weights, mu1, mu2):
    preds = []
    for p in points:
        f1 = gaussian_rbf(p, mu1)
        f2 = gaussian_rbf(p, mu2)
        val = np.dot([f1, f2, 1], weights)
        preds.append(np.round(val))
    return np.array(preds)
# XOR data
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
ys = np.array([0, 1, 1, 0])
mu1 = np.array([0, 1])
mu2 = np.array([1, 0])
# Train
w, from_1, from_2 = end_to_end(x1, x2, ys, mu1, mu2)

# Test predictions
test_points = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
preds = predict_matrix(test_points, w, mu1, mu2)
for pt, p in zip(test_points, preds):
    print(f"Input: {pt}, Predicted: {int(p)}")

```


<H3>OUTPUT:</H3>


<img width="1183" height="527" alt="image" src="https://github.com/user-attachments/assets/6b9feb56-6df2-44cc-8e1c-83246c54cfc6" />


<img width="339" height="90" alt="image" src="https://github.com/user-attachments/assets/9054090c-afdd-4341-adfa-61ac59185db3" />


<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








