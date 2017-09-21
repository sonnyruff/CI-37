function f = perceptron( x, w, theta )
    sum = dot(x, w);
    f = sigmoidFunc(sum - theta);
end