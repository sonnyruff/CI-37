function f = perceptron(x, w,theta)
    sum = dot(x, w);
    f = step(sum-theta);
end