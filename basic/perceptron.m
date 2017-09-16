function f = perceptron(x, w)
    sum = dot(x, w);
    f = step(sum);
end