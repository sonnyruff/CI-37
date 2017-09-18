function f = perceptron( x, w )
    threshold = 1;

    sum = dot(x, w);
    f = stepFunc(sum - threshold);
end