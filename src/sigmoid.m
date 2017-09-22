function f = sigmoid(a, b)
    f = 1.0 ./ ( 1.0 + exp(-dot(a, b)));
end