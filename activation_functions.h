struct ActivationFunction {
    double (*Forwards)(double);
    double (*Derivative)(double);
};

extern ActivationFunction Sigmoid;
extern ActivationFunction Relu;
extern ActivationFunction Identity;