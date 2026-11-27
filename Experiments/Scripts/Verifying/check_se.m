N_sims = 5;
rng(42);
for sim = 1:N_sims
    factory = OceanStats.Data.DataFactory('Seed', 42 + sim);
    [X_train1, Y_train1, mu_train1] = factory.create("ComplexFunctions", "N", 1000, "Function", "TensorProduct", "Features", 2, "Sigma", 0.12);
    [X_test1,  ~,        mu_test1]  = factory.create("ComplexFunctions", "N", 1000,  "Function", "TensorProduct", "Features", 2, "Sigma", 0.12);
    mars1 = OceanStats.Supervised.MARS.MARS('MaxInteractionDegree', 2, 'MaxTerms', 50, 'PenaltyCost', 3.0);
    mars1.fit(X_train1, Y_train1);
    MSE0_1 = mean((mean(Y_train1) - mu_test1).^2);
    MSE_1  = mean((mars1.predict(X_test1) - mu_test1).^2);
    R2_S1(sim) = (MSE0_1 - MSE_1) / MSE0_1;
end
disp(R2_S1);
disp(std(R2_S1)/sqrt(N_sims));
