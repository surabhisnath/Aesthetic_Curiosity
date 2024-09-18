# Cox regression models
models <- list(
    "1",
    "pid",
    "click_id",
    "pattern_id",
    "pid + click_id + pattern_id"
)

# Save all results to model_fits/Table_1_survival.csv
{
    df <- data.frame(matrix(ncol = 4, nrow = 0, dimnames =
    list(NULL, c("Id", "model", "AIC", "BIC"))))

    id <- 0
    for(formula in models) 
    {
        id <- id + 1
        fullformula <- paste("time_taken ~ ", formula)
        formula <- as.formula(fullformula)
        
        # calcultate the following metrics
        AICs <- numeric(num_folds)
        BICs <- numeric(num_folds)

        for(x in 1:num_folds)
        {
            model <- lm(formula, data = data_train_folds[[x]])

            AICs[x] <- AIC(model)
            BICs[x] <- BIC(model)
        }

        df[nrow(df) + 1, ] <- c(id, noquote(fullformula), c(mean(AICs), mean(BICs)))
    }

    write.csv(df, "model_fits/Table_1_survival.csv", row.names = FALSE)
}


{
  
  bestformula <- "time_taken ~ pid + click_id"

  f <- lm(bestformula, data = data)
}

# Save fixed effects to model_fits/ - Table 4
{
  model_summary <- summary(f)
  coefficients <- coef(f)
  standard_errors <- sqrt(diag(vcov(f)))
  variable_names <- rownames(summary(f)$coefficients)
  p_values <- coef(summary(f))[, "Pr(>|t|)"]
  signif_levels <- ifelse(p_values < 0.001, "***", ifelse(p_values < 0.01, "**", ifelse(p_values < 0.05, "*", ifelse(p_values < 0.1, ".", "NS"))))
  results_df <- data.frame(Variable = variable_names, Coefficients = coefficients, StdError = standard_errors, PValue = p_values, Significance = signif_levels)
  write.csv(results_df, file = "model_fits/Table_4_coeff_mixedeffects.csv", row.names = FALSE)
}