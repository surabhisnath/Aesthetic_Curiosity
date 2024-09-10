## Author: Surabhi S Nath
## Description: This script implements mixed effects models.
## on grid-search exploration data.
## Helper functions and code for plotting in utils/Utils.R
## model analysis table written to model_fits/
## model plots written to plots/

# Imports
{
  library(lme4)
  library(nlme)
  library("afex")
  library(lmerTest)
  library(sjPlot)
  library(sjmisc)
  library(ggplot2)
  library(insight)
  library(lattice)
  library(fitdistrplus)
  library(corrplot)
  library(MASS)
  library(merTools)
  library(car)
  library(jtools)
  library(interactions)
  library(modelr)

  source("Utils.R")

  # Set seed 
  set.seed(20) # Seed set randomly for reproducibility
}

# Setup
{
  # Read data
  data <- read.csv("../../csvs/grid-search/grid_data_withoutliers.csv")

  # Scale all variables in the data
  my_scale <- function(x) {
    as.numeric(scale(x))
  }

  data$Subject <- factor(data$Subject)
  data$grid_id <- factor(data$grid_id)
  data$pattern <- factor(data$pattern)
  data$uLSC <- my_scale(data$underlying_LSC)
  data$uLSCsq <- my_scale(data$underlying_LSC ^ 2)
  data$uInt <- my_scale(data$underlying_intricacy)
  data$uIntsq <- my_scale(data$underlying_intricacy ^ 2)
  data$fLSC <- my_scale(data$final_LSC)
  data$fLSCsq <- my_scale(data$final_LSC ^ 2)
  data$fInt <- my_scale(data$final_intricacy)
  data$fIntsq <- my_scale(data$final_intricacy ^ 2)
  data$chLSC <- my_scale(data$final_change_in_LSC)
  data$chInt <- my_scale(data$final_change_in_intricacy)
  data$avgchLSC <- my_scale(data$avg_change_in_LSC)
  data$avgchInt <- my_scale(data$avg_change_in_intricacy)
}

# Make correlation plot - not used in paper
{
    selected_data <- data[, c("num_clicks", "uLSC", "fLSC", "uInt", "fInt", "uLSCsq", "fLSCsq", "uIntsq", "fIntsq", "chLSC", "chInt", "avgchLSC", "avgchInt")]
    correlation_matrix <- cor(selected_data)
    colnames(correlation_matrix) <- c("num_clicks", "uLSC", "fLSC", "uInt", "fInt", "uLSCsq", "fLSCsq", "uIntsq", "fIntsq", "chLSC", "chInt", "avgchLSC", "avgchInt") # Rename columns
    rownames(correlation_matrix) <- c("num_clicks", "uLSC", "fLSC", "uInt", "fInt", "uLSCsq", "fLSCsq", "uIntsq", "fIntsq", "chLSC", "chInt", "avgchLSC", "avgchInt") # Rename columns
    correlation_matrix[lower.tri(correlation_matrix)] <- NA
    pdf(file = "plots/grid_level_correlations.pdf", width = 12, height = 12)
    par(mar=c(7,7,2,2), cex = 1.3, family="serif")

    # Plot the heatmap
    image(1:ncol(correlation_matrix), 1:nrow(correlation_matrix), t(correlation_matrix), 
        col = colorRampPalette(c("blue", "white", "red"))(20), axes = FALSE, xlab = "", ylab = "")

    # Add text annotations for the correlation values
    for (i in 1:nrow(correlation_matrix)) {
        for (j in 1:ncol(correlation_matrix)) {
            if (!is.na(correlation_matrix[i, j])) {
                text(j, i, round(correlation_matrix[i, j], 2), cex = 1.3)
            }
        }
    }

    # Customize the row and column names
    axis(1, at = 1:ncol(correlation_matrix), labels = colnames(correlation_matrix), las = 2, cex.axis = 1.4, family = "serif")
    axis(2, at = 1:nrow(correlation_matrix), labels = rownames(correlation_matrix), las = 2, cex.axis = 1.4, family = "serif")

    # save correlation plot
    dev.off()
}

# Split the data into 3 stratified folds
{
  num_folds <- 3

  folds <- data %>%
  group_by(Subject) %>%
  modelr::crossv_kfold(k = num_folds)

  temp_train_1 <- folds$train$`1`
  data_train_fold1 <- data[temp_train_1$idx, ]

  temp_test_1 <- folds$test$`1`
  data_test_fold1 <- data[temp_test_1$idx, ]

  temp_train_2 <- folds$train$`2`
  data_train_fold2 <- data[temp_train_2$idx, ]

  temp_test_2 <- folds$test$`2`
  data_test_fold2 <- data[temp_test_2$idx, ]

  temp_train_3 <- folds$train$`3`
  data_train_fold3 <- data[temp_train_3$idx, ]

  temp_test_3 <- folds$test$`3`
  data_test_fold3 <- data[temp_test_3$idx, ]
}

# Mixed Effects Models
models <- list(
  "1 + (1 | Subject)",
  
  "1 + ((uLSC + uInt + fLSC + fInt) | Subject)",

  "uLSC + (1 | Subject)",
  "uInt + (1 | Subject)",
  "fLSC + (1 | Subject)",
  "fInt + (1 | Subject)",

  "uLSC + uInt + (1 | Subject)",
  "fLSC + fInt + (1 | Subject)",
  "uLSC + fLSC + (1 | Subject)",
  "uInt + fInt + (1 | Subject)",

  "uLSC + uInt + fLSC + fInt + (1 | Subject)",
  
  "uLSC * fLSC + (1 | Subject)",
  "uInt * fInt + (1 | Subject)",
  "uLSC * uInt + (1 | Subject)",
  "fLSC * fInt + (1 | Subject)",
  
  "uLSC:fLSC + uInt:fInt + (1 | Subject)",
  "fLSC + fInt + uLSC:fLSC + uInt:fInt + (1 | Subject)",    # best
  # "fLSC + fInt + uLSC:fLSC + uInt:fInt + ((fLSC + uLSC + fInt + uInt) | Subject)",
  "fLSC + fInt + uLSC:fLSC + uInt:fInt + ((fLSC + fInt) | Subject)",
  # "fLSC + fInt + uLSC:fLSC + uInt:fInt + ((uLSC + uInt) | Subject)",
  "fLSC + fInt + uLSC:fLSC + uInt:fInt + (fLSC | Subject)",
  # "fLSC + fInt + uLSC:fLSC + uInt:fInt + (uLSC | Subject)",
  "fLSC + fInt + uLSC:fLSC + uInt:fInt + (fInt | Subject)",
  # "fLSC + fInt + uLSC:fLSC + uInt:fInt + (uInt | Subject)",
  "uLSC * fLSC + uInt * fInt + (1 | Subject)",
  "uLSC * uInt + fLSC * fInt + (1 | Subject)",
  "uLSC + uInt + fLSC + fInt + uLSC:fInt + fLSC:uInt + (1 | Subject)",
  "uLSC + fLSC + uInt + fInt + uLSC:fLSC + uInt:fInt + uLSC:uInt + fLSC:fInt + uLSC:fInt + fLSC:uInt + (1 | Subject)",

  # Supplementary analysis

  # change in complexity
  "chLSC + chInt + (1 | Subject)",
  "avgchLSC + avgchInt + (1 | Subject)",
  "uLSC + uInt + fLSC + fInt + chLSC + chInt + (1 | Subject)",
  "uLSC + uInt + fLSC + fInt + avgchLSC + avgchInt + (1 | Subject)",

  # quadratic effects
  "uLSCsq + uIntsq + fLSCsq + fIntsq + (1 | Subject)",
  "uLSCsq + fLSCsq + uIntsq + fIntsq + uLSC:fLSC + uInt:fInt + (1 | Subject)",
  "uLSCsq + uIntsq + fLSCsq + fIntsq + uLSC:fLSC + uInt:fInt + (1 | Subject)", # L^2 + I^2
  "(uLSCsq + fLSCsq + uLSC:fLSC) + (uIntsq + fIntsq + uInt:fInt) + (uLSC + fLSC):(uInt + fInt) + (1 | Subject)", # (L + I)^2
  "uLSC + fLSC + uInt + fInt + uLSCsq + uIntsq + fLSCsq + fIntsq + (1 | Subject)", # L + I + L^2 + I^2 - interactions
  "uLSC + fLSC + uInt + fInt + uLSCsq + uIntsq + fLSCsq + fIntsq + uLSC:fLSC + uInt:fInt + (1 | Subject)", # best = L + I + L^2 + I^2
  "uLSC + fLSC + uInt + fInt + uLSCsq + uIntsq + fLSCsq + fIntsq + uLSC:fLSC + uInt:fInt + uLSC:uInt + fLSC:fInt + (1 | Subject)",
  "uLSC + fLSC + uInt + fInt + uLSCsq + uIntsq + fLSCsq + fIntsq + uLSC:fLSC + uInt:fInt + uLSC:uInt + fLSC:fInt + uLSC:fInt + uInt:fLSC + (1 | Subject)" # L + I + (L + I)^2
)

# Save all results to model_fits/Table_3_mixedeffects.csv
{
  df <- data.frame(matrix(ncol = 13, nrow = 0, dimnames =
  list(NULL, c("Id", "model", "AIC", "BIC", "AIC/BIC Var",
  "Rsq train mean", "Rsq train var", "Rsq test mean", "Rsq test var",
  "RMSE train mean", "RMSE train var", "RMSE test mean", "RMSE test var"))))

  id <- 0
  for (formula in models) {
    id <- id + 1
    fullformula <- paste("num_clicks ~", formula)
    f1 <- lmer(fullformula, data = data_train_fold1,
    control = lmerControl(optimizer = "bobyqa"))
    f2 <- lmer(fullformula, data = data_train_fold2,
    control = lmerControl(optimizer = "bobyqa"))
    f3 <- lmer(fullformula, data = data_train_fold3,
    control = lmerControl(optimizer = "bobyqa"))

    # Analyse model fit
    metrics <- modelanalysis(num_folds,
    list(f1, f2, f3), list(data_train_fold1, data_train_fold2,
    data_train_fold3), list(data_test_fold1, data_test_fold2, data_test_fold3),
    FALSE, FALSE, fullformula) # set second last param to TRUE for printing
    df[nrow(df) + 1, ] <- c(id, noquote(fullformula), metrics)
  }

  write.csv(df, "model_fits/Table_3_mixedeffects.csv", row.names = FALSE)
}

# Evaluate performance of best model and make plots
# for train, test and random effects
# Plots saved to ./plots/
{
  bestformula <- "num_clicks ~ uLSC + fLSC + uInt + fInt + uLSC:fLSC + uInt:fInt + (1 | Subject)"

  f <- lmer(bestformula, data = data,
  control = lmerControl(optimizer = "nloptwrap"))

  # make and save model vs predictions of train and test - Figure 3a,b
  make_plots(f, data_train_fold1, data_test_fold1)

  # Plot and save random effects
  ggCaterpillar(ranef(f, condVar = TRUE))
  ggsave("plots/random_effects_num_clicks.pdf")
}

# Save fixed effects to model_fits/ - Table 4
{
  model_summary <- summary(f)
  coefficients <- fixef(f)
  standard_errors <- sqrt(diag(vcov(f)))
  variable_names <- rownames(summary(f)$coefficients)
  p_values <- coef(summary(f))[, "Pr(>|t|)"]
  signif_levels <- ifelse(p_values < 0.001, "***", ifelse(p_values < 0.01, "**", ifelse(p_values < 0.05, "*", ifelse(p_values < 0.1, ".", "NS"))))
  results_df <- data.frame(Variable = variable_names, Coefficients = coefficients, StdError = standard_errors, PValue = p_values, Significance = signif_levels)
  write.csv(results_df, file = "model_fits/Table_4_coeff_mixedeffects.csv", row.names = FALSE)
}

# plot interactions - Figure 3c,d
{
  p <- interact_plot(f,
      pred = fLSC, modx = uLSC, interval = TRUE,
      x.label = "fLSC", y.label = "Number of Clicks",
      legend.main = "uLSC", colors = "seagreen"
  ) + theme(
      axis.title = element_text(family = "serif", size = 44),
      axis.text = element_text(family = "serif", size = 26),
      legend.text = element_text(family = "serif", size = 30),
      legend.title = element_text(family = "serif", size = 40),
      strip.text = element_text(family = "serif")
  ) + scale_y_continuous(breaks = seq(0, 80, by = 20), limits = c(0, 80)) + scale_x_continuous(breaks = seq(-6, 3, by = 2), limits = c(-6, 3))

  # Figure 3c
  ggsave(filename = "plots/uLSC_fLSC_interaction.pdf", plot = p, width = 10, height = 10, units = "in")

  p <- interact_plot(f,
      pred = fInt, modx = uInt, interval = TRUE,
      x.label = "fInt", y.label = "Number of Clicks",
      legend.main = "uInt", colors = "seagreen"
  ) + theme(
      axis.title = element_text(family = "serif", size = 44),
      axis.text = element_text(family = "serif", size = 26),
      legend.text = element_text(family = "serif", size = 30),
      legend.title = element_text(family = "serif", size = 44),
      strip.text = element_text(family = "serif")
  ) + scale_y_continuous(breaks = seq(0, 50, by = 20), limits = c(0, 50)) + scale_x_continuous(breaks = seq(-2, 8, by = 2), limits = c(-2, 8))

  # Figure 3d
  ggsave(filename = "plots/uInt_fInt_interaction.pdf", plot = p, width = 10, height = 10, units = "in")
}

# Find extrema patterns
{
  threshold_low_uLSC1 <- quantile(data$uLSC, probs = 0.1)
  threshold_high_uLSC1 <- quantile(data$uLSC, probs = 0.9)
  threshold_low_uLSC2 <- quantile(data$uLSC, probs = 0.33)
  threshold_high_uLSC2 <- quantile(data$uLSC, probs = 0.66)

  threshold_low_fLSC1 <- quantile(data$fLSC, probs = 0.1)
  threshold_high_fLSC1 <- quantile(data$fLSC, probs = 0.9)
  threshold_low_fLSC2 <- quantile(data$fLSC, probs = 0.33)
  threshold_high_fLSC2 <- quantile(data$fLSC, probs = 0.66)

  low_uLSC_low_fLSC <- subset(data, uLSC <= threshold_low_uLSC1 & fLSC <= threshold_low_fLSC1)
  low_uLSC_high_fLSC <- subset(data, uLSC <= threshold_low_uLSC2 & fLSC >= threshold_high_fLSC2)
  high_uLSC_low_fLSC <- subset(data, uLSC >= threshold_high_uLSC2 & fLSC <= threshold_low_fLSC2)
  high_uLSC_high_fLSC <- subset(data, uLSC >= threshold_high_uLSC1 & fLSC >= threshold_high_fLSC1)

  low_uLSC_low_fLSC_toplot <- low_uLSC_low_fLSC %>%
    sample_n(nrow(low_uLSC_low_fLSC)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  low_uLSC_high_fLSC_toplot <- low_uLSC_high_fLSC %>%
    sample_n(nrow(low_uLSC_high_fLSC)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  high_uLSC_low_fLSC_toplot <- high_uLSC_low_fLSC %>%
    sample_n(nrow(high_uLSC_low_fLSC)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  high_uLSC_high_fLSC_toplot <- high_uLSC_high_fLSC %>%
    sample_n(nrow(high_uLSC_high_fLSC)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  print(low_uLSC_low_fLSC_toplot)

  print(low_uLSC_high_fLSC_toplot)

  print(high_uLSC_low_fLSC_toplot)

  print(high_uLSC_high_fLSC_toplot)

}

{
  threshold_low_uInt1 <- quantile(data$uInt, probs = 0.1)
  threshold_high_uInt1 <- quantile(data$uInt, probs = 0.9)
  threshold_low_uInt2 <- quantile(data$uInt, probs = 0.33)
  threshold_high_uInt2 <- quantile(data$uInt, probs = 0.66)

  threshold_low_fInt1 <- quantile(data$fInt, probs = 0.1)
  threshold_high_fInt1 <- quantile(data$fInt, probs = 0.9)
  threshold_low_fInt2 <- quantile(data$fInt, probs = 0.33)
  threshold_high_fInt2 <- quantile(data$fInt, probs = 0.66)

  low_uInt_low_fInt <- subset(data, uInt <= threshold_low_uInt1 & fInt <= threshold_low_fInt1)
  low_uInt_high_fInt <- subset(data, uInt <= threshold_low_uInt2 & fInt >= threshold_high_fInt2)
  high_uInt_low_fInt <- subset(data, uInt >= threshold_high_uInt2 & fInt <= threshold_low_fInt2)
  high_uInt_high_fInt <- subset(data, uInt >= threshold_high_uInt1 & fInt >= threshold_high_fInt1)

  low_uInt_low_fInt_toplot <- low_uInt_low_fInt %>%
    sample_n(nrow(low_uInt_low_fInt)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  low_uInt_high_fInt_toplot <- low_uInt_high_fInt %>%
    sample_n(nrow(low_uInt_high_fInt)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  high_uInt_low_fInt_toplot <- high_uInt_low_fInt %>%
    sample_n(nrow(high_uInt_low_fInt)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  high_uInt_high_fInt_toplot <- high_uInt_high_fInt %>%
    sample_n(nrow(high_uInt_high_fInt)) %>%
    filter(num_clicks < 81) %>%
    select(Subject, grid_id, num_clicks) %>%
    head(5)

  print(low_uInt_low_fInt_toplot)

  print(low_uInt_high_fInt_toplot)

  print(high_uInt_low_fInt_toplot)

  print(high_uInt_high_fInt_toplot)
}
