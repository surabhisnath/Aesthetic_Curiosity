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
  data <- read.csv("../../csvs/grid-search/grid_data_reevaluatedforreproduction_withoutoutliers.csv")

  # Scale all variables in the data
  my_scale <- function(x) {
    as.numeric(scale(x))
  }

  data$pid <- factor(data$pid)
  data$grid_id <- factor(data$grid_id)
  data$pattern <- factor(data$pattern_id)

  data$uLSC <- my_scale(data$uLSC)
  data$uLSCsq <- my_scale(data$uLSC ^ 2)
  data$uLSCdiff <- my_scale(data$uLSC_diff)
  data$uLSCperm <- my_scale(data$uLSC_permutation)
  data$uLSCrandom <- my_scale(data$uLSC_random)
  
  data$uInt <- my_scale(data$uInt)
  data$uIntsq <- my_scale(data$uInt ^ 2)
  data$uIntdiff <- my_scale(data$uInt_diff)
  data$uIntperm <- my_scale(data$uInt_permutation)
  data$uIntrandom <- my_scale(data$uInt_random)
  
  data$fLSCrandom <- my_scale(data$fLSC_random)
  data$fLSCpermutation <- my_scale(data$fLSC_permutation)
  data$fLSCpermutationfull <- my_scale(data$fLSC_permutation_full)
  data$fLSCpermutationsq <- my_scale(data$fLSC_permutation ^ 2)
  data$fIntrandom <- my_scale(data$fInt_random)
  data$fIntpermutation <- my_scale(data$fInt_permutation)
  data$fIntpermutationfull <- my_scale(data$fInt_permutation_full)
  data$fIntpermutationsq <- my_scale(data$fInt_permutation ^ 2)
}

# Make correlation plot - not used in paper
# {
#     selected_data <- data[, c("num_clicks", "uLSC", "uLSCrandom", "fLSCpermutation", "uInt", "uIntrandom", "fIntpermutation", "uLSCsq", "fLSCpermutationsq", "uIntsq", "fIntpermutationsq")]
#     correlation_matrix <- cor(selected_data)
#     colnames(correlation_matrix) <- c("num_clicks", "uLSC", "uLSCrandom", "fLSCpermutation", "uInt", "uIntrandom", "fIntpermutation", "uLSCsq", "fLSCpermutationsq", "uIntsq", "fIntpermutationsq") # Rename columns
#     rownames(correlation_matrix) <- c("num_clicks", "uLSC", "uLSCrandom", "fLSCpermutation", "uInt", "uIntrandom", "fIntpermutation", "uLSCsq", "fLSCpermutationsq", "uIntsq", "fIntpermutationsq") # Rename columns
#     correlation_matrix[lower.tri(correlation_matrix)] <- NA
#     pdf(file = "plots/grid_level_correlations.pdf", width = 12, height = 12)
#     par(mar=c(7,7,2,2), cex = 1.3, family="serif")

#     # Plot the heatmap
#     image(1:ncol(correlation_matrix), 1:nrow(correlation_matrix), t(correlation_matrix), 
#         col = colorRampPalette(c("blue", "white", "red"))(20), axes = FALSE, xlab = "", ylab = "")

#     # Add text annotations for the correlation values
#     for (i in 1:nrow(correlation_matrix)) {
#         for (j in 1:ncol(correlation_matrix)) {
#             if (!is.na(correlation_matrix[i, j])) {
#                 text(j, i, round(correlation_matrix[i, j], 2), cex = 1.3)
#             }
#         }
#     }

#     # Customize the row and column names
#     axis(1, at = 1:ncol(correlation_matrix), labels = colnames(correlation_matrix), las = 2, cex.axis = 1.4, family = "serif")
#     axis(2, at = 1:nrow(correlation_matrix), labels = rownames(correlation_matrix), las = 2, cex.axis = 1.4, family = "serif")

#     # save correlation plot
#     dev.off()
# }

# Split the data into 3 stratified folds
{
  num_folds <- 3

  folds <- data %>%
  group_by(pid) %>%
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
  "1 + (1 | pid)",
  
  "1 + ((uLSC + uInt + fLSCpermutationfull + fIntpermutationfull) | pid)",

  "uLSC + (1 | pid)",
  "uLSCdiff + (1 | pid)",

  "uInt + (1 | pid)",
  "uIntdiff + (1 | pid)",

  "fLSCpermutationfull + (1 | pid)",
  "fIntpermutationfull + (1 | pid)",

  "uLSC + uInt + (1 | pid)",
  "fLSCpermutationfull + fIntpermutationfull + (1 | pid)",
  "uLSC + fLSCpermutationfull + (1 | pid)",
  "uInt + fIntpermutationfull + (1 | pid)",

  "uLSC + uInt + fLSCpermutationfull + fIntpermutationfull + (1 | pid)",
  
  "uLSC * fLSCpermutationfull + (1 | pid)",
  "uInt * fIntpermutationfull + (1 | pid)",
  "uLSC * uInt + (1 | pid)",
  "fLSCpermutationfull * fIntpermutationfull + (1 | pid)",
  
  "uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + (1 | pid)",
  "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + (1 | pid)",    # best
  "fLSCpermutationfull + fIntpermutationfull + uLSCdiff:fLSCpermutationfull + uIntdiff:fIntpermutationfull + (1 | pid)",    # best
  "fLSCpermutationfull + fIntpermutationfull + uLSCperm:fLSCpermutationfull + uIntperm:fIntpermutationfull + (1 | pid)",    # best
  "fLSCpermutationfull + fIntpermutationfull + uLSCrandom:fLSCpermutationfull + uIntrandom:fIntpermutationfull + (1 | pid)",    # best

  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + ((fLSCpermutationfull + uLSC + fIntpermutationfull + uInt) | pid)",
  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + ((fLSCpermutationfull + fIntpermutationfull) | pid)",
  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + ((uLSC + uInt) | pid)",
  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + (fLSCpermutationfull | pid)",
  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + (uLSC | pid)",
  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + (fIntpermutationfull | pid)",
  # "fLSCpermutationfull + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + (uInt | pid)",
  "uLSC * fLSCpermutationfull + uInt * fIntpermutationfull + (1 | pid)",
  "uLSC * uInt + fLSCpermutationfull * fIntpermutationfull + (1 | pid)",
  "uLSC + uInt + fLSCpermutationfull + fIntpermutationfull + uLSC:fIntpermutationfull + fLSCpermutationfull:uInt + (1 | pid)",
  "uLSC + fLSCpermutationfull + uInt + fIntpermutationfull + uLSC:fLSCpermutationfull + uInt:fIntpermutationfull + uLSC:uInt + fLSCpermutationfull:fIntpermutationfull + uLSC:fIntpermutationfull + fLSCpermutation:uInt + (1 | pid)",

  # Supplementary analysis

  # change in complexity
  # "chLSC + chInt + (1 | pid)",
  # "avgchLSC + avgchInt + (1 | pid)",
  # "uLSC + uInt + fLSCpermutationfull + fIntpermutationfull + chLSC + chInt + (1 | pid)",
  # "uLSC + uInt + fLSCpermutationfull + fIntpermutationfull + avgchLSC + avgchInt + (1 | pid)",

  # quadratic effects
  "uLSCsq + uIntsq + fLSCpermutationsq + fIntpermutationsq + (1 | pid)",
  "uLSCsq + fLSCpermutationsq + uIntsq + fIntpermutationsq + uLSC:fLSCpermutation + uInt:fIntpermutation + (1 | pid)",
  "uLSCsq + uIntsq + fLSCpermutationsq + fIntpermutationsq + uLSC:fLSCpermutation + uInt:fIntpermutation + (1 | pid)", # L^2 + I^2
  "(uLSCsq + fLSCpermutationsq + uLSC:fLSCpermutation) + (uIntsq + fIntpermutationsq + uInt:fIntpermutation) + (uLSC + fLSCpermutation):(uInt + fIntpermutation) + (1 | pid)", # (L + I)^2
  "uLSC + fLSCpermutation + uInt + fIntpermutation + uLSCsq + uIntsq + fLSCpermutationsq + fIntpermutationsq + (1 | pid)", # L + I + L^2 + I^2 - interactions
  "uLSC + fLSCpermutation + uInt + fIntpermutation + uLSCsq + uIntsq + fLSCpermutationsq + fIntpermutationsq + uLSC:fLSCpermutation + uInt:fIntpermutation + (1 | pid)", # best = L + I + L^2 + I^2
  "uLSC + fLSCpermutation + uInt + fIntpermutation + uLSCsq + uIntsq + fLSCpermutationsq + fIntpermutationsq + uLSC:fLSCpermutation + uInt:fIntpermutation + uLSC:uInt + fLSCpermutation:fIntpermutation + (1 | pid)",
  "uLSC + fLSCpermutation + uInt + fIntpermutation + uLSCsq + uIntsq + fLSCpermutationsq + fIntpermutationsq + uLSC:fLSCpermutation + uInt:fIntpermutation + uLSC:uInt + fLSCpermutation:fIntpermutation + uLSC:fIntpermutation + uInt:fLSCpermutation + (1 | pid)" # L + I + (L + I)^2
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
  
  bestformula <- "num_clicks ~ fLSCpermutation + fIntpermutation + uLSC:fLSCpermutation + uInt:fIntpermutation + (1 | pid)"
  # bestformula <- "num_clicks ~ fLSCpermutation + fIntpermutation + uLSC:fLSCpermutation + uInt:fIntpermutation + (1 | pid)"


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
  # mean_value <- mean(data$uLSC, na.rm = TRUE)
  # sd_value <- sd(data$uLSC, na.rm = TRUE)
  # modx_values <- c(mean_value - 2 * sd_value, mean_value, mean_value + 2 * sd_value)
  p <- interact_plot(f,
      pred = fLSCpermutation, modx = uLSC, modx.values = c(-1, 0, 1), modx.labels = c("-1 SD", "Mean", "+1 SD"),
      interval = TRUE,
      x.label = "fLSCpermutation", y.label = "Number of Clicks",
      legend.main = "uLSC", colors = "seagreen",
      xlim = c(min(data$fLSCpermutation), max(data$fLSCpermutation)),
      ylim = c(min(data$num_clicks), max(data$num_clicks))
  ) + theme(
      axis.title = element_text(family = "serif", size = 44),
      axis.text = element_text(family = "serif", size = 26),
      legend.text = element_text(family = "serif", size = 30),
      legend.title = element_text(family = "serif", size = 40),
      strip.text = element_text(family = "serif")
  ) + scale_y_continuous(breaks = seq(0, 80, by = 20), limits = c(0, 80)) + scale_x_continuous(breaks = seq(-6, 3, by = 2), limits = c(-6, 3))

  # Figure 3c
  ggsave(filename = "plots/uLSC_fLSCpermutation_interaction.pdf", plot = p, width = 10, height = 10, units = "in")

  p <- interact_plot(f,
      pred = fIntpermutation, modx = uInt, interval = TRUE,
      x.label = "fIntpermutation", y.label = "Number of Clicks",
      legend.main = "uInt", colors = "seagreen"
  ) + theme(
      axis.title = element_text(family = "serif", size = 44),
      axis.text = element_text(family = "serif", size = 26),
      legend.text = element_text(family = "serif", size = 30),
      legend.title = element_text(family = "serif", size = 44),
      strip.text = element_text(family = "serif")
  ) + scale_y_continuous(breaks = seq(0, 50, by = 20), limits = c(0, 50)) + scale_x_continuous(breaks = seq(-2, 8, by = 2), limits = c(-2, 8))

  # Figure 3d
  ggsave(filename = "plots/uInt_fIntpermutation_interaction.pdf", plot = p, width = 10, height = 10, units = "in")
}

# Find extrema patterns
{
  threshold_low_uLSC1 <- quantile(data$uLSC, probs = 0.1)
  threshold_high_uLSC1 <- quantile(data$uLSC, probs = 0.9)
  threshold_low_uLSC2 <- quantile(data$uLSC, probs = 0.33)
  threshold_high_uLSC2 <- quantile(data$uLSC, probs = 0.66)

  threshold_low_fLSCpermutation1 <- quantile(data$fLSCpermutation, probs = 0.1)
  threshold_high_fLSCpermutation1 <- quantile(data$fLSCpermutation, probs = 0.9)
  threshold_low_fLSCpermutation2 <- quantile(data$fLSCpermutation, probs = 0.33)
  threshold_high_fLSCpermutation2 <- quantile(data$fLSCpermutation, probs = 0.66)

  low_uLSC_low_fLSCpermutation <- subset(data, uLSC <= threshold_low_uLSC1 & fLSCpermutation <= threshold_low_fLSCpermutation1)
  low_uLSC_high_fLSCpermutation <- subset(data, uLSC <= threshold_low_uLSC2 & fLSCpermutation >= threshold_high_fLSCpermutation2)
  high_uLSC_low_fLSCpermutation <- subset(data, uLSC >= threshold_high_uLSC2 & fLSCpermutation <= threshold_low_fLSCpermutation2)
  high_uLSC_high_fLSCpermutation <- subset(data, uLSC >= threshold_high_uLSC1 & fLSCpermutation >= threshold_high_fLSCpermutation1)

  low_uLSC_low_fLSCpermutation_toplot <- low_uLSC_low_fLSCpermutation %>%
    sample_n(nrow(low_uLSC_low_fLSCpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  low_uLSC_high_fLSCpermutation_toplot <- low_uLSC_high_fLSCpermutation %>%
    sample_n(nrow(low_uLSC_high_fLSCpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  high_uLSC_low_fLSCpermutation_toplot <- high_uLSC_low_fLSCpermutation %>%
    sample_n(nrow(high_uLSC_low_fLSCpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  high_uLSC_high_fLSCpermutation_toplot <- high_uLSC_high_fLSCpermutation %>%
    sample_n(nrow(high_uLSC_high_fLSCpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  print(low_uLSC_low_fLSCpermutation_toplot)

  print(low_uLSC_high_fLSCpermutation_toplot)

  print(high_uLSC_low_fLSCpermutation_toplot)

  print(high_uLSC_high_fLSCpermutation_toplot)

}

{
  threshold_low_uInt1 <- quantile(data$uInt, probs = 0.1)
  threshold_high_uInt1 <- quantile(data$uInt, probs = 0.9)
  threshold_low_uInt2 <- quantile(data$uInt, probs = 0.33)
  threshold_high_uInt2 <- quantile(data$uInt, probs = 0.66)

  threshold_low_fIntpermutation1 <- quantile(data$fIntpermutation, probs = 0.1)
  threshold_high_fIntpermutation1 <- quantile(data$fIntpermutation, probs = 0.9)
  threshold_low_fIntpermutation2 <- quantile(data$fIntpermutation, probs = 0.33)
  threshold_high_fIntpermutation2 <- quantile(data$fIntpermutation, probs = 0.66)

  low_uInt_low_fIntpermutation <- subset(data, uInt <= threshold_low_uInt1 & fIntpermutation <= threshold_low_fIntpermutation1)
  low_uInt_high_fIntpermutation <- subset(data, uInt <= threshold_low_uInt2 & fIntpermutation >= threshold_high_fIntpermutation2)
  high_uInt_low_fIntpermutation <- subset(data, uInt >= threshold_high_uInt2 & fIntpermutation <= threshold_low_fIntpermutation2)
  high_uInt_high_fIntpermutation <- subset(data, uInt >= threshold_high_uInt1 & fIntpermutation >= threshold_high_fIntpermutation1)

  low_uInt_low_fIntpermutation_toplot <- low_uInt_low_fIntpermutation %>%
    sample_n(nrow(low_uInt_low_fIntpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  low_uInt_high_fIntpermutation_toplot <- low_uInt_high_fIntpermutation %>%
    sample_n(nrow(low_uInt_high_fIntpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  high_uInt_low_fIntpermutation_toplot <- high_uInt_low_fIntpermutation %>%
    sample_n(nrow(high_uInt_low_fIntpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  high_uInt_high_fIntpermutation_toplot <- high_uInt_high_fIntpermutation %>%
    sample_n(nrow(high_uInt_high_fIntpermutation)) %>%
    filter(num_clicks < 81) %>%
    select(pid, grid_id, num_clicks) %>%
    head(5)

  print(low_uInt_low_fIntpermutation_toplot)

  print(low_uInt_high_fIntpermutation_toplot)

  print(high_uInt_low_fIntpermutation_toplot)

  print(high_uInt_high_fIntpermutation_toplot)
}