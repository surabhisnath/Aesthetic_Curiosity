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
  data <- read.csv("../../csvs/grid-search/grid_data_reevaluatedforreproduction.csv")

  # Scale all variables in the data
  my_scale <- function(x) {
    as.numeric(scale(x))
  }

  data$pid <- factor(data$pid)
  data$grid_id <- factor(data$grid_id)
  data$pattern <- factor(data$pattern_id)

  data$uLSCsq <- my_scale(data$uLSC ^ 2)
  data$uLSC <- my_scale(data$uLSC)
  data$uLSCr <- my_scale(data$uLSCr)
  data$uLSCpsq <- my_scale(data$uLSCp ^ 2)
  data$uLSCp <- my_scale(data$uLSCp)
  data$uLSCdiffr <- my_scale(data$uLSCdiffr)
  data$uLSCdiffp <- my_scale(data$uLSCdiffp)
  
  data$uIntsq <- my_scale(data$uInt ^ 2)
  data$uInt <- my_scale(data$uInt)
  data$uIntr <- my_scale(data$uIntr)
  data$uIntpsq <- my_scale(data$uIntp ^ 2)
  data$uIntp <- my_scale(data$uIntp)
  data$uIntdiffr <- my_scale(data$uIntdiffr)
  data$uIntdiffp <- my_scale(data$uIntdiffp)
  
  data$fLSCr <- my_scale(data$fLSCr)
  data$fLSCpold <- my_scale(data$fLSCpold)
  data$fLSCpsq <- my_scale(data$fLSCp ^ 2)
  data$fLSCp <- my_scale(data$fLSCp)
  
  data$fIntr <- my_scale(data$fIntr)
  data$fIntpold <- my_scale(data$fIntpold)
  data$fIntpsq <- my_scale(data$fIntp ^ 2)
  data$fIntp <- my_scale(data$fIntp)
}

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
  
  # "1 + ((uLSC + uInt + fLSCp + fIntp) | pid)",

  # "uLSC + (1 | pid)",
  # "uLSCr + (1 | pid)",
  "uLSCp + (1 | pid)",
  # "uLSCdiffr + (1 | pid)",
  # "uLSCdiffp + (1 | pid)",

  # "uInt + (1 | pid)",
  # "uIntr + (1 | pid)",
  "uIntp + (1 | pid)",
  # "uIntdiffr + (1 | pid)",
  # "uIntdiffp + (1 | pid)",

  # "fLSCpold + (1 | pid)",
  "fLSCp + (1 | pid)",
  # "fLSCr + (1 | pid)",

  # "fIntpold + (1 | pid)",
  "fIntp + (1 | pid)",
  # "fIntr + (1 | pid)",

  "uLSCp + uIntp + (1 | pid)",
  "fLSCp + fIntp + (1 | pid)",

  "uLSCp + uIntp + fLSCp + fIntp + (1 | pid)",
  "fLSCp + fLSCpsq + fIntp + fIntpsq + (1 | pid)",
  
  # "uLSC * fLSCp + (1 | pid)",
  # "uInt * fIntp + (1 | pid)",
  # "uLSC * uInt + (1 | pid)",
  # "fLSCp * fIntp + (1 | pid)",
  
  "fLSCpold + fIntpold + uLSC:fLSCpold + uInt:fIntpold + (1 | pid)",
  "fLSCpold + fIntpold + uLSCp:fLSCpold + uIntp:fIntpold + (1 | pid)",
  "fLSCpold + fIntpold + uLSCr:fLSCpold + uIntr:fIntpold + (1 | pid)",
  "fLSCpold + fIntpold + uLSCdiffp:fLSCpold + uIntdiffp:fIntpold + (1 | pid)",
  "fLSCpold + fIntpold + uLSCdiffr:fLSCpold + uIntdiffr:fIntpold + (1 | pid)",

  "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + (1 | pid)",
  "fLSCp + fIntp + uLSCp + uIntp + uLSCp:fLSCp + uIntp:fIntp + (1 | pid)",
  "fLSCp + fIntp + uLSCp:fLSCp + uIntp:fIntp + (1 | pid)",
  "fLSCp + fIntp + uLSCr:fLSCp + uIntr:fIntp + (1 | pid)",
  "fLSCp + fIntp + uLSCdiffp:fLSCp + uIntdiffp:fIntp + (1 | pid)",
  "fLSCp + fIntp + uLSCdiffr:fLSCp + uIntdiffr:fIntp + (1 | pid)",

  "fLSCr + fIntr + uLSC:fLSCr + uInt:fIntr + (1 | pid)",
  "fLSCr + fIntr + uLSCp:fLSCr + uIntp:fIntr + (1 | pid)",
  "fLSCr + fIntr + uLSCr:fLSCr + uIntr:fIntr + (1 | pid)",
  "fLSCr + fIntr + uLSCdiffp:fLSCr + uIntdiffp:fIntr + (1 | pid)",
  "fLSCr + fIntr + uLSCdiffr:fLSCr + uIntdiffr:fIntr + (1 | pid)"

  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + ((fLSCp + uLSC + fIntp + uInt) | pid)",
  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + ((fLSCp + fIntp) | pid)",
  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + ((uLSC + uInt) | pid)",
  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + (fLSCp | pid)",
  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + (uLSC | pid)",
  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + (fIntp | pid)",
  # "fLSCp + fIntp + uLSC:fLSCp + uInt:fIntp + (uInt | pid)",
  # "uLSC * fLSCp + uInt * fIntp + (1 | pid)",
  # "uLSC * uInt + fLSCp * fIntp + (1 | pid)",
  # "uLSC + uInt + fLSCp + fIntp + uLSC:fIntp + fLSCp:uInt + (1 | pid)",
  # "uLSC + fLSCp + uInt + fIntp + uLSC:fLSCp + uInt:fIntp + uLSC:uInt + fLSCp:fIntp + uLSC:fIntp + fLSCp:uInt + (1 | pid)",

  # Supplementary analysis

  # change in complexity
  # "chLSC + chInt + (1 | pid)",
  # "avgchLSC + avgchInt + (1 | pid)",
  # "uLSC + uInt + fLSCp + fIntp + chLSC + chInt + (1 | pid)",
  # "uLSC + uInt + fLSCp + fIntp + avgchLSC + avgchInt + (1 | pid)",

  # quadratic effects
)

# Save all results to model_fits/Table_3_mixedeffects.csv
{
  df <- data.frame(matrix(ncol = 14, nrow = 0, dimnames =
  list(NULL, c("Id", "model", "AIC", "BIC", "AIC/BIC Var",
  "VIF", "Rsq train mean", "Rsq train var", "Rsq test mean", "Rsq test var",
  "RMSE train mean", "RMSE train var", "RMSE test mean", "RMSE test var"))))

  id <- 0
  for (formula in models) 
  {
    id <- id + 1
    fullformula <- paste("num_clicks ~", formula)
    f1 <- lmer(fullformula, data = data_train_fold1,
    control = lmerControl(optimizer = "bobyqa"))
    f1_lm <- lm(sub("\\+[^+]*$", "", fullformula), data = data_train_fold1)
    
    f2 <- lmer(fullformula, data = data_train_fold2,
    control = lmerControl(optimizer = "bobyqa"))
    f2_lm <- lm(sub("\\+[^+]*$", "", fullformula), data = data_train_fold2)

    f3 <- lmer(fullformula, data = data_train_fold3,
    control = lmerControl(optimizer = "bobyqa"))
    f3_lm <- lm(sub("\\+[^+]*$", "", fullformula), data = data_train_fold3)

    print(fullformula)
    # Analyse model fit
    metrics <- modelanalysis(num_folds,
    list(f1, f2, f3), list(f1_lm, f2_lm, f3_lm), list(data_train_fold1, data_train_fold2,
    data_train_fold3), list(data_test_fold1, data_test_fold2, data_test_fold3),
    FALSE, FALSE, fullformula, length(gregexpr("\\+", fullformula)[[1]]), grepl(":", fullformula) ) # set second last param to TRUE for printing
    df[nrow(df) + 1, ] <- c(id, noquote(fullformula), metrics)
  }

  write.csv(df, "model_fits/Table_3_mixedeffects.csv", row.names = FALSE)
}

# Evaluate performance of best model and make plots
# for train, test and random effects
# Plots saved to ./plots/
{
  
  bestformula <- "num_clicks ~ fLSCp + fLSCpsq + fIntp + fIntpsq + (1 | pid)"

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
      pred = fLSCp, modx = uLSCp, modx.values = c(-1, 0, 1), modx.labels = c("-1 SD", "Mean", "+1 SD"),
      interval = TRUE,
      x.label = "fLSCp", y.label = "Number of Clicks",
      legend.main = "uLSCp", colors = "seagreen",
      xlim = c(min(data$fLSCp), max(data$fLSCp)),
      ylim = c(min(data$num_clicks), max(data$num_clicks))
  ) + theme(
      axis.title = element_text(family = "serif", size = 44),
      axis.text = element_text(family = "serif", size = 26),
      legend.text = element_text(family = "serif", size = 30),
      legend.title = element_text(family = "serif", size = 40),
      strip.text = element_text(family = "serif")
  ) + scale_y_continuous(breaks = seq(0, 80, by = 20), limits = c(0, 80)) + scale_x_continuous(breaks = seq(-6, 3, by = 2), limits = c(-6, 3))

  # Figure 3c
  ggsave(filename = "plots/uLSCp_fLSCp_interaction.pdf", plot = p, width = 10, height = 10, units = "in")

  p <- interact_plot(f,
      pred = fIntp, modx = uIntp, interval = TRUE,
      x.label = "fIntp", y.label = "Number of Clicks",
      legend.main = "uIntp", colors = "seagreen"
  ) + theme(
      axis.title = element_text(family = "serif", size = 44),
      axis.text = element_text(family = "serif", size = 26),
      legend.text = element_text(family = "serif", size = 30),
      legend.title = element_text(family = "serif", size = 44),
      strip.text = element_text(family = "serif")
  ) + scale_y_continuous(breaks = seq(0, 50, by = 20), limits = c(0, 50)) + scale_x_continuous(breaks = seq(-2, 8, by = 2), limits = c(-2, 8))

  # Figure 3d
  ggsave(filename = "plots/uIntp_fIntp_interaction.pdf", plot = p, width = 10, height = 10, units = "in")
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