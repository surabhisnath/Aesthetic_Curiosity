## Author: Surabhi S Nath
## Description: This script implements survival analysis (Cox regression) models.
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
    library("survival")
    library("survminer")
    library(jtools)
    library(interactions)
    library(dplyr)

    source("Utils.R")

    # Set seed 
    set.seed(20) # Seed set randomly for reproducibility
}

# Setup
{
    # Read data
    data = read.csv("../../csvs/grid-search/click_data_reevaluatedforreproduction.csv")
    # data = read.csv("utils/subject_click_info_coxreg_data23.csv")

    # Scale all variables in the data
    my_scale <- function(x) {
        as.numeric(scale(x))
    }

    data$pid <- factor(data$pid)
    data$click_id <- as.numeric(data$click_id)
    data$pattern_id <- factor(data$pattern_id)
    data$time_taken <- my_scale(data$time_taken)
    data$uLSC <- my_scale(data$underlying_SC)
    data$uLSCsq <- my_scale(data$underlying_SC ^ 2)
    data$uInt <- my_scale(data$underlying_intricacy)
    data$uIntsq <- my_scale(data$underlying_intricacy ^ 2)
    data$cLSC <- my_scale(data$current_SC)
    data$cLSCsq <- my_scale(data$current_SC ^ 2)
    data$cInt <- my_scale(data$current_intricacy)
    data$cIntsq <- my_scale(data$current_intricacy ^ 2)
    data$chLSC <- my_scale(data$current_change_in_SC)
    data$chInt <- my_scale(data$current_change_in_intricacy)
    data$avgchLSC <- my_scale(data$avg_change_in_SC)
    data$avgchInt <- my_scale(data$avg_change_in_intricacy)
}


{
    selected_data <- data[, c("uLSC", "cLSC", "uInt", "cInt", "uLSCsq", "cLSCsq", "uIntsq", "cIntsq", "chLSC", "chInt", "avgchLSC", "avgchInt")]
    correlation_matrix <- cor(selected_data)
    colnames(correlation_matrix) <- c("uLSC", "fLSC", "uInt", "fInt", "uLSCsq", "fLSCsq", "uIntsq", "fIntsq", "chLSC", "chInt", "avgchLSC", "avgchInt") # Rename columns
    rownames(correlation_matrix) <- c("uLSC", "fLSC", "uInt", "fInt", "uLSCsq", "fLSCsq", "uIntsq", "fIntsq", "chLSC", "chInt", "avgchLSC", "avgchInt") # Rename columns
    correlation_matrix[lower.tri(correlation_matrix)] <- NA
    pdf(file = "plots/click_level_correlations.pdf", width = 12, height = 12)
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

  data_train_folds <- list(data_train_fold1, data_train_fold2, data_train_fold3)
  data_test_folds <- list(data_test_fold1, data_test_fold2, data_test_fold3)
}

# Cox regression models
models <- list(
    "Subject",
    
    "uLSC + uInt + Subject",
    "cLSC + cInt + Subject",
    "uLSC + cLSC + Subject",
    "uInt + cInt + Subject",

    "uLSC + uInt + cLSC + cInt + Subject",
    
    "uLSC * cLSC + Subject",
    "uInt * cInt + Subject",
    "uLSC * uInt + Subject",
    "cLSC * cInt + Subject",
    
    "cLSC + cInt + uLSC:cLSC + uInt:cInt + Subject",
    "uLSC * cLSC + uInt * cInt + Subject",  # best
    "uLSC * uInt + cLSC * cInt + Subject",
    "uLSC + uInt + cLSC + cInt + uLSC:cInt + cLSC:uInt + Subject",
    "uLSC + cLSC + uInt + cInt + uLSC:cLSC + uInt:cInt + uLSC:uInt + cLSC:cInt + uLSC:cInt + cLSC:uInt + Subject",

    # Supplementary analysis
    
    # change in complexity
    "chLSC + chInt + Subject",
    "avgchLSC + avgchInt + Subject",
    "uLSC + uInt + cLSC + cInt + chLSC + chInt + Subject",
    "uLSC + uInt + cLSC + cInt + avgchLSC + avgchInt + Subject",

    # quadratic effects
    "uLSCsq + uIntsq + cLSCsq + cIntsq + Subject",
    "uLSCsq + cLSCsq + uIntsq + cIntsq + uLSC:cLSC + uInt:cInt + Subject",
    "uLSCsq + uIntsq + cLSCsq + cIntsq + uLSC:cLSC + uInt:cInt + Subject", # L^2 + I^2
    "(uLSCsq + cLSCsq + uLSC:cLSC) + (uIntsq + cIntsq + uInt:cInt) + (uLSC + cLSC):(uInt + cInt) + Subject", # (L + I)^2
    "uLSC + cLSC + uInt + cInt + uLSCsq + uIntsq + cLSCsq + cIntsq + Subject", # L + I + L^2 + I^2 - interactions
    "uLSC + cLSC + uInt + cInt + uLSCsq + uIntsq + cLSCsq + cIntsq + uLSC:cLSC + uInt:cInt + Subject", # best = L + I + L^2 + I^2
    "uLSC + cLSC + uInt + cInt + uLSCsq + uIntsq + cLSCsq + cIntsq + uLSC:cLSC + uInt:cInt + uLSC:uInt + cLSC:cInt + Subject",
    "uLSC + cLSC + uInt + cInt + uLSCsq + uIntsq + cLSCsq + cIntsq + uLSC:cLSC + uInt:cInt + uLSC:uInt + cLSC:cInt + uLSC:cInt + uInt:cLSC + Subject" # L + I + (L + I)^2
)

# Save all results to model_fits/Table_1_survival.csv
{
    df <- data.frame(matrix(ncol = 8, nrow = 0, dimnames =
    list(NULL, c("Id", "model", "AIC", "BIC",
        "Concordance train", "Concordance test", "log rank p", "likelihood ratio p"))))

    id <- 0
    for(formula in models) 
    {
        id <- id + 1
        fullformula <- paste("Surv(click_id_minus1, click_id, move_on) ~ ", formula)
        formula <- as.formula(fullformula)
        
        # calcultate the following metrics
        AICs <- numeric(num_folds)
        BICs <- numeric(num_folds)
        concordances_train <-numeric(num_folds)
        concordances_test <-numeric(num_folds)
        logrankp <- numeric(num_folds)
        likratiop <- numeric(num_folds)

        for(x in 1:num_folds)
        {
            model <- coxph(formula, data = data_train_folds[[x]])
            
            # calculate metrics and store in arrays
            modsum = summary(model)
            logrankp[[x]] <- modsum$logtest["pvalue"]
            likratiop[[x]] <- modsum$sctest["pvalue"]

            pred_train <- predict(model, newdata = data_train_folds[[x]], type = 'risk')
            concordances_train[[x]] <- survConcordance(Surv(click_id, move_on) ~ pred_train, data = data_train_folds[[x]])$concordance
            pred_test <- predict(model, newdata = data_test_folds[[x]], type = 'risk')
            concordances_test[[x]] <- survConcordance(Surv(click_id, move_on) ~ pred_test, data = data_test_folds[[x]])$concordance

            AICs[x] <- AIC(model)
            BICs[x] <- BIC(model)
        }

        df[nrow(df) + 1, ] <- c(id, noquote(fullformula), c(mean(AICs), mean(BICs), mean(concordances_train), mean(concordances_test), max(logrankp), max(likratiop)))
    }

    write.csv(df, "model_fits/Table_1_survival.csv", row.names = FALSE)
}


# Save fixed effects to model_fits/ - Table 2
{
    # bestformula <- as.formula("Surv(click_id, move_on) ~ uLSC * cLSC + uInt * cInt + Subject") # Subject excluded for reliability of estimates 

    bestformula <- as.formula("Surv(click_id, move_on) ~ cLSC + cInt + uLSC:cLSC + uInt:cInt + Subject")

    model <- coxph(bestformula, data = data)

    model_matrix <- model.matrix(model)
    model_df <- data.frame(model_matrix)
    lm_model <- lm(rep(1, nrow(model_df)) ~ ., data = model_df)
    vif_values <- vif(lm_model)
    # print(vif_values)         # uncomment if want to print VIFs in the best model

    coefficients <- coef(model)
    exp_coefficients <- exp(coefficients)
    coef_names <- names(coefficients)

    p_values <- summary(model)$coefficients[, "Pr(>|z|)"]
    signif_levels <- ifelse(p_values < 0.001, "***", ifelse(p_values < 0.01, "**", ifelse(p_values < 0.05, "*", ifelse(p_values < 0.1, ".", "NS"))))

    # Combine them into a data frame
    combined_data <- data.frame(Variable = coef_names, Coefficients = coefficients, ExpCoefficients = exp_coefficients, Significance = signif_levels)

    # Write the data frame to a CSV file
    write.csv(combined_data, file = "model_fits/Table_2_coeff_survival.csv", row.names = FALSE)
}


# Plot survival curves keeping all but one or two (for interactions) variables constant - Figure 2
{
    # cLSC manipulation - Figure 2a
    newdata <- data.frame(
        cLSC = seq(mean(data$cLSC, na.rm = TRUE) - 2*sd(data$cLSC, na.rm = TRUE), mean(data$cLSC, na.rm = TRUE) + 2*sd(data$cLSC, na.rm = TRUE), length.out = 3),
        # uLSC = mean(data$uLSC, na.rm = TRUE),
        # uInt = mean(data$uInt, na.rm = TRUE),
        cInt = mean(data$cInt, na.rm = TRUE),
        Subject = factor(21)
        )
    var <- "cLSC"
    filename <- paste0(var, "_survival_curve")
    survival_prob(model, newdata, filename, var, c("- 2SD", "mean", "+ 2SD"), c("salmon", "red", "darkred"))

    # cInt manipulation - not reported in paper
    newdata <- data.frame(
        cInt = seq(mean(data$cInt, na.rm = TRUE) - 2*sd(data$cInt, na.rm = TRUE), mean(data$cInt, na.rm = TRUE) + 2*sd(data$cInt, na.rm = TRUE), length.out = 3),
        # uLSC = mean(data$uLSC, na.rm = TRUE),
        cLSC = mean(data$cLSC, na.rm = TRUE),
        # uInt = mean(data$uInt, na.rm = TRUE),
        Subject = factor(21)
        )
    var <- "cInt"
    filename <- paste0(var, "_survival_curve")
    # survival_prob(model, newdata, filename, var, c("- 2SD", "mean", "+ 2SD"), c("gold", "darkorange", "chocolate"))
    survival_prob(model, newdata, filename, var, c("- 2SD", "mean", "+ 2SD"), c("tan", "sienna", "brown"))


    # uLSC x cLSC manipulation - Figure 2c
    v1 <- seq(mean(data$uLSC, na.rm = TRUE) - 2*sd(data$uLSC, na.rm = TRUE), mean(data$uLSC, na.rm = TRUE) + 2*sd(data$uLSC, na.rm = TRUE), length.out = 2)  # Values of uLSC for the three scenarios
    v2 <- seq(mean(data$cLSC, na.rm = TRUE) - 2*sd(data$cLSC, na.rm = TRUE), mean(data$cLSC, na.rm = TRUE) + 2*sd(data$cLSC, na.rm = TRUE), length.out = 2)   # Values of cLSC for the three scenarios
    newdata <- expand.grid(
        uLSC = v1,
        cLSC = v2
    )
    newdata$uInt <- mean(data$uInt, na.rm = TRUE)
    newdata$cInt <- mean(data$cInt, na.rm = TRUE)
    newdata$Subject <- factor(21)
    var <- "uLSC:cLSC"
    filename <- paste0(var, "_survival_curve")
    survival_prob(model, newdata, filename, var, c("1", "2", "3", "4"), c("lightgreen", "darkseagreen", "darkcyan", "slategray"))


    # Participant manipulation - Figure 2d
    unique_subjects <- unique(data$Subject)
    indices <- c(1, 12, 21)
    sampled_subjects <- unique_subjects[indices]
    newdata_list <- lapply(sampled_subjects, function(subj) {
    data.frame(
        # uLSC = mean(data$uLSC, na.rm = TRUE),
        cLSC = mean(data$cLSC, na.rm = TRUE),
        # uInt = mean(data$uInt, na.rm = TRUE),
        cInt = mean(data$cInt, na.rm = TRUE),
        Subject = factor(subj, levels = levels(unique_subjects[indices]))
    )
    })
    newdata <- bind_rows(newdata_list)
    var <- "Participant"
    filename <- paste0(var, "_survival_curve")
    survival_prob(model, newdata, filename, var, c("1", "12", "21"), c("lightblue", "dodgerblue", "midnightblue"))
}
