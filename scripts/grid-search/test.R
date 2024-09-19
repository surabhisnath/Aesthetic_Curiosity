
library(survival)
library(survminer)
library(ggplot2)
library(dplyr)

lung <- read.csv("lung.csv")

# Assuming 'lung_clean' is your cleaned dataset
lung_clean <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])

# Fit Kaplan-Meier model on cleaned data
km_fit <- survfit(Surv(time, status) ~ 1, data = lung_clean)

# Fit Cox proportional hazards model on cleaned data
cox_fit <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung_clean)

# Create Kaplan-Meier plot
km_plot <- ggsurvplot(km_fit, data = lung_clean, conf.int = FALSE, legend = "none",
                      ggtheme = theme_minimal(), risk.table = FALSE, color = "#2E9FDF")

# Create Cox model predicted survival curve plot
cox_plot <- ggsurvplot(survfit(cox_fit, data = lung_clean), data = lung_clean, conf.int = FALSE, 
                       legend.title = "Model", legend.labs = "Cox Model", 
                       ggtheme = theme_minimal(), risk.table = FALSE, color = "#e74900")

# Combine the two plots using ggplot2
combined_plot <- km_plot$plot +
  geom_line(data = cox_plot$plot$data, aes(x = time, y = surv, color = "Cox Model")) +
  scale_color_manual(values = c("Kaplan-Meier" = "#2E9FDF", "Cox Model" = "#e74900")) +
  labs(color = "Model") +
  theme(legend.position = "top")

# Display the combined plot
print(combined_plot)









# Assuming 'lung_clean' is your cleaned dataset
# lung_clean <- na.omit(lung[, c("time", "status", "age", "sex", "ph.ecog")])

# # Fit Kaplan-Meier model on cleaned data
# km_fit <- survfit(Surv(time, status) ~ 1, data = lung_clean)

# # Fit Cox proportional hazards model on cleaned data
# cox_fit <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung_clean)

# # Create Kaplan-Meier cumulative hazard plot without the plot layer
# km_cumhaz_data <- ggsurvplot(km_fit, data = lung_clean, fun = "cumhaz", conf.int = FALSE,
#                              legend = "none", ggtheme = theme_minimal(), risk.table = FALSE)

# # Extract the data from the Kaplan-Meier cumulative hazard plot
# km_cumhaz_df <- km_cumhaz_data$plot$data

# # Create Cox model cumulative hazard plot without the plot layer
# cox_cumhaz_data <- ggsurvplot(survfit(cox_fit, data = lung_clean), data = lung_clean, 
#                               fun = "cumhaz", conf.int = FALSE, legend = "none", 
#                               ggtheme = theme_minimal(), risk.table = FALSE)

# # Extract the data from the Cox model cumulative hazard plot
# cox_cumhaz_df <- cox_cumhaz_data$plot$data

# # Combine the two datasets into a single data frame
# combined_cumhaz_df <- rbind(
#   data.frame(time = km_cumhaz_df$time, cumhaz = km_cumhaz_df$surv, model = "Kaplan-Meier"),
#   data.frame(time = cox_cumhaz_df$time, cumhaz = cox_cumhaz_df$surv, model = "Cox Model")
# )

# # Plot the combined cumulative hazard curves
# combined_cumhaz_plot <- ggplot(combined_cumhaz_df, aes(x = time, y = cumhaz, color = model)) +
#   geom_line(size = 1) +
#   scale_color_manual(values = c("Kaplan-Meier" = "#2E9FDF", "Cox Model" = "#E7B800")) +
#   labs(title = "Cumulative Hazard: Kaplan-Meier vs Cox Model",
#        x = "Time",
#        y = "Cumulative Hazard",
#        color = "Model") +
#   theme_minimal() +
#   theme(legend.position = "top")

# # Display the combined plot
# print(combined_cumhaz_plot)

