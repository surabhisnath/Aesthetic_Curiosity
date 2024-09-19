# Load necessary libraries
library(interactions)
library(ggplot2)

# Use the mtcars dataset
data(mtcars)

# Fit a linear model with an interaction term
model <- lm(mpg ~ wt * factor(cyl), data = mtcars)
mtcars$cyl <- as.factor(mtcars$cyl)

# Generate the interaction plot
interact_plot(model, pred = wt, modx = cyl, 
              plot.points = TRUE,       # Show actual data points
              interval = TRUE,          # Show confidence intervals
              int.width = 0.95,         # 95% confidence interval
              modx.values = NULL,       # Default values (-1SD, Mean, +1SD)
              x.label = "Weight (1000 lbs)",
              y.label = "Miles per Gallon",
              legend.main = "Number of Cylinders",
              colors = "Qual1")         # Use a qualitative color palette
