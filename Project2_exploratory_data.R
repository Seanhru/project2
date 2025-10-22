# Project 2 Exploritory Data

air <- read.csv("/Users/arianaelahi/Desktop/cleaned_air_quality.csv")
View(air)

library(tidyverse)


# Plot 1
ggplot(data, aes(x = T, y = RH)) +
  geom_point(alpha = 0.5, color = "darkorange") +
  labs(title = "Temperature vs Relative Humidity",
       x = "Temperature (°C)", y = "Relative Humidity (%)") +
  theme_minimal()


#Plot 2

# Select only numeric columns
num_data <- select_if(air, is.numeric)

# Compute correlation matrix
corr_matrix <- cor(num_data, use = "complete.obs")

library(reshape2)
melted_corr <- melt(corr_matrix)

ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "blue", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap of Sensor Variables", x = "", y = "") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Plot 3
library(tidyverse)
library(GGally)

names(air) <- make.names(names(air), unique = TRUE)

# Keep only numeric columns
numeric_data <- air %>%
  select_if(is.numeric) %>%
  drop_na()

# Create a pairwise plot (scatterplots + correlations)
ggpairs(numeric_data,
        title = "Pairwise Relationships Between Air Quality Variables",
        upper = list(continuous = "cor"),
        lower = list(continuous = "points"),
        diag = list(continuous = "densityDiag"))

