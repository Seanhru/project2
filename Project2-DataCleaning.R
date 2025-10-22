#Cleaning data

data <- read.csv("/Users/arianaelahi/Desktop/OriginalAirQualityUCI.csv",
                 sep = ";", dec = ",", na.strings = c("NA", "-200"))
names(data) <- make.names(names(data), unique = TRUE)

write.csv(data, "/Users/arianaelahi/Desktop/cleaned_air_quality.csv", row.names = FALSE)

