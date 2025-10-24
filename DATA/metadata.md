# Air Quality Dataset Establishment

## Summary of Data

The dataset is a `.CSV` file containing **9,358 instances** of hourly averaged responses from **five different metal oxide chemical sensors** in an Italian city.  
The data was recorded between **March 2004 and February 2005**, and includes measurements of:

- Carbon monoxide (**CO**)  
- Non-metanic hydrocarbons (**NMHC**)  
- Benzene (**C₆H₆**)  
- Total nitrogen oxides (**NOx**)  
- Nitrogen dioxide (**NO₂**)  

Each observation also includes **date** and **time** information for when pollutant concentrations were recorded.  

**Original dataset:**  
[UCI Machine Learning Repository – Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)

---

## Provenance

The dataset contains **hourly averages** of air pollutant concentrations, temperature, and humidity collected in an Italian city.  
The data was gathered by **Saverio Vito** between **2004 and 2005**, and described in the paper:

> *On field calibration of an electronic nose for benzene estimation in an urban pollution monitoring scenario* (2008)

The dataset is publicly available through the **UC Irvine Machine Learning Repository** (linked above).  
It was **downloaded on October 15, 2025**, and uploaded to this project’s GitHub repository:

[GitHub Repository – DS 4002 Project 2](https://github.com/Seanhru/project2)

Since download, the data has been **cleaned, modified, and analyzed** for this project.  
All preprocessing steps, models, and results are **fully documented** in the repository.

---

## License

The dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.  
You are free to **share** and **adapt** the data as long as appropriate **credit is given**.  
Commercial use of the dataset is **not permitted**.

---

## Data Dictionary

| **Column** | **Description** | **Data Type / Example** |
|-------------|----------------|-------------------------|
| **Date** | Month/day/year that data was collected. | Date variable. Example: `3/10/2004` |
| **Time** | Time that data was collected. | Categorical variable. Example: `18:00:00` |
| **CO(GT)** | True hourly averaged concentration of CO in mg/m³. | Integer. Example: `2.6` |
| **PT08.S1(CO)** | Hourly averaged sensor response for CO. | Categorical. Example: `1360` |
| **NMHC(GT)** | True hourly averaged overall concentration of NMHC in µg/m³. | Integer. Example: `150` |
| **C6H6(GT)** | True hourly averaged concentration of benzene in µg/m³. | Continuous. Example: `11.9` |
| **PT08.S2(NMHC)** | Hourly averaged sensor response for NMHC. | Categorical. Example: `1046` |
| **NOx(GT)** | True hourly averaged concentration of NOx in ppb. | Integer. Example: `166` |
| **PT08.S3(NOx)** | Hourly averaged sensor response for NOx. | Categorical. Example: `1056` |
| **NO2(GT)** | True hourly averaged concentration of NO₂ in µg/m³. | Integer. Example: `113` |
| **PT08.S4(NO2)** | Hourly averaged sensor response for NO₂. | Categorical. Example: `1692` |
| **PT08.S5(O3)** | Hourly averaged sensor response for O₃. | Categorical. Example: `1268` |
| **T** | Temperature in °C. | Continuous. Example: `13.6` |
| **RH** | Relative humidity in %. | Continuous. Example: `48.9` |
| **AH** | Absolute humidity. | Continuous. Example: `0.7578` |

---

## Ethical Statement

Results from this model **may not accurately represent** the real-world air quality of the studied location.  
Before using these findings to guide **policy**, **public health**, or **environmental initiatives**, additional **validation and testing** should be conducted.  
This ensures that results are not **misinterpreted or misused** in decision-making contexts.

---

## Exploratory Plots
[EDA1.pdf](https://github.com/user-attachments/files/23126729/EDA1.pdf)
[EDA2.pdf](https://github.com/user-attachments/files/23126731/EDA2.pdf)
[EDA3.pdf](https://github.com/user-attachments/files/23126733/EDA3.pdf)

