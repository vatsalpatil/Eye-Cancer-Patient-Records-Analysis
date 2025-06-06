# Eye Cancer Patient Data Analysis Report

![EyeCancerPatientsDataAnalysisImage](https://github.com/user-attachments/assets/cb76c0d5-bf88-41d7-a291-2405c22c81cc)


This report presents a comprehensive analysis of a dataset containing medical and demographic information for 5,000 patients diagnosed with eye cancer, including Melanoma, Retinoblastoma, and Lymphoma. The dataset includes patient demographics (age, gender, country), clinical details (cancer type, laterality, stage at diagnosis), treatment information (type, intensity), outcomes (survival time, outcome status), and genetic factors (genetic markers, family history). The analysis aims to uncover patterns, relationships, and predictors of patient outcomes to inform medical research and clinical practice. Below, we summarize the key findings from the statistical, survival, clustering, and predictive analyses performed.

## 1. Univariate Analysis: Understanding Patient Characteristics

### Age Distribution
- **Summary**: The patient population has a mean age of 45.0 years (SD = 25.9), with ages ranging from 1 to 90 years. The distribution is relatively uniform, with a median age of 44 years, indicating a diverse age range affected by eye cancer.
- **Insight**: The wide age range suggests that eye cancer affects both pediatric and adult populations, necessitating age-specific treatment approaches.

### Gender Distribution
- **Summary**: The dataset includes 1,652 males (33.0%), 1,628 females (32.6%), and 1,720 patients identified as "Other" (34.4%), showing a balanced gender distribution.
- **Insight**: The balanced gender distribution indicates no strong gender predisposition for eye cancer in this cohort, allowing for gender-agnostic analyses.

### Cancer Type Distribution
- **Summary**: The dataset includes three cancer types: Melanoma (1,691 patients, 33.8%), Retinoblastoma (1,672 patients, 33.4%), and Lymphoma (1,637 patients, 32.7%), with roughly equal prevalence.
- **Insight**: The near-equal distribution of cancer types suggests a diverse patient cohort, suitable for comparative analyses across cancer types.

### Laterality Distribution
- **Summary**: Laterality is distributed as Bilateral (1,695 patients, 33.9%), Left (1,686 patients, 33.7%), and Right (1,619 patients, 32.4%), indicating no significant bias toward one eye.
- **Insight**: The balanced laterality distribution suggests that eye cancer does not predominantly affect one side, which is important for surgical and treatment planning.

### Stage at Diagnosis
- **Summary**: Patients are diagnosed across four stages: Stage II (1,287 patients, 25.7%), Stage III (1,281 patients, 25.6%), Stage IV (1,242 patients, 24.8%), and Stage I (1,190 patients, 23.8%).
- **Insight**: The even distribution across stages indicates a mix of early and advanced diagnoses, highlighting the importance of early detection to potentially improve outcomes.

### Outcome Status
- **Summary**: Outcomes are distributed as Deceased (1,710 patients, 34.2%), In Remission (1,675 patients, 33.5%), and Active (1,615 patients, 32.3%).
- **Insight**: The similar proportions of outcomes suggest a complex interplay of factors influencing survival, warranting further investigation into predictors of remission or mortality.

### Survival Time
- **Summary**: Survival time ranges from 1 to 120 months, with a mean of 60.7 months (SD = 34.5) and a median of 60 months.
- **Insight**: The wide range and moderate mean survival time indicate variability in patient outcomes, likely influenced by cancer type, stage, and treatment.

### Country Distribution
- **Summary**: The top 10 countries by patient count are South Africa (544), Australia (513), France (505), UK (504), Japan (496), India (495), Brazil (491), USA (489), Canada (486), and Germany (477).
- **Insight**: The global distribution of patients suggests potential geographic variations in incidence or reporting, which could be explored for environmental or healthcare system influences.

### Genetic Markers
- **Summary**: Genetic markers are split between BRAF Mutation (2,503 patients, 50.1%) and None (2,497 patients, 49.9%).
- **Insight**: The near-equal split suggests that genetic predisposition (e.g., BRAF mutation) is a significant factor for half the cohort, relevant for targeted therapies.

### Family History
- **Summary**: Family history of eye cancer is present in 2,462 patients (49.2%) and absent in 2,538 patients (50.8%).
- **Insight**: The high prevalence of family history indicates a potential hereditary component, which could guide genetic screening protocols.

## 2. Bivariate Analysis: Exploring Relationships

### Age by Cancer Type
- **Result**: Mean ages are similar across cancer types (Retinoblastoma: 45.2, Lymphoma: 45.0, Melanoma: 44.9). The Kruskal-Wallis test (stat = 0.11, p = 0.9454) shows no significant difference.
- **Conclusion**: Age does not significantly vary by cancer type, suggesting that all three cancers affect patients across similar age ranges.

### Treatment Type vs. Outcome Status
- **Result**: The crosstab shows balanced distributions across Chemotherapy, Radiation, and Surgery for Active, Deceased, and In Remission outcomes. The Chi-Square test (chi2 = 4.27, p = 0.3706) indicates no significant association.
- **Conclusion**: Treatment type alone does not strongly predict outcome status, suggesting other factors (e.g., stage, genetics) may play a larger role.

### Genetic Markers vs. Cancer Type
- **Result**: BRAF Mutation and None are evenly distributed across Lymphoma, Melanoma, and Retinoblastoma. The Chi-Square test (chi2 = 0.51, p = 0.7732) shows no significant association.
- **Conclusion**: Genetic markers (BRAF vs. None) do not strongly correlate with specific cancer types, indicating that other genetic or environmental factors may drive cancer type.

### Family History vs. Outcome Status
- **Result**: Family history (True/False) shows similar outcome distributions. The Chi-Square test (chi2 = 0.25, p = 0.8823) indicates no significant association.
- **Conclusion**: Family history does not significantly influence outcome status, suggesting that hereditary factors may not be primary drivers of prognosis in this cohort.

### Stage at Diagnosis vs. Outcome Status
- **Result**: Outcomes are relatively balanced across stages I–IV. The Chi-Square test (chi2 = 6.40, p = 0.3804) shows no significant association.
- **Conclusion**: Stage at diagnosis does not strongly predict outcome status, which is surprising and may indicate effective treatments across stages or data limitations.

## 3. Survival Analysis: Kaplan-Meier Estimates
- **Result**: Median survival times are 106 months for Retinoblastoma, 104 months for Melanoma, and 104 months for Lymphoma. Survival probabilities at 12, 24, 60, and 120 months are similar across cancer types, with Retinoblastoma showing slightly higher survival at 120 months (12.2%) compared to Melanoma (16.6%) and Lymphoma (19.5%).
- **Conclusion**: Survival curves indicate comparable long-term survival across cancer types, with Retinoblastoma patients having a slight advantage at longer time points. This suggests similar disease progression patterns, possibly due to effective treatments.

## 4. Geo-Demographic Trends
- **Result**: Among the top 10 countries, Canada has the highest average survival time (62.8 months), followed by Germany (62.4 months) and the USA (62.4 months). South Africa has the lowest (62.1 months). Remission rates range from 31.1% (Germany) to 35.8% (Canada), and deceased rates are similar (32.1%–36.5%).
- **Conclusion**: Slight variations in survival time and remission rates across countries may reflect differences in healthcare access, treatment quality, or patient demographics, though differences are modest.

## 5. Interaction Effects
### Age Group and Stage vs. Outcome Status
- **Result**: Outcomes are distributed across age groups (Child, Young Adult, Adult, Senior) and stages (I–IV) without clear patterns. For example, Seniors in Stage I have higher counts of Deceased (150) and In Remission (147) compared to Active (110).
- **Conclusion**: No strong interaction effect is evident, suggesting that age and stage combined do not distinctly predict outcomes, possibly due to balanced treatment efficacy across groups.

### Treatment Type and Genetic Markers vs. Outcome
- **Result**: Outcomes are balanced across combinations of treatment type and genetic markers. For example, Surgery with BRAF Mutation has 304 In Remission cases, while Chemotherapy with None has 303 Deceased cases.
- **Conclusion**: The lack of strong patterns suggests that treatment and genetic markers interact complexly, and neither alone strongly predicts outcomes.

## 6. Time-Based Analysis
- **Result**: From 2019 to 2024, average survival time peaks in 2024 (62.9 months) and is lowest in 2021 (59.5 months). Remission rates range from 31.9% (2019) to 36.0% (2023), and deceased rates are stable (33.1%–35.3%).
- **Conclusion**: A slight increase in survival time in recent years (2024) may indicate improvements in treatment or earlier diagnosis, though remission and deceased rates remain stable.

## 7. Clustering Analysis
- **Result**: K-means clustering (3 clusters) identified:
  - **Cluster 0**: Younger patients (mean age 22.2 years), shorter survival (35.7 months), mostly Retinoblastoma, predominantly Deceased.
  - **Cluster 1**: Middle-aged patients (mean age 45.9 years), longest survival (94.7 months), mostly Melanoma, predominantly Deceased.
  - **Cluster 2**: Older patients (mean age 69.4 years), shorter survival (34.8 months), mostly Melanoma, predominantly In Remission.
- **Conclusion**: Clusters reveal distinct patient profiles based on age and survival time, with younger and older patients having shorter survival times, possibly due to aggressive disease or comorbidities, while middle-aged patients fare better.

## 8. Feature Importance for Outcome Prediction
- **Result**: Random Forest analysis identifies Age (27.4%), Radiation Therapy (26.6%), and Chemotherapy (20.4%) as the top predictors of outcome status, followed by Stage at Diagnosis (7.8%), Cancer Type (6.2%), Treatment Type (5.2%), Family History (3.5%), and Genetic Markers (2.9%). The model has low accuracy (33%), with precision, recall, and F1-scores around 0.33 for all outcomes.
- **Conclusion**: Age and treatment intensity (radiation, chemotherapy) are the most influential factors for predicting outcomes, though the model's low accuracy suggests complex interactions or insufficient predictive power in the features used.

## 9. Treatment Intensity Patterns
- **Result**: Mean treatment intensity (radiation dose + chemotherapy sessions) is highest for Lymphoma at Stage III (46.9) and lowest for Retinoblastoma at Stage I (43.0). Intensities are similar across cancer types and stages (43.0–46.9).
- **Conclusion**: Treatment intensity does not vary significantly by cancer type or stage, suggesting standardized treatment protocols across the cohort.

## 10. Survival Disparities by Laterality
- **Result**: Mean survival times are similar across laterality: Right (61.9 months), Left (60.5 months), Bilateral (59.8 months). Remission and deceased rates are also comparable. The Cox model shows no significant effect of laterality (p = 0.2594) or age (p = 0.7847) on survival.
- **Conclusion**: Laterality does not significantly impact survival outcomes, indicating that the affected eye(s) do not influence prognosis in this dataset.

## Overall Conclusions
1. **Patient Diversity**: The dataset reflects a diverse cohort with balanced distributions across age, gender, cancer type, laterality, and stage, making it suitable for broad analyses but challenging for identifying strong predictors due to uniformity.
2. **Limited Predictive Power**: Bivariate analyses (e.g., Chi-Square tests) show no significant associations between key variables (e.g., treatment type, genetic markers, stage) and outcomes, suggesting complex interactions or data limitations.
3. **Survival Patterns**: Survival times are comparable across cancer types, with slight advantages for Retinoblastoma at longer time points. Clustering reveals distinct patient groups, with middle-aged patients having the longest survival.
4. **Key Predictors**: Age and treatment intensity (radiation, chemotherapy) are the most important predictors of outcome, but the Random Forest model's low accuracy indicates that additional features or more complex models may be needed.
5. **Geo-Demographic Insights**: Modest variations in survival and remission rates across countries suggest potential influences of healthcare systems, but differences are not substantial.
6. **Clinical Implications**: The lack of strong associations with stage, genetics, or laterality suggests that current treatments may be equally effective across subgroups, but personalized approaches could be explored for younger and older patients with poorer outcomes.

## Recommendations
- **Further Data Collection**: Include additional features (e.g., specific treatment protocols, genetic subtypes beyond BRAF) to improve predictive models.
- **Advanced Modeling**: Explore non-linear models (e.g., neural networks) or interaction terms to capture complex relationships.
- **Focus on Age Groups**: Tailor interventions for younger and older patients, who show shorter survival times in clustering analysis.
- **Geographic Studies**: Investigate healthcare system differences in countries like Canada and South Africa to understand survival variations.

This analysis provides a foundation for understanding eye cancer patient outcomes and highlights areas for further research to improve prognosis and treatment strategies.
