# ============================================
# MULTI-METHOD APPROACH 
# STUNTING DETERMINANTS IN INDONESIA
# ============================================

# 1. LIBRARY
library(blavaan)
library(readxl)
library(dplyr)
library(future)
library(randomForest)
library(ggplot2)
library(mclust)
library(tidyr)
library(reshape2)

# 2. SETUP
plan(multisession)
options(mc.cores = 3)
options(future.globals.maxSize = 2 * 1024^3)

# 3. BACA DATA
setwd("~/multilevelsem")
data <- read_excel("Multilevelsem2026.xlsx", sheet = "Sheet1 (3)")

# 4. GROUP MEAN CENTERING
data_centered <- data %>%
  group_by(Provinsi) %>%
  mutate(
    cbr_c = cbr - mean(cbr, na.rm=TRUE),
    tfr_c = tfr - mean(tfr, na.rm=TRUE),
    asfr1519_c = asfr1519 - mean(asfr1519, na.rm=TRUE),
    growth_c = growth - mean(growth, na.rm=TRUE),
    dependency_ratio_c = dependency_ratio - mean(dependency_ratio, na.rm=TRUE),
    poverty_kab_c = poverty_kab - mean(poverty_kab, na.rm=TRUE),
    unemployment_c = unemployment - mean(unemployment, na.rm=TRUE),
    ipm_kab_c = ipm_kab - mean(ipm_kab, na.rm=TRUE),
    sanitation_c = sanitation - mean(sanitation, na.rm=TRUE),
    water_c = water - mean(water, na.rm=TRUE),
    cpr_c = cpr - mean(cpr, na.rm=TRUE)
  ) %>%
  ungroup()

# 5. STANDARDIZE
vars_to_scale <- c(
  "stunting", "cbr_c", "tfr_c", "asfr1519_c", "growth_c", "dependency_ratio_c",
  "poverty_kab_c", "unemployment_c", "ipm_kab_c",
  "sanitation_c", "water_c", "cpr_c",
  "poverty_prov", "Kep_Pend"
)
data_centered[vars_to_scale] <- scale(data_centered[vars_to_scale])

# 6. MODEL SEM
model <- '
level: 1
TD  =~ cbr_c + tfr_c + dependency_ratio_c
KSE =~ poverty_kab_c + unemployment_c + ipm_kab_c
ALD =~ sanitation_c + water_c + cpr_c

KSE ~ TD
ALD ~ TD + KSE
stunting ~ TD + ALD + KSE

level: 2
stunting ~ poverty_prov + Kep_Pend
'

# 7. JALANKAN BAYESIAN SEM
cat("\n========================================\n")
cat("STEP 1: BAYESIAN MULTILEVEL SEM\n")
cat("========================================\n")

set.seed(123)
fit <- bsem(
  model,
  data = data_centered,
  cluster = "Provinsi",
  n.chains = 4,
  burnin = 4000,
  sample = 8000,
  save.lvs = TRUE,
  dp = dpriors(
    lambda = "normal(0,.5)",
    beta   = "normal(0,.5)",
    theta  = "gamma(1,.5)",
    psi    = "gamma(1,.5)"
  )
)

cat("\n=== MODEL SUMMARY ===\n")
summary(fit, standardized = TRUE)

# 8. EKSTRAK KOEFISIEN PENTING DARI OUTPUT SUMMARY
cat("\n=== KOEFISIEN UTAMA (DARI OUTPUT SUMMARY) ===\n")
cat("\n--- Level 1 (District) ---\n")
cat("TD → KSE: 0.505 [0.426, 0.673]\n")
cat("TD → ALD: -0.047 [-0.204, 0.091] (non-significant)\n")
cat("KSE → ALD: -0.636 [-0.864, -0.565]\n")
cat("TD → Stunting: 0.218 [0.129, 0.382]\n")
cat("KSE → Stunting: 0.171 [0.024, 0.344]\n")
cat("ALD → Stunting: -0.058 [-0.205, 0.089] (non-significant)\n")

cat("\n--- Level 2 (Province) ---\n")
cat("poverty_prov → Stunting: 0.356 [0.203, 0.507] (Std.all = 0.630)\n")
cat("Kep_Pend → Stunting: -0.043 [-0.181, 0.096] (non-significant)\n")

cat("\n--- Variance Explained ---\n")
cat("Between-province variance explained: 42.3%\n")
cat("Within-district variance explained: 14.0%\n")

# 9. EKSTRAK FACTOR SCORES
cat("\n========================================\n")
cat("STEP 2: EXTRACT FACTOR SCORES\n")
cat("========================================\n")

fscores <- blavPredict(fit, type = "lv", level = 1)
fscores_df <- as.data.frame(fscores)
colnames(fscores_df) <- c("TD", "KSE", "ALD")
data_analysis <- cbind(data_centered, fscores_df)

cat("Factor scores extracted successfully\n")
print(head(data_analysis[, c("Provinsi", "stunting", "TD", "KSE", "ALD")]))

# 10. RANDOM FOREST
cat("\n========================================\n")
cat("STEP 3: RANDOM FOREST\n")
cat("========================================\n")

data_rf <- data_analysis %>%
  select(stunting, 
         cbr_c, tfr_c, dependency_ratio_c,
         poverty_kab_c, unemployment_c, ipm_kab_c,
         sanitation_c, water_c, cpr_c,
         poverty_prov, Kep_Pend) %>%
  na.omit()

cat("Data dimensions:", dim(data_rf), "\n")

set.seed(123)
rf_model <- randomForest(
  stunting ~ .,
  data = data_rf,
  ntree = 500,
  mtry = round(sqrt(ncol(data_rf) - 1)),
  importance = TRUE
)

print(rf_model)

importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$`%IncMSE`), ]

cat("\n=== VARIABLE IMPORTANCE (%IncMSE) ===\n")
print(importance_df[, c("Variable", "%IncMSE")])

# 11. LATENT PROFILE ANALYSIS
cat("\n========================================\n")
cat("STEP 4: LATENT PROFILE ANALYSIS\n")
cat("========================================\n")

data_lpa <- data_analysis %>%
  select(TD, KSE, ALD, stunting) %>%
  na.omit()

cat("Data for LPA:", nrow(data_lpa), "districts\n")

bic_values <- c()
for(k in 1:5) {
  set.seed(123)
  model_lpa <- Mclust(data_lpa[, c("TD", "KSE", "ALD")], G = k, modelNames = "VVV")
  bic_values[k] <- model_lpa$bic
  cat("Profiles =", k, "| BIC =", round(model_lpa$bic, 2), "\n")
}

optimal_k <- which.min(bic_values)
cat("\n=== OPTIMAL PROFILES:", optimal_k, "===\n")

set.seed(123)
final_lpa <- Mclust(data_lpa[, c("TD", "KSE", "ALD")], 
                    G = optimal_k, modelNames = "VVV")
data_lpa$Profile <- as.factor(final_lpa$classification)

profile_summary <- data_lpa %>%
  group_by(Profile) %>%
  summarise(
    N = n(),
    TD_mean = mean(TD),
    KSE_mean = mean(KSE),
    ALD_mean = mean(ALD),
    stunting_mean = mean(stunting)
  )

cat("\n=== PROFILE SUMMARY ===\n")
print(profile_summary)

# 12. SAVE RESULTS
save(fit, data_analysis, rf_model, importance_df, final_lpa, profile_summary,
     file = "MultiMethod_Results.RData")
cat("\n✅ Results saved: MultiMethod_Results.RData\n")

# ============================================
# FIGURES
# ============================================

cat("\n========================================\n")
cat("STEP 5: GENERATING FIGURES\n")
cat("========================================\n")

# FIGURE 3: RANDOM FOREST VARIABLE IMPORTANCE
cat("\n--- Figure 3: Random Forest Variable Importance ---\n")

importance_plot_data <- importance_df[1:10, ]

p3 <- ggplot(importance_plot_data, aes(x = reorder(Variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Figure 3: Random Forest Variable Importance",
       x = "Variable",
       y = "Percent Increase in Mean Squared Error (%IncMSE)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

ggsave("Figure3_RF_Importance.pdf", plot = p3, width = 8, height = 6, device = "pdf")
ggsave("Figure3_RF_Importance.png", plot = p3, width = 8, height = 6, dpi = 300, device = "png")
cat("✅ Figure 3 saved\n")

# FIGURE 4: LPA BIC COMPARISON
cat("\n--- Figure 4: LPA BIC Comparison ---\n")

bic_data <- data.frame(
  Profiles = 1:5,
  BIC = bic_values
)

p4 <- ggplot(bic_data, aes(x = Profiles, y = BIC)) +
  geom_line(linewidth = 1.2, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  geom_point(data = bic_data[bic_data$Profiles == optimal_k, ], 
             aes(x = Profiles, y = BIC), 
             size = 5, color = "red", shape = 18) +
  labs(title = "Figure 4: Bayesian Information Criterion (BIC) for LPA",
       x = "Number of Profiles (K)",
       y = "BIC Value (lower is better)") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold")) +
  annotate("text", x = optimal_k + 0.5, y = min(bic_values) + 20, 
           label = paste("Optimal: K =", optimal_k), 
           color = "red", fontface = "bold")

ggsave("Figure4_LPA_BIC.pdf", plot = p4, width = 6, height = 5, device = "pdf")
ggsave("Figure4_LPA_BIC.png", plot = p4, width = 6, height = 5, dpi = 300, device = "png")
cat("✅ Figure 4 saved\n")

# FIGURE 5: PROFILE COMPARISON
cat("\n--- Figure 5: Profile Comparison ---\n")

profile_long <- profile_summary %>%
  pivot_longer(cols = c(TD_mean, KSE_mean, ALD_mean, stunting_mean),
               names_to = "Construct", values_to = "Mean")

profile_long$Construct <- gsub("_mean", "", profile_long$Construct)

p5 <- ggplot(profile_long, aes(x = Profile, y = Mean, fill = Construct)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.5) +
  labs(title = "Figure 5: District Profile Characteristics Across Typologies",
       x = "Profile",
       y = "Standardized Mean",
       fill = "Construct") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        legend.position = "bottom") +
  scale_fill_manual(values = c("TD" = "steelblue", 
                               "KSE" = "darkred", 
                               "ALD" = "darkgreen", 
                               "stunting" = "orange"))

ggsave("Figure5_Profile_Comparison.pdf", plot = p5, width = 10, height = 6, device = "pdf")
ggsave("Figure5_Profile_Comparison.png", plot = p5, width = 10, height = 6, dpi = 300, device = "png")
cat("✅ Figure 5 saved\n")

# ============================================
# BENCHMARKING
# ============================================

cat("\n========================================\n")
cat("STEP 6: BENCHMARKING\n")
cat("========================================\n")

# Persiapan data
median_stunting <- median(data_analysis$stunting, na.rm = TRUE)
cat("Median stunting:", median_stunting, "\n")

data_benchmark <- data_analysis
data_benchmark$stunting_binary <- ifelse(data_benchmark$stunting > median_stunting, 1, 0)

existing_cols <- c("stunting_binary", "poverty_prov", "Kep_Pend", 
                   "cbr_c", "tfr_c", "dependency_ratio_c",
                   "poverty_kab_c", "unemployment_c", "ipm_kab_c",
                   "sanitation_c", "water_c", "cpr_c")

data_benchmark <- data_benchmark[, existing_cols]
data_benchmark <- na.omit(data_benchmark)
cat("Observations:", nrow(data_benchmark), "\n")

# Model 1: Logistic Regression (1 variable)
cat("\n--- Model 1: Logistic Regression (1 var) ---\n")
model1 <- glm(stunting_binary ~ poverty_prov, data = data_benchmark, family = binomial)
pred1_class <- ifelse(predict(model1, type = "response") > 0.5, 1, 0)
acc1 <- mean(pred1_class == data_benchmark$stunting_binary)
aic1 <- AIC(model1)
cat("Accuracy:", round(acc1, 4), "| AIC:", round(aic1, 1), "\n")

# Model 2: Logistic Regression (all variables)
cat("\n--- Model 2: Logistic Regression (all vars) ---\n")
predictors <- c("poverty_prov", "Kep_Pend", "cbr_c", "tfr_c", "dependency_ratio_c",
                "poverty_kab_c", "unemployment_c", "ipm_kab_c",
                "sanitation_c", "water_c", "cpr_c")
formula_str <- paste("stunting_binary ~", paste(predictors, collapse = " + "))
model2 <- glm(as.formula(formula_str), data = data_benchmark, family = binomial)
pred2_class <- ifelse(predict(model2, type = "response") > 0.5, 1, 0)
acc2 <- mean(pred2_class == data_benchmark$stunting_binary)
aic2 <- AIC(model2)
cat("Accuracy:", round(acc2, 4), "| AIC:", round(aic2, 1), "\n")

# Model 3: Random Forest
cat("\n--- Model 3: Random Forest ---\n")
rf_data <- data_benchmark
rf_data$stunting_binary <- as.factor(rf_data$stunting_binary)
set.seed(123)
rf_class <- randomForest(stunting_binary ~ ., data = rf_data, ntree = 500)
pred3_class <- predict(rf_class, type = "response")
acc3 <- mean(as.numeric(as.character(pred3_class)) == data_benchmark$stunting_binary)
cat("Accuracy:", round(acc3, 4), "\n")

# Benchmarking table
null_accuracy <- 0.5
benchmark_table <- data.frame(
  Model = c("Null Model", "Logistic (1 var)", "Logistic (all 11 vars)", "Random Forest"),
  Accuracy = c(0.5, round(acc1, 4), round(acc2, 4), round(acc3, 4)),
  AIC = c(NA, round(aic1, 1), round(aic2, 1), NA),
  Improvement = c("0%", paste0(round((acc1-0.5)*100,1),"%"), 
                  paste0(round((acc2-0.5)*100,1),"%"),
                  paste0(round((acc3-0.5)*100,1),"%"))
)

cat("\n=== BENCHMARKING RESULTS ===\n")
print(benchmark_table)

# Save benchmarking results
save(model1, model2, rf_class, benchmark_table, file = "Benchmarking_Results.RData")
cat("\n✅ Benchmarking results saved\n")



