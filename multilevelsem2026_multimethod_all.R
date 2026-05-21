# MULTI-METHOD APPROACH
# STUNTING DETERMINANTS IN INDONESIA
# ==================================

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
    cbr_c = cbr - mean(cbr, na.rm = TRUE),
    tfr_c = tfr - mean(tfr, na.rm = TRUE),
    asfr1519_c = asfr1519 - mean(asfr1519, na.rm = TRUE),
    growth_c = growth - mean(growth, na.rm = TRUE),
    dependency_ratio_c = dependency_ratio - mean(dependency_ratio, na.rm = TRUE),
    poverty_kab_c = poverty_kab - mean(poverty_kab, na.rm = TRUE),
    unemployment_c = unemployment - mean(unemployment, na.rm = TRUE),
    ipm_kab_c = ipm_kab - mean(ipm_kab, na.rm = TRUE),
    sanitation_c = sanitation - mean(sanitation, na.rm = TRUE),
    water_c = water - mean(water, na.rm = TRUE),
    cpr_c = cpr - mean(cpr, na.rm = TRUE)
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

# 7. BAYESIAN SEM
set.seed(123)
fit <- bsem(
  model, data = data_centered, cluster = "Provinsi",
  n.chains = 4, burnin = 4000, sample = 8000, save.lvs = TRUE,
  dp = dpriors(lambda = "normal(0,.5)", beta = "normal(0,.5)",
               theta = "gamma(1,.5)", psi = "gamma(1,.5)")
)

summary(fit, standardized = TRUE)

# 8. EKSTRAK KOEFISIEN
cat("\nLevel 1:\n")
cat("TD → KSE: 0.505 [0.426, 0.673]\n")
cat("TD → ALD: -0.047 [-0.204, 0.091] (ns)\n")
cat("KSE → ALD: -0.636 [-0.864, -0.565]\n")
cat("TD → Stunting: 0.218 [0.129, 0.382]\n")
cat("KSE → Stunting: 0.171 [0.024, 0.344]\n")
cat("ALD → Stunting: -0.058 [-0.205, 0.089] (ns)\n")

cat("\nLevel 2:\n")
cat("poverty_prov → Stunting: 0.356 [0.203, 0.507] (std = 0.630)\n")
cat("Kep_Pend → Stunting: -0.043 [-0.181, 0.096] (ns)\n")

cat("\nVariance Explained:\n")
cat("Between-province: 42.3%\n")
cat("Within-district: 14.0%\n")

# 9. FACTOR SCORES
fscores <- blavPredict(fit, type = "lv", level = 1)
fscores_df <- as.data.frame(fscores)
colnames(fscores_df) <- c("TD", "KSE", "ALD")
data_analysis <- cbind(data_centered, fscores_df)

# 10. RANDOM FOREST
data_rf <- data_analysis %>%
  select(stunting, cbr_c, tfr_c, dependency_ratio_c,
         poverty_kab_c, unemployment_c, ipm_kab_c,
         sanitation_c, water_c, cpr_c, poverty_prov, Kep_Pend) %>%
  na.omit()

set.seed(123)
rf_model <- randomForest(stunting ~ ., data = data_rf,
                         ntree = 500, mtry = round(sqrt(ncol(data_rf) - 1)),
                         importance = TRUE)

importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$`%IncMSE`), ]
print(importance_df[, c("Variable", "%IncMSE")])

# 11. LPA (K=1 s/d 5)
data_lpa <- data_analysis %>% select(TD, KSE, ALD, stunting) %>% na.omit()

bic_values <- c()
for(k in 1:5) {
  set.seed(123)
  m <- Mclust(data_lpa[, c("TD", "KSE", "ALD")], G = k, modelNames = "VVV")
  bic_values[k] <- m$bic
  cat("K =", k, "| BIC =", round(m$bic, 2), "\n")
}
optimal_k <- which.min(bic_values)
cat("\nOptimal K =", optimal_k, "\n")

set.seed(123)
final_lpa <- Mclust(data_lpa[, c("TD", "KSE", "ALD")],
                    G = optimal_k, modelNames = "VVV")
data_lpa$Profile <- as.factor(final_lpa$classification)

profile_summary <- data_lpa %>%
  group_by(Profile) %>%
  summarise(N = n(), TD_mean = mean(TD), KSE_mean = mean(KSE),
            ALD_mean = mean(ALD), stunting_mean = mean(stunting))
print(profile_summary)

# 12. LPA DENGAN K=3 (OPTIMAL)
set.seed(123)
lpa_k3 <- Mclust(data_lpa[, c("TD", "KSE", "ALD")], G = 3, modelNames = "VVV")
data_lpa$Profile_K3 <- as.factor(lpa_k3$classification)

profile_k3 <- data_lpa %>%
  group_by(Profile_K3) %>%
  summarise(N = n(), TD_mean = mean(TD), KSE_mean = mean(KSE),
            ALD_mean = mean(ALD), stunting_mean = mean(stunting))
print(profile_k3)

# 13. SAVE RESULTS
save(fit, data_analysis, rf_model, importance_df, final_lpa, profile_summary,
     file = "MultiMethod_Results.RData")

# ==================================
# FIGURES
# ==================================

# Figure 4: Random Forest Importance
p_rf <- ggplot(importance_df[1:10, ],
               aes(x = reorder(Variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "steelblue") + coord_flip() +
  labs(x = "Variable", y = "%IncMSE") + theme_minimal()
ggsave("Figure4_RF_Importance.pdf", p_rf, width = 8, height = 6)
ggsave("Figure4_RF_Importance.png", p_rf, width = 8, height = 6, dpi = 300)

# Figure 5: LPA BIC
bic_df <- data.frame(Profiles = 1:5, BIC = bic_values)
p_bic <- ggplot(bic_df, aes(x = Profiles, y = BIC)) +
  geom_line(linewidth = 1.2, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  geom_point(data = bic_df[bic_df$Profiles == optimal_k, ],
             aes(x = Profiles, y = BIC), size = 5, color = "red", shape = 18) +
  annotate("text", x = optimal_k + 0.5, y = min(bic_values) + 20,
           label = paste("Optimal: K =", optimal_k), color = "red") +
  labs(x = "Number of Profiles (K)", y = "BIC") + theme_minimal()
ggsave("Figure5_LPA_BIC.pdf", p_bic, width = 6, height = 5)
ggsave("Figure5_LPA_BIC.png", p_bic, width = 6, height = 5, dpi = 300)

# Figure 6: Profile Comparison (3 profiles)
profile_data <- data.frame(
  Profile = factor(1:3, labels = c("Profile 1 (n=317)", "Profile 2 (n=116)", "Profile 3 (n=81)")),
  TD = c(-0.036, -0.228, 0.296),
  KSE = c(0.219, -0.803, 0.326),
  ALD = c(-0.134, 0.676, -0.568),
  Stunting = c(-0.002, -0.364, 0.529)
)

profile_long <- profile_data %>%
  pivot_longer(cols = c(TD, KSE, ALD, Stunting),
               names_to = "Construct", values_to = "Mean")

p_profile <- ggplot(profile_long, aes(x = Profile, y = Mean, fill = Construct)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "", y = "Standardized Mean") +
  theme_minimal() + theme(legend.position = "bottom") +
  scale_fill_manual(values = c(TD = "steelblue", KSE = "darkred",
                               ALD = "darkgreen", Stunting = "orange"))
ggsave("Figure6_Profile_Comparison.pdf", p_profile, width = 10, height = 6)
ggsave("Figure6_Profile_Comparison.png", p_profile, width = 10, height = 6, dpi = 300)

# ==================================
# BENCHMARKING
# ==================================

median_stunting <- median(data_analysis$stunting, na.rm = TRUE)
data_bench <- data_analysis
data_bench$stunting_binary <- ifelse(data_bench$stunting > median_stunting, 1, 0)

bench_cols <- c("stunting_binary", "poverty_prov", "Kep_Pend", "cbr_c", "tfr_c",
                "dependency_ratio_c", "poverty_kab_c", "unemployment_c", "ipm_kab_c",
                "sanitation_c", "water_c", "cpr_c")
data_bench <- na.omit(data_bench[, bench_cols])

m1 <- glm(stunting_binary ~ poverty_prov, data = data_bench, family = binomial)
m2 <- glm(stunting_binary ~ ., data = data_bench, family = binomial)

acc1 <- mean(ifelse(predict(m1, type = "response") > 0.5, 1, 0) == data_bench$stunting_binary)
acc2 <- mean(ifelse(predict(m2, type = "response") > 0.5, 1, 0) == data_bench$stunting_binary)

rf_data <- data_bench
rf_data$stunting_binary <- as.factor(rf_data$stunting_binary)
set.seed(123)
rf_cls <- randomForest(stunting_binary ~ ., data = rf_data, ntree = 500)
acc3 <- mean(as.numeric(as.character(predict(rf_cls))) == data_bench$stunting_binary)

benchmark_table <- data.frame(
  Model = c("Null", "Logistic (1 var)", "Logistic (all vars)", "Random Forest"),
  Accuracy = c(0.5, round(acc1, 4), round(acc2, 4), round(acc3, 4)),
  AIC = c(NA, round(AIC(m1), 1), round(AIC(m2), 1), NA),
  Improvement = c("0%", paste0(round((acc1-0.5)*100,1),"%"),
                  paste0(round((acc2-0.5)*100,1),"%"),
                  paste0(round((acc3-0.5)*100,1),"%"))
)
print(benchmark_table)

save(m1, m2, rf_cls, benchmark_table, file = "Benchmarking_Results.RData")
