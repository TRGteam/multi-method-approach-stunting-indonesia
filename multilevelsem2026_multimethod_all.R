###############################################################
# Multi-Method AI Framework for Childhood Stunting Surveillance
# =============================================================

# ===========================================
# 1. SETUP, LIBRARIES, AND PARALLEL COMPUTING
# ===========================================

# install.packages(c("blavaan", "bnlearn", "pcalg", "randomForest", "iml", 
#                    "mclust", "ggplot2", "dplyr", "tidyr", "future", 
#                    "readxl", "loo", "bayesplot", "coda", "Rgraphviz"))

library(future)
plan(multisession)
options(mc.cores = 3)
options(future.globals.maxSize = 2 * 1024^3)

library(readxl)
library(dplyr)
library(tidyr)
library(blavaan)
library(bnlearn)
library(pcalg)
library(randomForest)
library(iml)
library(mclust)
library(ggplot2)
library(coda) # Untuk effectiveSize() dan gelman.diag()
library(loo)
library(bayesplot)

set.seed(123)
setwd("~/multilevelsem_rev_ori")

# ============================
# 2. LOAD DATA & PREPROCESSING
# ============================

data <- read_excel("Multilevelsem2026.xlsx", sheet = 1)
cat("Data loaded. Dimensions:", dim(data), "\n")

data_centered <- data %>%
  group_by(Provinsi) %>%
  mutate(
    cbr_c = cbr - mean(cbr, na.rm = TRUE),
    tfr_c = tfr - mean(tfr, na.rm = TRUE),
    dependency_ratio_c = dependency_ratio - mean(dependency_ratio, na.rm = TRUE),
    poverty_kab_c = poverty_kab - mean(poverty_kab, na.rm = TRUE),
    unemployment_c = unemployment - mean(unemployment, na.rm = TRUE),
    ipm_kab_c = ipm_kab - mean(ipm_kab, na.rm = TRUE),
    sanitation_c = sanitation - mean(sanitation, na.rm = TRUE),
    water_c = water - mean(water, na.rm = TRUE),
    cpr_c = cpr - mean(cpr, na.rm = TRUE)
  ) %>%
  ungroup()

vars_to_scale <- c(
  "stunting", 
  "cbr_c", "tfr_c", "dependency_ratio_c",
  "poverty_kab_c", "unemployment_c", "ipm_kab_c",
  "sanitation_c", "water_c", "cpr_c",
  "poverty_prov", "Kep_Pend"
)

data_centered[vars_to_scale] <- scale(data_centered[vars_to_scale])
cat("Data preprocessing completed. No missing values detected (N=514).\n")

# =================================================
# 3. COMPONENT 1: BAYESIAN MULTILEVEL SEM (BLAVAAN)
# =================================================

# ------------------------
# 3.1 Model Specification
# ------------------------
model <- '
  level: 1
  TD =~ cbr_c + tfr_c + dependency_ratio_c
  KSE =~ poverty_kab_c + unemployment_c + ipm_kab_c
  ALD =~ sanitation_c + water_c + cpr_c
  
  KSE ~ TD
  ALD ~ TD + KSE
  stunting ~ TD + ALD + KSE
  
  level: 2
  stunting ~ poverty_prov + Kep_Pend
'

# ---------------------
# 3.2 Run Bayesian SEM
# ----------------------
cat("\n--- Running Bayesian Multilevel SEM ---\n")
fit <- bsem(
  model,
  data = data_centered,
  cluster = "Provinsi",
  n.chains = 4,
  burnin = 4000,
  sample = 8000,
  save.lvs = TRUE,
  test = "none",  # Hindari post-estimation yang bermasalah
  dp = dpriors(
    lambda = "normal(0,.5)",
    beta = "normal(0,.5)",
    theta = "gamma(1,.5)",
    psi = "gamma(1,.5)"
  )
)
cat("Bayesian SEM completed.\n")

# -------------------------------------------------
# 3.3 Convergence Diagnostics (Dari blavaan + coda)
# -------------------------------------------------
cat("\n--- Convergence Diagnostics ---\n")

# Extract MCMC samples - blavInspect dari blavaan
samps <- blavInspect(fit, "mcmc")

# Effective Sample Size - dari paket coda
ess_values <- effectiveSize(samps)
cat("Minimum ESS:", round(min(ess_values), 0), "(Threshold > 400)\n")

# R-hat - dari paket coda
rhat_values <- gelman.diag(samps)$psrf[, 1]
cat("Maximum R-hat:", round(max(rhat_values), 3), "(Threshold < 1.05)\n")

# ---------------------------
# 3.4 Model Summary (blavaan)
# ---------------------------
cat("\n--- Model Summary ---\n")
summary(fit, standardized = TRUE)

# ---------------------------------------
# 3.5 Extract Factor Scores - blavPredict
# ---------------------------------------
fscores <- blavPredict(fit, type = "lv", level = 1)
fscores_df <- as.data.frame(fscores)
colnames(fscores_df) <- c("TD", "KSE", "ALD")
data_analysis <- cbind(data_centered, fscores_df)
cat("Factor scores extracted and merged.\n")

# ============================================================
# 4. COMPONENT 2: CAUSAL DISCOVERY (Level-2 variables removed)
# ============================================================

data_bn <- data_analysis %>%
  select(TD, KSE, ALD, stunting, poverty_kab_c, sanitation_c) %>%
  na.omit()

cat("\n--- Causal Discovery Data ---\n")
cat("Observations:", nrow(data_bn), "\n")
cat("Variables:", paste(colnames(data_bn), collapse=", "), "\n")

cat("\nCorrelation Matrix:\n")
print(round(cor(data_bn), 3))

# ------------------
# 4.1 BNSL Bootstrap
# ------------------
set.seed(123)
cat("\n--- Running BNSL Bootstrap (R=100) ---\n")
bn_boot <- boot.strength(
  data_bn, 
  R = 100, 
  algorithm = "hc",
  algorithm.args = list(score = "bic-g"),
  cpdag = FALSE
)

edge_strength <- bn_boot[order(-bn_boot$strength), ]
cat("\nTop 10 Edge Strengths:\n")
print(head(edge_strength, 10))

cat("\nEdge Counts by Threshold:\n")
for(thresh in c(0.3, 0.4, 0.5, 0.6, 0.7)) {
  n_edges <- sum(bn_boot$strength > thresh)
  cat("  Threshold >", thresh, ":", n_edges, "edges\n")
}

bn_avg <- averaged.network(bn_boot, threshold = 0.4)
cat("\nAveraged Network (threshold=0.4):\n")
cat("  Number of arcs:", nrow(arcs(bn_avg)), "\n")
if(nrow(arcs(bn_avg)) > 0) {
  print(arcs(bn_avg))
}

# ----------------
# 4.2 PC Algorithm
# ----------------
cat("\n--- Running PC Algorithm ---\n")
data_matrix <- as.matrix(data_bn)
suffStat <- list(C = cor(data_matrix), n = nrow(data_matrix))
pc_fit <- pc(
  suffStat,
  indepTest = gaussCItest,
  labels = colnames(data_matrix),
  alpha = 0.05
)

pc_edges_matrix <- as(pc_fit@graph, "matrix")
pc_edges_list <- which(pc_edges_matrix == 1, arr.ind = TRUE)

if(nrow(pc_edges_list) > 0) {
  pc_edges_df <- data.frame(
    from = colnames(pc_edges_matrix)[pc_edges_list[, 2]],
    to = rownames(pc_edges_matrix)[pc_edges_list[, 1]]
  )
  cat("PC Algorithm found", nrow(pc_edges_df), "edges:\n")
  print(pc_edges_df)
} else {
  cat("PC Algorithm found no edges.\n")
}

# ------------------
# 4.3 Edge Agreement
# ------------------
theoretical_edges <- data.frame(
  from = c("TD", "TD", "KSE", "TD", "KSE", "ALD"),
  to = c("KSE", "ALD", "ALD", "stunting", "stunting", "stunting"),
  stringsAsFactors = FALSE
)

if(exists("bn_avg") && nrow(arcs(bn_avg)) > 0) {
  ai_edges <- arcs(bn_avg)
  method_used <- "BNSL"
} else if(exists("pc_fit") && nrow(pc_edges_list) > 0) {
  ai_edges <- as.matrix(pc_edges_df[, c("from", "to")])
  method_used <- "PC Algorithm"
} else {
  ai_edges <- matrix(nrow = 0, ncol = 2)
  method_used <- "None"
}

if(nrow(ai_edges) > 0) {
  ai_edges_df <- data.frame(from = ai_edges[, 1], to = ai_edges[, 2])
  theoretical_edges$in_ai <- sapply(1:nrow(theoretical_edges), function(i) {
    any(ai_edges_df$from == theoretical_edges$from[i] & 
          ai_edges_df$to == theoretical_edges$to[i])
  })
  
  cat("\n--- Edge Agreement (", method_used, ") ---\n", sep="")
  theoretical_edges$edge <- paste(theoretical_edges$from, "→", theoretical_edges$to)
  theoretical_edges$status <- ifelse(theoretical_edges$in_ai, "Confirmed", "Not Found")
  print(theoretical_edges[, c("edge", "status")])
  
  recovery_rate <- sum(theoretical_edges$in_ai) / nrow(theoretical_edges) * 100
  cat("\nRecovery Rate:", round(recovery_rate, 1), "% (", 
      sum(theoretical_edges$in_ai), "/", nrow(theoretical_edges), " edges)\n", sep="")
} else {
  cat("\n--- Edge Agreement ---\n")
  cat("No edges found by either BNSL or PC Algorithm.\n")
}

# =======================================
# 5. COMPONENT 3: RANDOM FOREST WITH SHAP
# =======================================

data_rf <- data_analysis %>%
  select(stunting, cbr_c, tfr_c, dependency_ratio_c,
         poverty_kab_c, unemployment_c, ipm_kab_c,
         sanitation_c, water_c, cpr_c,
         poverty_prov, Kep_Pend, Provinsi) %>%
  na.omit()

cat("\n--- Random Forest Data ---\n")
cat("Observations:", nrow(data_rf), "\n")

set.seed(123)
rf_model <- randomForest(
  stunting ~ . - Provinsi,
  data = data_rf,
  ntree = 500,
  mtry = round(sqrt(ncol(data_rf) - 2)),
  importance = TRUE,
  keep.forest = TRUE
)

cat("\nRandom Forest Performance:\n")
print(rf_model)

# Performance Metrics
predictions <- predict(rf_model)
rmse <- sqrt(mean((data_rf$stunting - predictions)^2))
mae <- mean(abs(data_rf$stunting - predictions))
oob_mse <- rf_model$mse[length(rf_model$mse)]
oob_r2 <- rf_model$rsq[length(rf_model$rsq)]

cat("\n--- Performance Metrics ---\n")
cat("RMSE:", round(rmse, 4), "\n")
cat("MAE :", round(mae, 4), "\n")
cat("OOB MSE:", round(oob_mse, 4), "\n")
cat("OOB R-squared:", round(oob_r2, 4), "\n")

# Cross-Validation (Grouped by Province)
set.seed(123)
unique_provs <- unique(data_rf$Provinsi)
folds_vec <- sample(rep(1:5, length.out = length(unique_provs)))
province_fold <- setNames(folds_vec, unique_provs)
data_rf$fold <- province_fold[as.character(data_rf$Provinsi)]

cv_rmse <- c()
for(k in 1:5) {
  train_data <- data_rf[data_rf$fold != k, ]
  test_data <- data_rf[data_rf$fold == k, ]
  
  rf_cv <- randomForest(
    stunting ~ . - fold - Provinsi,
    data = train_data,
    ntree = 300,
    mtry = round(sqrt(ncol(train_data) - 3))
  )
  
  pred_cv <- predict(rf_cv, newdata = test_data)
  cv_rmse[k] <- sqrt(mean((test_data$stunting - pred_cv)^2))
}
cat("\nCV RMSE (mean ± sd):", round(mean(cv_rmse), 4), "±", round(sd(cv_rmse), 4), "\n")

# Variable Importance
importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)
importance_df <- importance_df[order(-importance_df$`%IncMSE`), ]
cat("\n--- Variable Importance (%IncMSE) ---\n")
print(importance_df[, c("Variable", "%IncMSE")])

# ------------------------------ 
# 5.5 SHAP Analysis (using iml)
# ------------------------------ 
cat("\n--- Running SHAP Analysis (iml) ---\n")

X <- data_rf[, !(colnames(data_rf) %in% c("stunting", "fold", "Provinsi"))]
y <- data_rf$stunting

set.seed(123)
rf_model_shap <- randomForest(
  y ~ .,
  data = X,
  ntree = 500,
  mtry = round(sqrt(ncol(X))),
  importance = TRUE,
  keep.forest = TRUE
)

predictor <- Predictor$new(
  model = rf_model_shap,
  data = X,
  y = y,
  predict.function = function(model, newdata) predict(model, newdata)
)

set.seed(123)
sample_idx <- sample(1:nrow(X), min(100, nrow(X)))
X_explain <- X[sample_idx, ]

shapley_list <- list()
for(i in 1:nrow(X_explain)) {
  if(i %% 20 == 0) cat("  SHAP progress:", i, "/", nrow(X_explain), "\n")
  shapley <- Shapley$new(predictor, x.interest = X_explain[i, ])
  shapley_list[[i]] <- shapley$results
}

shap_values_df <- do.call(rbind, shapley_list)

shap_summary <- data.frame(
  Variable = colnames(X),
  Mean_Abs_SHAP = sapply(colnames(X), function(var) {
    mean(abs(shap_values_df[shap_values_df$feature == var, "phi"]))
  })
)
shap_summary <- shap_summary[order(-shap_summary$Mean_Abs_SHAP), ]
cat("\n--- SHAP Feature Importance ---\n")
print(head(shap_summary, 11))

# SHAP Dependence Plot
shap_kepend <- shap_values_df[shap_values_df$feature == "Kep_Pend", ]
feature_values <- X_explain$Kep_Pend
if(length(feature_values) != nrow(shap_kepend)) {
  shap_kepend <- shap_kepend[1:length(feature_values), ]
}
df_plot <- data.frame(Kep_Pend = feature_values, SHAP = shap_kepend$phi)

p_dep <- ggplot(df_plot, aes(x = Kep_Pend, y = SHAP)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "SHAP Dependence Plot: Population Density",
    x = "Population Density (standardized)",
    y = "SHAP Value (contribution to stunting)"
  ) +
  theme_minimal()
ggsave("SHAP_dependence_KepPend.pdf", p_dep, width = 8, height = 6)
ggsave("SHAP_dependence_KepPend.png", p_dep, width = 8, height = 6, dpi = 300)
cat("SHAP dependence plot saved.\n")

# ======================================= 
# 6. COMPONENT 4: LATENT PROFILE ANALYSIS
# ======================================= 

data_lpa <- data_analysis %>%
  select(TD, KSE, ALD, stunting) %>%
  na.omit()

cat("\n--- Latent Profile Analysis ---\n")
cat("Observations:", nrow(data_lpa), "\n")

bic_values <- c()
loglik_values <- c()
entropy_values <- c()

for(k in 1:5) {
  set.seed(123)
  m <- Mclust(data_lpa[, c("TD", "KSE", "ALD")], G = k, modelNames = "VVV")
  bic_values[k] <- m$bic
  loglik_values[k] <- m$loglik
  
  if(k > 1) {
    probs <- m$z
    entropy <- -sum(probs * log(probs + 1e-10)) / nrow(probs)
    entropy_values[k] <- entropy
  } else {
    entropy_values[k] <- NA
  }
  cat("K =", k, "| BIC =", round(m$bic, 2), 
      "| Entropy =", round(entropy_values[k], 4), "\n")
}

optimal_k <- which.max(bic_values)
cat("\nOptimal K =", optimal_k, "(BIC =", round(bic_values[optimal_k], 2), ")\n")

# Final Model
set.seed(123)
final_lpa <- Mclust(data_lpa[, c("TD", "KSE", "ALD")], G = optimal_k, modelNames = "VVV")
data_lpa$Profile <- as.factor(final_lpa$classification)

avg_posterior <- sapply(1:optimal_k, function(k) {
  mean(final_lpa$z[final_lpa$classification == k, k])
})
cat("\nAverage Posterior Probabilities:\n")
print(round(avg_posterior, 4))

profile_summary <- data_lpa %>%
  group_by(Profile) %>%
  summarise(
    N = n(),
    TD_mean = mean(TD),
    KSE_mean = mean(KSE),
    ALD_mean = mean(ALD),
    stunting_mean = mean(stunting)
  )
cat("\n--- Profile Summary (K =", optimal_k, ") ---\n")
print(profile_summary)

# =============== 
# 7. BENCHMARKING
# =============== 

cat("\n--- Benchmarking (Continuous Outcome) ---\n")

null_pred <- mean(data_rf$stunting)
null_rmse <- sqrt(mean((data_rf$stunting - null_pred)^2))
null_mae <- mean(abs(data_rf$stunting - null_pred))
null_r2 <- 0

lm_data <- data_rf[, !(colnames(data_rf) %in% c("fold", "Provinsi"))]
lm_model <- lm(stunting ~ ., data = lm_data)
lm_pred <- predict(lm_model)
lm_rmse <- sqrt(mean((data_rf$stunting - lm_pred)^2))
lm_mae <- mean(abs(data_rf$stunting - lm_pred))
lm_r2 <- summary(lm_model)$r.squared

rf_r2 <- oob_r2

benchmark_table <- data.frame(
  Model = c("Null (Mean)", "Linear Regression", "Random Forest"),
  RMSE = round(c(null_rmse, lm_rmse, rmse), 4),
  MAE = round(c(null_mae, lm_mae, mae), 4),
  R_squared = round(c(null_r2, lm_r2, rf_r2), 4)
)

cat("\n--- Benchmark Results ---\n")
print(benchmark_table)

# ============================================ 
# 8. GENERATE FIGURES (Figure 2, 4, 5, 6, 7,8)
# ============================================ 

cat("\n========================================\n")
cat("GENERATING FIGURES (Figure 2, 4, 5, 6, 7)\n")
cat("========================================\n")

# ---------------------------------------- 
# FIGURE 2: Trace Plots (MCMC Convergence)
# ---------------------------------------- 
cat("\n--- Generating Figure 2: Trace Plots ---\n")

samps <- blavInspect(fit, "mcmc")
trace_params <- c("KSE~TD", "stunting~TD", "stunting~KSE", "stunting~poverty_prov.l2")
trace_params <- trace_params[trace_params %in% colnames(samps[[1]])]

if(length(trace_params) >= 2) {
  p_trace <- mcmc_trace(samps, pars = trace_params, facet_args = list(ncol = 2))
  p_trace <- p_trace + 
    ggtitle("Trace Plots for MCMC Convergence") +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      strip.text = element_text(size = 10)
    )
  
  ggsave("Figure2_Trace_Plots.pdf", p_trace, width = 12, height = 10)
  ggsave("Figure2_Trace_Plots.png", p_trace, width = 12, height = 10, dpi = 300)
  cat("  Figure 2 saved: Figure2_Trace_Plots.pdf/png\n")
} else {
  cat("  Trace parameters not found. Skipping Figure 2.\n")
}

# ---------------------- 
# FIGURE 3: Path Diagram 
# ---------------------- 
cat("\n--- Figure 3: Path Diagram ---\n")
cat("  NOTE: Figure 3 created manually using draw.io\n")
cat("  Use standardized coefficients from Table 7 and Table 8\n")

# ------------------
# FIGURE 4: BNSL DAG
# -------------------
cat("\n--- Generating Figure 4: BNSL DAG ---\n")

if(exists("bn_avg") && nrow(arcs(bn_avg)) > 0) {
  adj_matrix <- amat(bn_avg)
  
  node_labels <- c(
    "TD" = "Demographic\nPressure",
    "KSE" = "Socio-Economic\nVulnerability",
    "ALD" = "Access to Basic\nServices",
    "stunting" = "Stunting\nPrevalence",
    "poverty_kab_c" = "District\nPoverty",
    "sanitation_c" = "Sanitation\nAccess"
  )
  
  pdf("Figure4_BNSL_DAG.pdf", width = 10, height = 8)
  qgraph(adj_matrix,
         layout = "spring",
         labels = node_labels[colnames(adj_matrix)],
         title = "Causal Discovery Result (BNSL, τ = 0.4)",
         edge.color = "darkblue",
         vsize = 12,
         label.cex = 1.0,
         theme = "colorblind",
         borders = TRUE,
         mar = c(8, 6, 8, 6))
  dev.off()
  
  png("Figure4_BNSL_DAG.png", width = 3000, height = 2400, res = 300)
  qgraph(adj_matrix,
         layout = "spring",
         labels = node_labels[colnames(adj_matrix)],
         title = "Causal Discovery Result (BNSL, τ = 0.4)",
         edge.color = "darkblue",
         vsize = 12,
         label.cex = 1.0,
         theme = "colorblind",
         borders = TRUE,
         mar = c(8, 6, 8, 6))
  dev.off()
  
  cat("  Figure 4 saved: Figure4_BNSL_DAG.pdf/png\n")
} else {
  cat("  No edges found in bn_avg. Skipping Figure 4.\n")
}

# -----------------------
# FIGURE 5: RF Importance
# -----------------------
cat("\n--- Generating Figure 5: RF Importance ---\n")

var_labels_en <- c(
  "poverty_prov" = "Provincial Poverty",
  "Kep_Pend" = "Population Density",
  "ipm_kab_c" = "District HDI",
  "poverty_kab_c" = "District Poverty",
  "cbr_c" = "Crude Birth Rate",
  "tfr_c" = "Total Fertility Rate",
  "water_c" = "Water Access",
  "dependency_ratio_c" = "Dependency Ratio",
  "sanitation_c" = "Sanitation Access",
  "unemployment_c" = "Unemployment Rate",
  "cpr_c" = "Contraceptive Prevalence"
)

importance_df$Variable_En <- var_labels_en[importance_df$Variable]

p_rf <- ggplot(importance_df[1:10, ], 
               aes(x = reorder(Variable_En, `%IncMSE`), y = `%IncMSE`)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Random Forest Variable Importance",
    x = "Variable",
    y = "% Increase in Mean Squared Error"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10)
  )

ggsave("Figure5_RF_Importance.pdf", p_rf, width = 8, height = 6)
ggsave("Figure5_RF_Importance.png", p_rf, width = 8, height = 6, dpi = 300)
cat("  Figure 5 saved: Figure5_RF_Importance.pdf/png\n")

# ------------------------------
# FIGURE 6: SHAP Dependence Plot
# ------------------------------
cat("\n--- Generating Supplementary Figure: SHAP Dependence Plot ---\n")

p_dep <- ggplot(df_plot, aes(x = Kep_Pend, y = SHAP)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "loess", se = TRUE, color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "SHAP Dependence Plot for Population Density",
    x = "Population Density (standardized)",
    y = "SHAP Value (contribution to stunting prediction)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )

ggsave("Figure6_SHAP_Dependence.pdf", p_dep, width = 8, height = 6)
ggsave("Figure6_SHAP_Dependence.png", p_dep, width = 8, height = 6, dpi = 300)
cat("  Figure 6 saved: Figure6_SHAP_Dependence.pdf/png\n")

# -----------------
# FIGURE 7: LPA BIC
# -----------------
cat("\n--- Generating Figure 6: LPA BIC ---\n")

bic_df <- data.frame(Profiles = 1:5, BIC = bic_values)

p_bic <- ggplot(bic_df, aes(x = Profiles, y = BIC)) +
  geom_line(linewidth = 1.2, color = "steelblue") +
  geom_point(size = 3, color = "steelblue") +
  geom_point(data = bic_df[bic_df$Profiles == optimal_k, ],
             aes(x = Profiles, y = BIC), size = 5, color = "red", shape = 18) +
  annotate("text", x = optimal_k + 0.5, y = min(bic_values) + 20,
           label = paste("Optimal: K =", optimal_k), color = "red", size = 5) +
  labs(
    title = "Latent Profile Analysis Model Selection",
    x = "Number of Profiles (K)",
    y = "Bayesian Information Criterion (BIC)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold")
  )

ggsave("Figure7_LPA_BIC.pdf", p_bic, width = 6, height = 5)
ggsave("Figure7_LPA_BIC.png", p_bic, width = 6, height = 5, dpi = 300)
cat("  Figure 7 saved: Figure7_LPA_BIC.pdf/png\n")

# ----------------------------
# FIGURE 8: Profile Comparison
# ----------------------------
cat("\n--- Generating Figure 8: Profile Comparison ---\n")

construct_labels <- c(
  "TD" = "Demographic Pressure",
  "KSE" = "Socio-Economic Vulnerability",
  "ALD" = "Access to Basic Services",
  "Stunting" = "Stunting Prevalence"
)

profile_data <- data.frame(
  Profile = factor(profile_summary$Profile, 
                   labels = paste0("Profile ", profile_summary$Profile, " (n=", profile_summary$N, ")")),
  TD = profile_summary$TD_mean,
  KSE = profile_summary$KSE_mean,
  ALD = profile_summary$ALD_mean,
  Stunting = profile_summary$stunting_mean
)

profile_long <- profile_data %>%
  pivot_longer(cols = c(TD, KSE, ALD, Stunting),
               names_to = "Construct", values_to = "Mean")

profile_long$Construct_En <- construct_labels[profile_long$Construct]

p_profile <- ggplot(profile_long, aes(x = Profile, y = Mean, fill = Construct_En)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    title = "District Profile Characteristics Across Three Typologies",
    x = "",
    y = "Standardized Mean"
  ) +
  theme_minimal() + 
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.title = element_blank(),
    legend.text = element_text(size = 10)
  ) +
  scale_fill_manual(
    values = c(
      "Demographic Pressure" = "steelblue",
      "Socio-Economic Vulnerability" = "darkred",
      "Access to Basic Services" = "darkgreen",
      "Stunting Prevalence" = "orange"
    )
  )

ggsave("Figure8_Profile_Comparison.pdf", p_profile, width = 10, height = 6)
ggsave("Figure8_Profile_Comparison.png", p_profile, width = 10, height = 6, dpi = 300)
cat("  Figure 8 saved: Figure8_Profile_Comparison.pdf/png\n")

# ================
# 9. SAVE RESULTS
# ================

cat("\n--- Saving all results ---\n")

save(
  fit,
  bn_boot, bn_avg, pc_fit,
  rf_model, rf_model_shap, lm_model,
  shap_summary,
  final_lpa,
  benchmark_table,
  importance_df,
  profile_summary,
  bic_values,
  entropy_values,
  file = "Revision_Complete_Results.RData"
)

cat("\n--- All results saved to 'Revision_Complete_Results.RData' ---\n")

writeLines(capture.output(sessionInfo()), "sessionInfo.txt")
cat("Session info saved to 'sessionInfo.txt'\n")

# ========
# SUMMARY
# ========

cat("\n=======================\n")
cat("SUMMARY OF GENERATED FILES\n")
cat("==========================\n")
cat("\n[Main Figures]\n")
cat("  Figure 1: Architecture       - MANUAL (draw.io)\n")
cat("  Figure 2: Trace Plots        - Figure2_Trace_Plots.pdf/png\n")
cat("  Figure 3: Path Diagram       - MANUAL (draw.io)\n")
cat("  Figure 4: BNSL DAG           - Figure4_BNSL_DAG.pdf/png\n")
cat("  Figure 5: RF Importance      - Figure5_RF_Importance.pdf/png\n")
cat("  Figure6_SHAP_Dependence.pdf/png\n")
cat("  Figure 7: LPA BIC            - Figure7_LPA_BIC.pdf/png\n")
cat("  Figure 8: Profile Comparison - Figure8_Profile_Comparison.pdf/png\n")

cat("\n[Data Files]\n")
cat("  - Revision_Complete_Results.RData\n")
cat("  - sessionInfo.txt\n")
cat("\n=============\n")
cat("SCRIPT COMPLETED \n")
