# Interaction model

##
library(tidyverse)
library(lme4)
library(emmeans)
library(lmerTest)
library(gridExtra)
library(MASS)
library(ggplot2)
library(performance)
library(texreg) ## for latex-table, otherwise not needed
library(dplyr)
library(sjPlot)
library(cowplot)
library(xtable)
library(ggpubr)
library(parameters)

###############
## Load data ##
###############

# This is search time data
df <- read.csv("output/data/vsgui10k_search_times_with_factors.csv", header = T)

df$cue <- relevel(factor(df$cue), ref = "Image")   # Change baselines here to check for alternatives
df$category <- relevel(factor(df$category), ref = "Web")


# Tweak column names for clarity
colnames(df)[which(names(df) == "m6_0")] <- "ContourCongestion"
colnames(df)[which(names(df) == "m5_0")] <- "FigureGroundContrast"

path = "output/figs/3-interaction"
dir.create(path, showWarnings = FALSE)

####################
## Model fitting ##
####################

## Interaction model (in main paper)
fit_interaction <- lmer(search_time ~
                        Factor1 * absent+
                        Factor3 * absent +
                        Factor4 *absent +
                        ContourCongestion *absent+
                        category* absent +
                        cue * absent +
                        (1 | pid) +
                        (1 | img_name),
                        data = df)

## Interaction model with fixations as DV (in supplement)
fit_interaction_fixations <- lmer(n_fixations ~
                          Factor1 * absent+
                          Factor3 * absent +
                          Factor4 *absent +
                          ContourCongestion *absent+
                          category* absent +
                          cue * absent +
                          (1 | pid) +
                          (1 | img_name),
                        data = df)

## Interaction model (gamma link, in supplement)
fit_interaction_glmer <- glmer(search_time ~
                        Factor1 *absent +
                        Factor3 * absent +
                        Factor4 *absent+
                        ContourCongestion *absent +
                        category *absent+
                        cue *absent+
                        (1 | pid) +
                        (1 | img_name),
                        family = Gamma(link = "log"),
                        data = df) 


##########
## Plot ##
##########

set_theme(
  base = theme_bw(),
  theme.font = "sans",
  axis.title.color = "black",
  geom.label.color = "black",
  axis.textcolor.x = "black",
  axis.textcolor.y = "black",
  axis.textsize.x = 1.2,
  axis.textsize.y = 1.2,
  axis.title.size = 1.3,
)

# Set labels manually
interaction_labels = c("Target absent × Category=Mobile", 
                      "Target absent × Category=Desktop", 
                      "Category=Mobile", 
                      "Factor 3 (Color variability 1,3)", 
                      "Target absent × Factor 3 (Color variability 1,3)", 
                      "Category=Desktop",
                      "Factor 4 (Grid quality)",
                      "Contour congestion",
                      "Factor 1 (Visual clutter)",
                      "Target absent × Factor 1 (Visual clutter)",
                      "Target absent x Cue=Text+color",
                      "Cue=Text",
                      "Target absent × Factor 4 (Grid quality)",
                      "Target absent × Contour congestion",
                      "Cue=Text+color",
                      "Target absent × Cue=Text",
                      "Target absent")

interaction_order = c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)

interaction_groups =           c("A", "A", "C", "A", "A", "C", "B", "A", "A", "A", "C", "A", "A", "B","B", "A", "C")

interaction_groups_fixations = c("A", "A", "C", "A", "A", "C", "B", "A", "A", "A", "C", "A", "A", "B", "B", "A", "A")
interaction_groups_glmer =     c("D", "C", "C", "C", "C", "C", "D", "C", "D", "D", "D", "C", "D", "D", "C", "D", "D")


interaction_order = c(2,16,9,13,12,8,17,10,1,5,4,6,11,3,7,14,15)

set_order = c(2, 10, 6, 5, 7, 11, 1, 3, 8, 4, 9)
set_labels = c("Target absent × Category=Mobile", 
               "Category=Mobile", 
               "Target absent × Category=Desktop", 
               "Category=Desktop",
               "Set size",
               "Target absent x Cue=Text+color",
               "Target absent × Set size",
                "Cue=Text",
                "Cue=Text+color",
                "Target absent × Cue=Text",
                "Target absent")
set_groups = c("A", "A", "B", "B", "A", "A", "A", "B","B", "A","C")

# Factor1 * absent -> 1,2
# Factor3 * absent -> 3,4
# Factor4 * absent -> 5,6
# m6_0 * absent -> 7,8
# category* absent -> 9,10, 11, 12
# cue * absent + -> 13,14,15,16


group.colors <- c(A = "#000080", B = "#a40000", C ="azure4", D="black")

pforest <- plot_model(fit_interaction, type="est", title="", axis.title="Estimates (s)", show.values=TRUE, show.p=TRUE, show.ci=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-3,6), order.terms = interaction_order, axis.labels=interaction_labels, group.terms=interaction_groups, colors = group.colors,  vline.color = "#696969")
pforest_std <- plot_model(fit_interaction, type="std", title="", axis.title="Estimates (standardized)", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-3,2), order.terms = interaction_order, axis.labels = interaction_labels, group.terms=interaction_groups, colors = group.colors,  vline.color = "#696969") 

# Add a custom legend
pforest <- pforest + 
  scale_color_manual(name = "", 
                     values =c("#a40000", "#000080", "azure4"), 
                     labels = c("Increase", "Decrease", "Not significant")) +
  guides(color = guide_legend(ncol=2)) +
  theme(legend.position = "bottom")

# Add a custom legend
pforest_std <- pforest_std + 
  scale_color_manual(name = "", 
                     values =c("#a40000", "#000080", "azure4"), 
                     labels = c("Increase", "Decrease", "Not significant")) +
  guides(color = guide_legend(ncol=2)) +
  theme(legend.position = "bottom")

# Save
path = "output/figs/3-interaction/fig9.pdf"
pdf(path, width=10, height=9)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

# Plot for fixations
pforest <- plot_model(fit_interaction_fixations, type="est", title="", axis.title="Estimates (fixations)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-10,22), order.terms = interaction_order, axis.labels=interaction_labels, group.terms=interaction_groups_fixations,  vline.color = "#696969")

pforest_std <- plot_model(fit_interaction_fixations, type="std", title="", axis.title="Estimates (standardized)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), axis.lim = c(-2,2),  order.terms = interaction_order, axis.labels = interaction_labels ,group.terms=interaction_groups_fixations,  vline.color = "#696969")

# Save
path = "output/figs/3-interaction/supp_fig15.pdf"
pdf(path, width=10, height=9)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

# Plot GLMER
pforest <- plot_model(fit_interaction_glmer, type="est", title="", axis.title="Estimates", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35), order.terms = interaction_order, axis.labels=interaction_labels, group.terms=interaction_groups_glmer,  transform=NULL)

pforest_std <- plot_model(fit_interaction_glmer, type="est", title="", axis.title="Estimates (transformed)", colors="black", show.values=TRUE, show.p=TRUE, value.offset=c(0.35, 0.35),transform="exp",order.terms = interaction_order, axis.labels = interaction_labels ,group.terms=interaction_groups_glmer)

# Save
path = "output/figs/3-interaction/supp_fig16.pdf"
pdf(path, width=10, height=9)
cowplot::plot_grid(plotlist = list(pforest, pforest_std), ncol=2, labels=c("A","B"))
dev.off()

####################
## Data for plots ##
####################

# These are used to plot Fig. 9 in main paper (also graphical abstract)

print(get_model_data(fit_interaction, type="pred", terms = c("Factor1", "absent")), n = Inf)
print(get_model_data(fit_interaction, type="pred", terms = c("Factor3", "absent")), n = Inf)
print(get_model_data(fit_interaction, type="pred", terms = c("Factor4", "absent")), n = Inf)
print(get_model_data(fit_interaction, type="pred", terms = c("ContourCongestion[0, 0.25, 0.5, 0.75, 1]", "absent")), n = Inf)
print(get_model_data(fit_interaction, type="pred", terms = c("absent", "cue")), n = Inf)
print(get_model_data(fit_interaction, type="pred", terms = c("absent", "category")), n = Inf)

print(plot_model(fit_interaction, type="pred", terms = c("category", "absent")), n = Inf)

###################################
## Assumptions for linear models ##
###################################

df$res<- residuals(fit_interaction)
df$res2 <- df$res^2 
  
linearity_plot <- df %>% 
ggplot(mapping = aes(x = res, y=search_time)) +
geom_point(alpha=.05) + labs(y= "Search time", x = "Residuals")
  
residual_plot <- df %>% 
  ggplot(mapping = aes(x = res)) +
    geom_histogram()  + labs(y= "Count", x = "Residuals")
  
qqplot <- df %>% 
  ggplot(mapping = aes(sample = search_time)) +
  stat_qq()  + labs(y= "Sample", x = "Theoretical")

# Save
pdf("output/figs/3-interaction/supp_fig21.pdf", width=10, height=3)
cowplot::plot_grid(plotlist = list(linearity_plot, residual_plot, qqplot), ncol=3, labels=c("A","B", "C"))
dev.off()

## Gamma
df$res<- residuals(fit_interaction_glmer)
df$res2 <- df$res^2

linearity_plot <- df %>% 
  ggplot(mapping = aes(x = res, y=search_time)) +
  geom_point(alpha=.05) + labs(y= "Search time", x = "Residuals")

residual_plot <- df %>% 
  ggplot(mapping = aes(x = res)) +
  geom_histogram() + labs(y= "Count", x = "Residuals")

qqplot <- df %>% 
  ggplot(mapping = aes(sample = res)) +
  stat_qq() + labs(y= "Sample", x = "Theoretical")

pdf("output/figs/3-interaction/supp_fig22.pdf", width=10, height=3)
cowplot::plot_grid(plotlist = list(linearity_plot, residual_plot, qqplot), ncol=3, labels=c("A","B", "C"))
dev.off()

###################
## Print tabular ##
###################

# Use this to check SDs, etc.

tab_model(fit_interaction, fit_interaction_glmer, show.reflvl = TRUE, show.se = TRUE, show.std=TRUE,show.icc=TRUE, dv.labels = c("Interaction model (linear)", "Interaction model (non-linear)"))